"""A simple local desktop interface for Visitor Assistant."""

import base64
import json
import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox

import cv2
import numpy as np

from memory_store import (
    answer_person_question,
    connect_database,
    answer_meeting_question,
    format_meeting_answer,
    generate_notification,
    get_upcoming_meetings,
    interpret_voice_utterance,
    save_memory,
    save_appointment,
    save_profile,
    speak_text,
)
from visitor_assistant import (
    MATCH_THRESHOLD,
    PEOPLE_PATH,
    SAMPLES_TO_COLLECT,
    face_embedding,
    load_models,
    load_people,
    open_camera,
    start_voice_transcriber,
)


BG = "#F3F6FA"
CARD = "#FFFFFF"
INK = "#182334"
SUBTLE = "#697789"
BLUE = "#2563EB"
GREEN = "#138A58"
LINE = "#E2E8F0"


class VisitorAssistantApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Visitor Assistant")
        self.root.geometry("1240x800")
        self.root.minsize(1000, 700)
        self.root.configure(bg=BG)

        self.events = queue.Queue()
        self.frames = queue.Queue(maxsize=1)
        self.state = {
            "running": False,
            "stop": False,
            "paused": False,
            "muted": False,
            "speaking": False,
            "voice_ready": False,
            "current_visitor": "Waiting for camera",
            "relationship": "",
            "last_reminder": "Start the camera to see local visitor reminders here.",
            "candidate_samples": [],
            "candidate_embedding": None,
            "candidate_last_seen": 0.0,
            "candidate_sampling_notified": False,
            "candidate_ready_notified": False,
        }
        self.model_data = None
        self.photo = None
        self.guide_window = None
        self.speech_lock = threading.Lock()

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._poll_events)

    def _build_ui(self):
        header = tk.Frame(self.root, bg=CARD, height=86, highlightbackground=LINE, highlightthickness=1)
        header.pack(fill="x")
        header.pack_propagate(False)
        title_group = tk.Frame(header, bg=CARD)
        title_group.pack(side="left", padx=28, pady=15)
        tk.Label(title_group, text="Visitor Assistant", bg=CARD, fg=INK,
                 font=("Segoe UI", 21, "bold")).pack(anchor="w")
        tk.Label(title_group, text="A little context, right when you need it.", bg=CARD, fg=SUBTLE,
                 font=("Segoe UI", 10)).pack(anchor="w", pady=(1, 0))
        self.local_badge = tk.Label(header, text="●  LOCAL & PRIVATE", bg="#EAF7F0", fg=GREEN,
                                    font=("Segoe UI", 9, "bold"), padx=13, pady=8)
        self.local_badge.pack(side="right", padx=28)

        content = tk.Frame(self.root, bg=BG)
        content.pack(fill="both", expand=True, padx=22, pady=18)
        content.grid_columnconfigure(0, weight=7, minsize=550)
        content.grid_columnconfigure(1, weight=3, minsize=280)
        content.grid_rowconfigure(0, weight=1)

        camera_card = self._card(content)
        camera_card.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        camera_card.grid_rowconfigure(1, weight=1)
        camera_card.grid_columnconfigure(0, weight=1)
        camera_header = tk.Frame(camera_card, bg=CARD)
        camera_header.grid(row=0, column=0, sticky="ew", padx=19, pady=(16, 10))
        tk.Label(camera_header, text="Live camera", bg=CARD, fg=INK,
                 font=("Segoe UI", 13, "bold")).pack(side="left")
        self.camera_status = tk.Label(camera_header, text="●  Camera off", bg=CARD, fg=SUBTLE,
                                      font=("Segoe UI", 9))
        self.camera_status.pack(side="right")
        self.preview = tk.Label(camera_card, text="Your camera preview will appear here",
                                bg="#111827", fg="#D8E1EF", font=("Segoe UI", 12),
                                compound="center")
        self.preview.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))

        right = tk.Frame(content, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)

        visitor_card = self._card(right)
        visitor_card.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        tk.Label(visitor_card, text="IN VIEW", bg=CARD, fg=SUBTLE,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=18, pady=(17, 9))
        self.visitor_name = tk.Label(visitor_card, text="Waiting for camera", bg=CARD, fg=INK,
                                     font=("Segoe UI", 20, "bold"), wraplength=300, justify="left")
        self.visitor_name.pack(anchor="w", padx=18)
        self.visitor_relation = tk.Label(visitor_card, text="", bg=CARD, fg=SUBTLE,
                                         font=("Segoe UI", 10), wraplength=300, justify="left")
        self.visitor_relation.pack(anchor="w", padx=18, pady=(4, 18))

        reminder_card = self._card(right)
        reminder_card.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        tk.Label(reminder_card, text="REMINDER", bg=CARD, fg=SUBTLE,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=18, pady=(17, 8))
        self.reminder_text = tk.Label(reminder_card, text=self.state["last_reminder"], bg=CARD, fg=INK,
                                      font=("Segoe UI", 11), wraplength=300, justify="left",
                                      anchor="nw")
        self.reminder_text.pack(fill="x", padx=18, pady=(0, 18))

        voice_card = self._card(right)
        voice_card.grid(row=2, column=0, sticky="ew")
        tk.Label(voice_card, text="VOICE CONTROLS", bg=CARD, fg=SUBTLE,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=18, pady=(17, 7))
        tk.Label(voice_card, text="Say a save instruction, reminder, or question.\nAlso: pause, resume, repeat, mute, unmute, stop.",
                 bg=CARD, fg=INK, font=("Segoe UI", 10), justify="left").pack(anchor="w", padx=18)
        self.voice_status = tk.Label(voice_card, text="Voice commands start with the camera.",
                                     bg=CARD, fg=SUBTLE, font=("Segoe UI", 9), wraplength=300,
                                     justify="left")
        self.voice_status.pack(anchor="w", padx=18, pady=(9, 17))
        self.heard_text = tk.Label(voice_card, text="", bg=CARD, fg=BLUE,
                                   font=("Segoe UI", 9), wraplength=300, justify="left")
        self.heard_text.pack(anchor="w", padx=18, pady=(0, 14))

        controls = tk.Frame(self.root, bg=BG)
        controls.pack(fill="x", padx=22, pady=(0, 18))
        self.start_button = self._button(controls, "Start camera", self.start, BLUE, "white")
        self.start_button.pack(side="left", padx=(0, 9))
        self.pause_button = self._button(controls, "Pause recognition", self.toggle_pause, CARD, INK, outline=True)
        self.pause_button.pack(side="left", padx=(0, 9))
        self.mute_button = self._button(controls, "Mute speech", self.toggle_mute, CARD, INK, outline=True)
        self.mute_button.pack(side="left", padx=(0, 9))
        self._button(controls, "Voice examples", self.open_guide, CARD, INK, outline=True).pack(side="right")

    @staticmethod
    def _card(parent):
        return tk.Frame(parent, bg=CARD, highlightbackground=LINE, highlightthickness=1)

    @staticmethod
    def _button(parent, text, command, bg, fg, outline=False):
        button = tk.Button(
            parent, text=text, command=command, bg=bg, fg=fg,
            activebackground=("#1D4ED8" if bg == BLUE else "#EEF2F7"),
            activeforeground=("white" if bg == BLUE else INK),
            relief="flat", bd=0, padx=17, pady=11, cursor="hand2",
            font=("Segoe UI", 10, "bold"),
        )
        if outline:
            button.configure(highlightbackground=LINE, highlightthickness=1)
        return button

    def start(self):
        if self.state["running"]:
            return
        self.state.update(running=True, stop=False, paused=False, muted=False, voice_error="", voice_ready=False)
        self.start_button.configure(text="Stop camera", command=self.stop)
        self.camera_status.configure(text="●  Starting…", fg="#C27912")
        self.voice_status.configure(text="Starting local voice recognition…")
        threading.Thread(target=self._camera_worker, daemon=True).start()

    def stop(self):
        self.state["stop"] = True
        self.camera_status.configure(text="●  Stopping…", fg="#C27912")

    def toggle_pause(self):
        if not self.state["running"]:
            return
        self.state["paused"] = not self.state["paused"]
        self.pause_button.configure(text="Resume recognition" if self.state["paused"] else "Pause recognition")

    def toggle_mute(self):
        if not self.state["running"]:
            return
        self.state["muted"] = not self.state["muted"]
        self.mute_button.configure(text="Unmute speech" if self.state["muted"] else "Mute speech")

    def _camera_worker(self):
        camera = None
        voice_worker = None
        busy = set()
        last_seen = {}
        people = {}
        last_people_check = 0.0
        try:
            if self.model_data is None:
                self.model_data = load_models()
            detector, recognizer = self.model_data
            people = load_people()
            camera = open_camera()
            voice_worker = start_voice_transcriber(
                self.state, lambda phrase: self.events.put(("transcript", phrase))
            )
            failed_reads = 0
            camera_ready = False

            while not self.state["stop"]:
                ok, frame = camera.read()
                if not ok:
                    failed_reads += 1
                    if failed_reads >= 50:
                        raise RuntimeError(
                            "The camera opened, but no video frames arrived. Check Windows camera permission "
                            "and close other apps that may be using the camera."
                        )
                    time.sleep(0.02)
                    continue
                failed_reads = 0
                if not camera_ready:
                    camera_ready = True
                    self.events.put(("running", ""))
                now = time.monotonic()
                if now - last_people_check > 0.5:
                    if self.state.pop("people_changed", False):
                        people = load_people()
                    last_people_check = now
                faces = None if self.state["paused"] else self._detect(detector, frame)
                names = []
                unknown_count = 0
                if faces is not None:
                    for face in faces:
                        x, y, width, height = [int(value) for value in face[:4]]
                        current = face_embedding(recognizer, frame, face)
                        best_name, best_score = "Unknown", -1.0
                        for name, saved_embedding in people.items():
                            saved = np.asarray(saved_embedding, dtype=np.float32).reshape(1, -1)
                            score = float(recognizer.match(current, saved, cv2.FaceRecognizerSF_FR_COSINE))
                            if score > best_score:
                                best_name, best_score = name, score
                        if best_score < MATCH_THRESHOLD:
                            best_name = "Unknown"
                            unknown_count += 1
                        else:
                            names.append(best_name)
                            now = time.monotonic()
                            if now - last_seen.get(best_name, 0.0) > 10 and best_name not in busy:
                                busy.add(best_name)
                                self._request_reminder(best_name, busy)
                            last_seen[best_name] = now

                        color = (56, 190, 120) if best_name != "Unknown" else (25, 165, 245)
                        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                        label = best_name if best_name == "Unknown" else f"{best_name}  {best_score:.2f}"
                        cv2.putText(frame, label, (x, max(25, y - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.65, color, 2, cv2.LINE_AA)

                self._collect_candidate(faces, frame, recognizer, unknown_count)
                if names:
                    # A known visitor must not inherit a pending unknown face sample.
                    self.state["candidate_samples"] = []
                    self.state["candidate_embedding"] = None
                    self.state["candidate_sampling_notified"] = False
                    self.state["candidate_ready_notified"] = False
                    self.state["current_visitor"] = names[0] if len(names) == 1 else f"{len(names)} recognized visitors"
                    self._refresh_relationship(names[0])
                elif unknown_count:
                    self._relationship_for = None
                    if faces is not None and len(faces) > 1:
                        self.state["candidate_samples"] = []
                        self.state["candidate_embedding"] = None
                        self.state["candidate_sampling_notified"] = False
                        self.state["candidate_ready_notified"] = False
                    self.state["current_visitor"] = "Unrecognized visitor"
                    count = len(self.state["candidate_samples"])
                    self.state["relationship"] = (
                        f"{count}/{SAMPLES_TO_COLLECT} temporary face samples · not saved"
                    )
                elif not self.state["paused"]:
                    age = time.monotonic() - self.state.get("candidate_last_seen", 0.0)
                    if self.state["candidate_samples"] and age <= 60:
                        self.state["current_visitor"] = "Recent unrecognized visitor"
                        count = len(self.state["candidate_samples"])
                        self.state["relationship"] = (
                            f"{count}/{SAMPLES_TO_COLLECT} temporary face samples · not saved"
                        )
                    else:
                        self.state["candidate_samples"] = []
                        self.state["candidate_embedding"] = None
                        self.state["candidate_sampling_notified"] = False
                        self.state["candidate_ready_notified"] = False
                        self.state["current_visitor"] = "No known visitor"
                        self.state["relationship"] = ""
                        self._relationship_for = None
                self._draw_status(frame)
                try:
                    self.frames.put_nowait(frame)
                except queue.Full:
                    try:
                        self.frames.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self.frames.put_nowait(frame)
                    except queue.Full:
                        pass
                time.sleep(0.01)
        except Exception as exc:
            self.events.put(("error", str(exc)))
        finally:
            self.state["stop"] = True
            if voice_worker is not None and voice_worker.is_alive():
                voice_worker.join(timeout=1)
            if camera is not None:
                camera.release()
            self.events.put(("stopped", ""))

    @staticmethod
    def _detect(detector, frame):
        height, width = frame.shape[:2]
        detector.setInputSize((width, height))
        return detector.detect(frame)[1]

    def _request_reminder(self, name, busy):
        def generate():
            try:
                message = generate_notification(name)
                client, database = connect_database()
                try:
                    profile = database.people.find_one({"name": name}) or {}
                finally:
                    client.close()
                relationship = profile.get("relationship", "")
            except Exception as exc:
                message = f"{name} is here."
                relationship = ""
                self.events.put(("notice", f"Using a simple reminder ({exc})"))
            self.state["last_reminder"] = message
            self.state["current_visitor"] = name
            self.state["relationship"] = relationship
            busy.discard(name)
            self.events.put(("reminder", message))
            if not self.state["muted"]:
                self._speak(message)
        threading.Thread(target=generate, daemon=True).start()

    def _speak(self, text):
        if self.state["muted"] or not text:
            return
        with self.speech_lock:
            if self.state["muted"]:
                return
            self.state["speaking"] = True
            try:
                speak_text(text)
            except Exception as exc:
                self.events.put(("notice", f"Speech unavailable: {exc}"))
            finally:
                self.state["speaking"] = False

    def _process_voice_utterance(self, transcript):
        phrase = " ".join(transcript.lower().strip().split()).strip(" .,!?;:")
        controls = {
            "pause": "pause", "pause recognition": "pause",
            "resume": "resume", "resume recognition": "resume",
            "mute": "mute", "mute speech": "mute",
            "unmute": "unmute", "unmute speech": "unmute",
            "repeat": "repeat", "repeat reminder": "repeat",
            "stop": "stop", "stop assistant": "stop",
        }
        if phrase in controls:
            command = controls[phrase]
            if command == "pause":
                self.state["paused"] = True
                self.events.put(("assistant_response", "Recognition paused."))
            elif command == "resume":
                self.state["paused"] = False
                self.events.put(("assistant_response", "Recognition resumed."))
            elif command == "mute":
                self.state["muted"] = True
                self.events.put(("assistant_response", "Speech muted."))
            elif command == "unmute":
                self.state["muted"] = False
                self.events.put(("assistant_response", "Speech unmuted."))
                threading.Thread(target=self._speak, args=("Speech unmuted.",), daemon=True).start()
            elif command == "repeat":
                reminder = self.state.get("last_reminder", "There is no saved reminder yet.")
                threading.Thread(target=self._speak, args=(reminder,), daemon=True).start()
            else:
                self.state["stop"] = True
                self.events.put(("assistant_response", "Stopping the assistant."))
            return

        self.events.put(("heard", transcript))
        threading.Thread(target=self._interpret_and_handle, args=(transcript,), daemon=True).start()

    def _interpret_and_handle(self, transcript):
        try:
            known_names = list(load_people().keys())
            current_person = self.state.get("current_visitor", "")
            if current_person in {"Unrecognized visitor", "Recent unrecognized visitor", "No known visitor", "Waiting for camera"}:
                current_person = ""
            request = interpret_voice_utterance(transcript, known_names, current_person)
            intent = request.get("intent")
            if intent == "ignore":
                command_words = ("save", "remember", "reminder", "register", "meeting", "tell me", "who is")
                if any(word in transcript.casefold() for word in command_words):
                    self.events.put((
                        "notice",
                        f"I heard: {transcript}. Try: ‘Save this person as Alex, he is my friend’ or ‘Remember that Alex is bringing the documents.’",
                    ))
                return
            if intent == "save_person":
                name = request.get("name", "").strip()
                relationship = request.get("relationship", "").strip()
                candidate = self.state.get("candidate_embedding")
                if not name:
                    self._respond("I didn't catch the person's name. Please say it again.")
                    return
                if not candidate:
                    self._respond("I don't have a temporary face sample yet. Let one face stay in view, then say the save instruction again.")
                    return
                save_profile(name, relationship)
                people = load_people()
                people[name] = candidate
                temporary_path = PEOPLE_PATH.with_suffix(".tmp")
                temporary_path.write_text(json.dumps(people, indent=2), encoding="utf-8")
                temporary_path.replace(PEOPLE_PATH)
                self.state["candidate_samples"] = []
                self.state["candidate_embedding"] = None
                self.state["candidate_sampling_notified"] = False
                self.state["candidate_ready_notified"] = False
                self.state["people_changed"] = True
                self.state["current_visitor"] = name
                self.state["relationship"] = relationship
                relation_text = f", your {relationship}" if relationship else ""
                self._respond(f"Saved {name}{relation_text}. I'll recognize them next time.")
                return
            if intent == "remember":
                memory = request.get("memory", "").strip()
                if not memory:
                    self._respond("I didn't catch the reminder. Please say, 'Remember that…' and repeat it.")
                    return
                name = request.get("name", "").strip() or current_person or "General"
                appointment_request = request.get("is_appointment") or any(
                    word in transcript.lower() for word in ("meeting", "appointment")
                )
                if appointment_request:
                    with_person = request.get("meeting_with", "").strip()
                    date_iso = request.get("date_iso", "").strip()
                    time_24h = request.get("time_24h", "").strip()
                    missing = []
                    if not with_person:
                        missing.append("who it is with")
                    if not date_iso:
                        missing.append("the date")
                    if not time_24h:
                        missing.append("the time")
                    if missing:
                        self._respond("I can book that. Please tell me " + ", ".join(missing) + ".")
                        return
                    save_appointment(memory, with_person, date_iso, time_24h)
                    date_label = request.get("date_text", "").strip() or date_iso
                    time_label = request.get("time_text", "").strip() or time_24h
                    response = f"Booked {memory} with {with_person} for {date_label} at {time_label}."
                else:
                    save_memory(name, memory)
                    response = f"I saved that reminder for {name}."
                self._respond(response)
                return
            if intent == "ask_meetings":
                self._respond(answer_meeting_question(
                    transcript, request.get("date_scope", ""), request.get("meeting_with", "")
                ))
                return
            if intent == "ask_person":
                name = request.get("name", "").strip() or current_person
                answer = answer_person_question(name, transcript)
                self._respond(answer)
        except Exception as exc:
            self._respond(f"I couldn't complete that request: {exc}")

    def _respond(self, text):
        self.state["last_reminder"] = text
        self.events.put(("assistant_response", text))
        self._speak(text)

    def _refresh_relationship(self, name):
        if getattr(self, "_relationship_for", None) == name:
            return
        self._relationship_for = name
        def lookup():
            try:
                client, database = connect_database()
                try:
                    profile = database.people.find_one({"name": name}) or {}
                finally:
                    client.close()
                self.state["relationship"] = profile.get("relationship", "")
            except Exception:
                self.state["relationship"] = ""
        threading.Thread(target=lookup, daemon=True).start()

    def _collect_candidate(self, faces, frame, recognizer, unknown_count):
        if faces is None or len(faces) != 1 or unknown_count != 1:
            return
        samples = self.state["candidate_samples"]
        now = time.monotonic()
        if samples and now - self.state.get("candidate_last_seen", 0.0) > 3:
            samples.clear()
            self.state["candidate_embedding"] = None
            self.state["candidate_sampling_notified"] = False
            self.state["candidate_ready_notified"] = False
        if now < self.state.get("next_candidate_sample", 0):
            return
        sample = face_embedding(recognizer, frame, faces[0])
        if samples:
            sample_vector = sample.reshape(-1)
            prior = np.mean(np.stack(samples), axis=0).reshape(-1)
            similarity = float(
                np.dot(sample_vector, prior)
                / max(float(np.linalg.norm(sample_vector) * np.linalg.norm(prior)), 1e-12)
            )
            if similarity < MATCH_THRESHOLD:
                samples.clear()
                self.state["candidate_embedding"] = None
                self.state["candidate_sampling_notified"] = False
                self.state["candidate_ready_notified"] = False
        samples.append(sample)
        if len(samples) > SAMPLES_TO_COLLECT:
            del samples[0]
        self.state["next_candidate_sample"] = now + 0.3
        self.state["candidate_last_seen"] = now
        if not self.state.get("candidate_sampling_notified"):
            self.state["candidate_sampling_notified"] = True
            self.events.put(("sampling", f"Unfamiliar face detected. Sampling started (1/{SAMPLES_TO_COLLECT}); keep one face in view."))
        if len(samples) >= SAMPLES_TO_COLLECT:
            average = np.mean(np.stack(samples), axis=0)
            average /= max(float(np.linalg.norm(average)), 1e-12)
            self.state["candidate_embedding"] = average.reshape(-1).tolist()
            if not self.state.get("candidate_ready_notified"):
                self.state["candidate_ready_notified"] = True
                self.events.put(("sample_ready", "Face sample ready. You can save now by saying: Save this person as Alex, he is my friend."))

    def _draw_status(self, frame):
        status = "Recognition paused" if self.state["paused"] else "Listening for visitors"
        cv2.putText(frame, status, (16, frame.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (240, 245, 250), 2, cv2.LINE_AA)

    def _poll_events(self):
        try:
            while True:
                event, payload = self.events.get_nowait()
                if event == "running":
                    self.camera_status.configure(text="●  Camera on", fg=GREEN)
                    self.voice_status.configure(text="Listening locally for voice commands.")
                elif event == "reminder":
                    self.reminder_text.configure(text=payload)
                elif event == "transcript":
                    self._process_voice_utterance(payload)
                elif event == "heard":
                    self.heard_text.configure(text=f"Heard: {payload}")
                elif event == "assistant_response":
                    self.reminder_text.configure(text=payload)
                    self.voice_status.configure(text="Done. You can ask another question or give a save instruction.")
                elif event == "notice":
                    self.voice_status.configure(text=payload)
                    self.state["notice_until"] = time.monotonic() + 6
                elif event == "sampling":
                    self.voice_status.configure(text=payload)
                    self.state["notice_until"] = time.monotonic() + 6
                    threading.Thread(target=self._speak, args=(payload,), daemon=True).start()
                elif event == "sample_ready":
                    self.visitor_relation.configure(text=f"{SAMPLES_TO_COLLECT}/{SAMPLES_TO_COLLECT} samples · ready to save")
                    self.voice_status.configure(text=payload)
                    self.state["notice_until"] = time.monotonic() + 10
                    threading.Thread(target=self._speak, args=(payload,), daemon=True).start()
                elif event == "error":
                    messagebox.showerror("Could not start the assistant", payload)
                elif event == "stopped":
                    self.state["running"] = False
                    self.start_button.configure(text="Start camera", command=self.start)
                    self.camera_status.configure(text="●  Camera off", fg=SUBTLE)
                    self.voice_status.configure(text="Camera stopped. Start it to listen again.")
        except queue.Empty:
            pass

        try:
            frame = self.frames.get_nowait()
            h, w = frame.shape[:2]
            max_w, max_h = 840, 590
            scale = min(max_w / w, max_h / h)
            if scale < 1:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            encoded_ok, png = cv2.imencode(".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            if not encoded_ok:
                raise RuntimeError("OpenCV could not encode the camera preview.")
            encoded = base64.b64encode(png.tobytes()).decode("ascii")
            self.photo = tk.PhotoImage(data=encoded, format="png")
            self.preview.configure(image=self.photo, text="")
        except queue.Empty:
            pass
        except Exception as exc:
            self.preview.configure(image="", text="Camera preview could not be displayed")
            self.camera_status.configure(text="●  Preview error", fg="#B54708")
            self.voice_status.configure(text=str(exc))

        self.visitor_name.configure(text=self.state["current_visitor"])
        relationship = self.state["relationship"]
        self.visitor_relation.configure(text=relationship if relationship else "")
        if self.state["running"]:
            self.pause_button.configure(text="Resume recognition" if self.state["paused"] else "Pause recognition")
            self.mute_button.configure(text="Unmute speech" if self.state["muted"] else "Mute speech")
            if self.state.get("voice_ready") and not self.state.get("speaking"):
                notice_active = time.monotonic() < self.state.get("notice_until", 0)
                if "Done." not in self.voice_status.cget("text") and not notice_active:
                    self.voice_status.configure(text="Listening locally. Say a save instruction, reminder, or question.")
            voice_error = self.state.get("voice_error", "")
            if voice_error:
                self.voice_status.configure(text=f"Microphone unavailable: {voice_error}")
        self.root.after(40, self._poll_events)

    def open_guide(self):
        if self.guide_window and self.guide_window.winfo_exists():
            self.guide_window.lift()
            return
        window = tk.Toplevel(self.root)
        self.guide_window = window
        window.title("Voice examples")
        window.geometry("540x480")
        window.resizable(False, False)
        window.configure(bg=BG)
        card = self._card(window)
        card.pack(fill="both", expand=True, padx=18, pady=18)
        tk.Label(card, text="Use your voice", bg=CARD, fg=INK,
                 font=("Segoe UI", 18, "bold")).pack(anchor="w", padx=20, pady=(20, 4))
        tk.Label(card, text="Speak these instructions in English. Face samples remain temporary until you explicitly save a person.",
                 bg=CARD, fg=SUBTLE, font=("Segoe UI", 10), wraplength=480,
                 justify="left").pack(anchor="w", padx=20, pady=(0, 17))
        examples = [
            ("Save a face", "Save him as Alex. He is my friend."),
            ("Save a reminder", "Remember that Alex is bringing the documents tomorrow."),
            ("Save a meeting", "Remember I have a meeting with Alex tomorrow at 3 PM."),
            ("Ask about someone", "Tell me about Alex."),
            ("Ask about meetings", "Is there any meeting?"),
            ("Controls", "Pause recognition · Resume · Repeat reminder · Mute · Unmute · Stop assistant"),
        ]
        for heading, phrase in examples:
            row = tk.Frame(card, bg=CARD)
            row.pack(fill="x", padx=20, pady=6)
            tk.Label(row, text=heading, bg=CARD, fg=INK, width=19, anchor="w",
                     font=("Segoe UI", 9, "bold")).pack(side="left")
            tk.Label(row, text=phrase, bg=CARD, fg=SUBTLE, wraplength=300,
                     justify="left", anchor="w", font=("Segoe UI", 9)).pack(side="left", fill="x", expand=True)
        tk.Label(card, text="Speak one command at a time. For a new person, keep one unknown face in view until the sample count reaches 15/15, then say: ‘Save this person as Alex, he is my friend.’ For a reminder, say: ‘Remember that Alex is bringing the documents tomorrow.’",
                 bg=CARD, fg=BLUE, font=("Segoe UI", 9), wraplength=480,
                 justify="left").pack(anchor="w", padx=20, pady=(15, 18))

    def _on_close(self):
        self.state["stop"] = True
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = VisitorAssistantApp(root)
    root.mainloop()
