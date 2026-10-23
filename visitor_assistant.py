"""Local webcam face enrollment and recognition prototype.

This stores face embeddings (not webcam images) in known_faces.json.
Only enroll people who have agreed to participate.
"""

import json
import queue
import sys
import threading
import time
import textwrap
from pathlib import Path

import cv2
import numpy as np
from memory_store import generate_notification, speak_text


ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
DETECTOR_PATH = MODELS / "face_detection_yunet.onnx"
RECOGNIZER_PATH = MODELS / "face_recognition_sface.onnx"
PEOPLE_PATH = ROOT / "known_faces.json"
MATCH_THRESHOLD = 0.363
SAMPLES_TO_COLLECT = 15
VOICE_MODEL_PATH = MODELS / "vosk-model-small-en-us-0.15"


def load_models():
    for model_path in (DETECTOR_PATH, RECOGNIZER_PATH):
        if not model_path.exists():
            raise SystemExit(f"Model file not found: {model_path}")

    detector = cv2.FaceDetectorYN.create(
        str(DETECTOR_PATH), "", (320, 320), 0.9, 0.3, 5000
    )
    recognizer = cv2.FaceRecognizerSF.create(str(RECOGNIZER_PATH), "")
    return detector, recognizer


def open_camera():
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise SystemExit("Could not open the webcam. Check that it is connected and not in use.")
    return camera


def detected_faces(detector, frame):
    height, width = frame.shape[:2]
    detector.setInputSize((width, height))
    result = detector.detect(frame)
    return result[1]


def face_embedding(recognizer, frame, face):
    aligned_face = recognizer.alignCrop(frame, face)
    return recognizer.feature(aligned_face).astype(np.float32)


def start_voice_commands(state):
    """Listen for a small fixed set of commands; audio and command text stay local."""
    if not VOICE_MODEL_PATH.exists():
        raise SystemExit(
            f"Voice model not found: {VOICE_MODEL_PATH}\n"
            "Download and extract vosk-model-small-en-us-0.15 into the models folder."
        )

    try:
        import sounddevice as sd
        from vosk import KaldiRecognizer, Model
    except ImportError as exc:
        raise SystemExit(
            "Voice commands need vosk and sounddevice. Run: "
            "python -m pip install vosk sounddevice"
        ) from exc

    audio_queue = queue.Queue(maxsize=20)
    commands = [
        "pause", "pause recognition", "resume", "resume recognition",
        "repeat", "repeat reminder", "mute", "unmute", "stop", "stop assistant",
        "[unk]",
    ]
    model = Model(str(VOICE_MODEL_PATH))
    recognizer = KaldiRecognizer(model, 16000, json.dumps(commands))

    def audio_callback(indata, frames, timing, status):
        if status:
            print(f"Microphone: {status}")
        try:
            audio_queue.put_nowait(bytes(indata))
        except queue.Full:
            pass

    def listen():
        try:
            with sd.RawInputStream(
                samplerate=16000, blocksize=8000, dtype="int16", channels=1,
                callback=audio_callback,
            ):
                print("Voice commands ready: pause, resume, repeat reminder, mute, unmute, stop assistant.")
                while not state["stop"]:
                    try:
                        audio = audio_queue.get(timeout=0.25)
                    except queue.Empty:
                        continue
                    if recognizer.AcceptWaveform(audio):
                        phrase = json.loads(recognizer.Result()).get("text", "").strip()
                        if phrase:
                            print(f"Voice heard: {phrase}")
                        if phrase in {"pause", "pause recognition"}:
                            state["paused"] = True
                            print("Voice command: paused recognition.")
                        elif phrase in {"resume", "resume recognition"}:
                            state["paused"] = False
                            print("Voice command: resumed recognition.")
                        elif phrase in {"repeat", "repeat reminder"}:
                            message = state.get("last_reminder", "")
                            if message and not state["muted"]:
                                speak_text(message)
                            elif not message:
                                speak_text("There is no reminder to repeat yet.")
                        elif phrase == "mute":
                            state["muted"] = True
                            print("Voice command: spoken reminders muted.")
                        elif phrase == "unmute":
                            state["muted"] = False
                            print("Voice command: spoken reminders unmuted.")
                        elif phrase in {"stop", "stop assistant"}:
                            state["stop"] = True
                            print("Voice command: stopping the assistant.")
        except Exception as exc:
            state["voice_error"] = str(exc)
            print(f"Voice commands stopped: {exc}")

    worker = threading.Thread(target=listen, daemon=True)
    worker.start()
    return worker


def start_voice_transcriber(state, on_text):
    """Transcribe speech locally for explicit save, reminder, and question requests."""
    if not VOICE_MODEL_PATH.exists():
        raise SystemExit(
            f"Voice model not found: {VOICE_MODEL_PATH}\n"
            "Download and extract vosk-model-small-en-us-0.15 into the models folder."
        )
    try:
        import sounddevice as sd
        from vosk import KaldiRecognizer, Model
    except ImportError as exc:
        raise SystemExit(
            "Voice commands need vosk and sounddevice. Run: "
            "python -m pip install vosk sounddevice"
        ) from exc

    audio_queue = queue.Queue(maxsize=24)
    model = Model(str(VOICE_MODEL_PATH))
    recognizer = KaldiRecognizer(model, 16000)

    def audio_callback(indata, frames, timing, status):
        if state.get("speaking"):
            return
        try:
            audio_queue.put_nowait(bytes(indata))
        except queue.Full:
            pass

    def listen():
        try:
            with sd.RawInputStream(
                samplerate=16000, blocksize=8000, dtype="int16", channels=1,
                callback=audio_callback,
            ):
                state["voice_ready"] = True
                print("Local speech recognition ready.")
                was_speaking = False
                while not state["stop"]:
                    if state.get("speaking"):
                        while not audio_queue.empty():
                            try:
                                audio_queue.get_nowait()
                            except queue.Empty:
                                break
                        if not was_speaking:
                            recognizer.Reset()
                            was_speaking = True
                        time.sleep(0.05)
                        continue
                    was_speaking = False
                    try:
                        audio = audio_queue.get(timeout=0.25)
                    except queue.Empty:
                        continue
                    if recognizer.AcceptWaveform(audio):
                        phrase = json.loads(recognizer.Result()).get("text", "").strip()
                        if phrase:
                            on_text(phrase)
        except Exception as exc:
            state["voice_error"] = str(exc)
            state["voice_ready"] = False
            print(f"Local speech recognition stopped: {exc}")

    worker = threading.Thread(target=listen, daemon=True)
    worker.start()
    return worker


def load_people():
    if not PEOPLE_PATH.exists():
        return {}
    try:
        return json.loads(PEOPLE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise SystemExit(f"Could not read {PEOPLE_PATH.name}: {exc}")


def enroll(name, detector, recognizer):
    camera = open_camera()
    samples = []
    next_sample_at = 0.0
    print(f"Enrolling {name}. Look toward the camera; press Q to cancel.")

    try:
        while len(samples) < SAMPLES_TO_COLLECT:
            ok, frame = camera.read()
            if not ok:
                continue

            faces = detected_faces(detector, frame)
            now = time.monotonic()
            if faces is not None and len(faces) == 1 and now >= next_sample_at:
                samples.append(face_embedding(recognizer, frame, faces[0]))
                next_sample_at = now + 0.3

            message = f"Samples: {len(samples)}/{SAMPLES_TO_COLLECT}  |  Q to cancel"
            cv2.putText(frame, message, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (40, 220, 40), 2, cv2.LINE_AA)
            if faces is not None and len(faces) > 1:
                cv2.putText(frame, "One person at a time", (12, 62),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                            cv2.LINE_AA)
            cv2.imshow("Enroll person", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Enrollment cancelled.")
                return
    finally:
        camera.release()
        cv2.destroyAllWindows()

    mean_embedding = np.mean(np.stack(samples), axis=0)
    mean_embedding /= max(float(np.linalg.norm(mean_embedding)), 1e-12)
    people = load_people()
    people[name] = mean_embedding.reshape(-1).tolist()
    PEOPLE_PATH.write_text(json.dumps(people, indent=2), encoding="utf-8")
    print(f"Saved {name}'s face template to {PEOPLE_PATH.name} (no camera images saved).")


def recognize(detector, recognizer):
    people = load_people()
    if not people:
        raise SystemExit('No enrolled people yet. First run: python visitor_assistant.py enroll "Name"')

    camera = open_camera()
    print("Recognition is running locally. Press Q in the camera window to stop.")
    last_seen = {}
    notification_state = {"text": "", "until": 0.0}
    notification_busy = set()
    runtime_state = {
        "paused": False, "muted": False, "stop": False,
        "last_reminder": "", "voice_error": "",
    }
    window_name = "Visitor Assistant — local recognition"

    voice_worker = None

    def request_notification(name):
        def work():
            try:
                message = generate_notification(name)
            except Exception as exc:
                print(f"Context reminder unavailable: {exc}")
                message = f"{name} is here."
            notification_state["text"] = message
            notification_state["until"] = time.monotonic() + 8
            runtime_state["last_reminder"] = message
            notification_busy.discard(name)
            print(f"Reminder: {message}")
            if not runtime_state["muted"]:
                try:
                    speak_text(message)
                except Exception as exc:
                    print(f"Speech unavailable: {exc}")

        threading.Thread(target=work, daemon=True).start()

    try:
        voice_worker = start_voice_commands(runtime_state)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        while True:
            if runtime_state["stop"]:
                break
            ok, frame = camera.read()
            if not ok:
                continue

            faces = None if runtime_state["paused"] else detected_faces(detector, frame)
            if faces is not None:
                for face in faces:
                    x, y, width, height = [int(value) for value in face[:4]]
                    current = face_embedding(recognizer, frame, face)
                    best_name, best_score = "Unknown", -1.0

                    for name, saved_embedding in people.items():
                        saved = np.asarray(saved_embedding, dtype=np.float32).reshape(1, -1)
                        score = float(recognizer.match(
                            current, saved, cv2.FaceRecognizerSF_FR_COSINE
                        ))
                        if score > best_score:
                            best_name, best_score = name, score

                    if best_score < MATCH_THRESHOLD:
                        best_name = "Unknown"
                    else:
                        now = time.monotonic()
                        if now - last_seen.get(best_name, 0.0) > 10 and best_name not in notification_busy:
                            notification_busy.add(best_name)
                            notification_state["text"] = f"{best_name} is here."
                            notification_state["until"] = now + 15
                            print(f"Recognized: {best_name}")
                            request_notification(best_name)
                        last_seen[best_name] = now
                    color = (40, 220, 40) if best_name != "Unknown" else (0, 165, 255)
                    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                    label = f"{best_name} ({best_score:.2f})" if best_name != "Unknown" else best_name
                    cv2.putText(frame, label, (x, max(24, y - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

            if notification_state["text"] and time.monotonic() < notification_state["until"]:
                lines = textwrap.wrap(notification_state["text"], width=58)[:2]
                for line_index, line in enumerate(lines):
                    cv2.putText(frame, line, (12, frame.shape[0] - 18 - 28 * (len(lines) - 1 - line_index)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        runtime_state["stop"] = True
        if voice_worker is not None and voice_worker.is_alive():
            voice_worker.join(timeout=1.0)
        camera.release()
        cv2.destroyAllWindows()


def main():
    if len(sys.argv) < 2 or sys.argv[1].lower() not in {"enroll", "run", "profile", "remember"}:
        raise SystemExit(
            'Usage:\n  python visitor_assistant.py enroll "Person Name"\n'
            "  python visitor_assistant.py run\n"
            '  python visitor_assistant.py profile "Person Name" "relationship"\n'
            '  python visitor_assistant.py remember "Person Name" "short useful note"'
        )

    mode = sys.argv[1].lower()
    if mode in {"profile", "remember"}:
        from memory_store import main as memory_main
        sys.argv[1] = mode
        memory_main()
        return

    detector, recognizer = load_models()
    if mode == "enroll":
        name = " ".join(sys.argv[2:]).strip()
        if not name:
            raise SystemExit('Please provide a name, for example: enroll "Maya"')
        enroll(name, detector, recognizer)
    else:
        recognize(detector, recognizer)


if __name__ == "__main__":
    main()
