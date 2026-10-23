# SPONGE

SPONGE is a local-first visitor assistant prototype. It uses a camera to recognize people whose profiles have been explicitly saved, and lets you ask by voice about their relationship, reminders, and appointments.

The current prototype runs on a Windows PC. It uses OpenCV for face detection and matching, Vosk for local speech recognition, Ollama for conversational responses, and MongoDB for reminders, appointments, and conversation memory. Smart-glasses hardware is a future direction; this repository does not run on glasses yet.

## Features

- PC webcam or phone camera input through the web app
- Explicit voice-based enrollment of a visitor profile
- Profile reminders and relationship details
- Voice questions and spoken responses
- Appointment and meeting information
- Local processing with Ollama and MongoDB

## Requirements

- Windows 10 or 11
- Python 3.14 (the prototype was developed with Python 3.14.7)
- MongoDB Community Server running locally at `mongodb://127.0.0.1:27017`
- Ollama installed and running locally
- A webcam, or a phone on the same private Tailscale network for phone-camera use

The face-recognition and Vosk speech model files are included under `models/`. Ollama models are downloaded separately.

## Setup

1. Install MongoDB Community Server and Ollama.
2. In PowerShell, open the project folder and download the Ollama models:

   ```powershell
   ollama pull qwen3:4b
   ollama pull nomic-embed-text
   ```

3. Create the Python environment and install the packages:

   ```powershell
   py -3.14 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install --upgrade pip
   .\.venv\Scripts\python.exe -m pip install numpy opencv-python pymongo sounddevice vosk tzdata
   ```

4. Start the app by double-clicking `Open Visitor Assistant on Phone.bat`. The app runs on the PC at `http://127.0.0.1:8765`.

For phone camera and microphone access, both devices need to be signed in to the same Tailscale account. Start the app first, then run `Enable secure phone camera.bat` and open its HTTPS address on the phone.

## Local data and privacy

Saved face profiles are kept in the local `known_faces.json` file and are intentionally not included in this repository. MongoDB data is also stored locally. Do not commit face profiles or personal visitor data. Get consent before enrolling anyone, and make sure people can tell when the camera is active.

## Notes

- The assistant uses pretrained models; this project does not train an LLM from scratch.
- Keep the PC running while using the app. Ollama and MongoDB must be available locally.
- Use Tailscale for remote phone access; do not expose the app directly to the public internet.
