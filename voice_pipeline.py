"""voice_pipeline.py — Complete voice assistant pipeline (Node 0 + Node 2)."""
import sys
import json
import time
import socket
import subprocess
import tempfile
import os
from datetime import datetime
from pathlib import Path
import numpy as np

from voice_sender import record_utterance, save_wav, discover_node2, RECORDINGS_DIR
from test_pipeline import call_ollama, _build_prompt, HISTORY, MAX_HISTORY
from executor import execute
import requests

# ────────── Config ──────────
WAKE_WORD = "wife"
PIPER_BIN = os.path.expanduser("~/piper/piper/piper")
PIPER_VOICE = os.path.expanduser("~/piper/piper/en_US-lessac-medium.onnx")

def speak(text: str):
    """Generate audio with Piper and play it."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        out = f.name
    try:
        subprocess.run(
            [PIPER_BIN, "--model", PIPER_VOICE, "--output_file", out],
            input=text.encode(), check=True, capture_output=True,
        )
        subprocess.run(["aplay", "-q", out], check=True)
    finally:
        os.unlink(out)

def main():
    print("🔍 Searching for Node 2 (Dell)...")
    node2 = discover_node2()
    if not node2:
        print("❌ Node 2 not found. Is the Dell server running?")
        sys.exit(1)
    print(f"✅ Connected to Node 2: http://{node2}")
    
    print(f"\n🎙️  Say '{WAKE_WORD} <command>' (e.g., '{WAKE_WORD} open youtube')")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            # 1. Record audio via VAD
            audio = record_utterance()
            duration = len(audio) / 16000
            if duration < 0.5:
                continue
                
            # 2. Save temp file & send to Dell
            ts = datetime.now().strftime("%H%M%S")
            path = RECORDINGS_DIR / f"tmp_{ts}.wav"
            save_wav(audio, path)
            
            with open(path, "rb") as fh:
                r = requests.post(f"http://{node2}/transcribe", files={"file": fh}, timeout=15)
            os.unlink(path)
            
            text = r.json().get("text", "").strip()
            if not text:
                continue
                
            # Filter out Whisper background noise hallucinations
            NOISE_TAGS = ["[", "(", "music playing", "blank_audio", "dramatic music", "crowd talking"]
            if any(tag in text.lower() for tag in NOISE_TAGS):
                continue 
                
            print(f"👂 Heard: {text}")

            # 3. Check for wake word (catches phonetic mistakes)
            low = text.lower().strip()
            
            # Whisper often mistakes "wife" for these words
            WAKE_WORDS = [
                "wife", "life", "wipe", "white", "why", "bye"
                "wifi", "wine", "wide", "five", "knife", "wise"
            ]
            
            best_idx = -1
            best_word_len = 0
            
            for w in WAKE_WORDS:
                idx = low.find(w)
                if idx != -1 and (best_idx == -1 or idx < best_idx):
                    best_idx = idx
                    best_word_len = len(w)
            
            if best_idx == -1:
                continue
                
            command = low[best_idx + best_word_len:].strip(" .,!?")
                
            if not command:
                continue
                
            print(f"⚡ Command: {command}")

            print("   (Thinking...)")
            t0 = time.time()
            resp = call_ollama(_build_prompt(command))
            raw = resp.get("response", "").strip()
            
            try:
                intent = json.loads(raw)
            except json.JSONDecodeError:
                speak("Sorry, I didn't understand that.")
                continue
                
            speech = execute(intent)
            t1 = time.time()
            
            print(f"🗣️  Response ({t1-t0:.1f}s): {speech}")
            speak(speech)
            
            HISTORY.append({"user": command, "json": raw})
            if len(HISTORY) > MAX_HISTORY:
                HISTORY.pop(0)

    except KeyboardInterrupt:
        print("\n👋 Stopped.")

if __name__ == "__main__":
    main()