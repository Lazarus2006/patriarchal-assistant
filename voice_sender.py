"""voice_sender.py — VAD-based voice capture on Node 0.

Run:
    python3 voice_sender.py                # record mode
    python3 voice_sender.py --send         # + discover Node 2, send audio
    python3 voice_sender.py --debug        # per-frame S/_ classification
"""
import argparse
import socket
import sys
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import webrtcvad

# ────────── Audio ──────────
SAMPLE_RATE      = 16000
FRAME_MS         = 30
FRAME_SAMPLES    = SAMPLE_RATE * FRAME_MS // 1000

# ────────── VAD tuning ──────────
VAD_MODE         = 3          # 0=permissive, 3=aggressive.
SILENCE_FRAMES   = 20
SPEECH_TRIGGER   = 8        # 8 * 30ms = 240ms speech starts recording.
MIN_UTTERANCE_S  = 0.5        # ignore files shorter than this
MAX_DURATION_S   = 3.5
MAX_FRAMES       = int(SAMPLE_RATE * MAX_DURATION_S / FRAME_SAMPLES)

# ────────── Signal floor ──────────
# Reject frames that VAD labels "speech" but that are actually near-silent hiss.
# Peak amplitude on int16: 32767 = full scale. 300 ≈ very quiet; 500 ≈ whisper.
MIN_PEAK_AMPLITUDE = 1500

RECORDINGS_DIR   = Path.home() / "recordings"
NODE2_MDNS       = "_whisper._tcp.local."

vad = webrtcvad.Vad(VAD_MODE)


# ────────── Recording ──────────
def record_utterance(debug: bool = False) -> np.ndarray:
    """Block until speech starts then ends. Returns int16 array."""
    buffer = []
    speech_streak = 0
    silence_streak = 0
    recording = False
    frame_idx = 0

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='int16',
        blocksize=FRAME_SAMPLES,
    ) as stream:
        for _ in range(3):
            stream.read(FRAME_SAMPLES)

        while True:
            frame, overflowed = stream.read(FRAME_SAMPLES)
            frame = frame.flatten()
            frame_idx += 1

            if overflowed:
                if debug:
                    print("!", end="", flush=True)
                continue

            peak = int(np.max(np.abs(frame))) if len(frame) else 0
            is_vad_speech = vad.is_speech(frame.tobytes(), SAMPLE_RATE)
            is_speech = is_vad_speech and (peak >= MIN_PEAK_AMPLITUDE)

            if debug:
                print("S" if is_speech else "_", end="", flush=True)
                if frame_idx % 50 == 0:
                    print(" ", flush=True)

            if not recording:
                if is_speech:
                    speech_streak += 1
                    buffer.append(frame)
                    if speech_streak >= SPEECH_TRIGGER:
                        recording = True
                        silence_streak = 0
                        if not debug:
                            print("🎤 ", end="", flush=True)
                else:
                    speech_streak = 0
                    buffer = []
            else:
                buffer.append(frame)
                if len(buffer) >= MAX_FRAMES:
                    if not debug:
                        print("⏱️ ", end="", flush=True)
                    break
                if is_speech:
                    silence_streak = 0
                else:
                    silence_streak += 1
                    if silence_streak >= SILENCE_FRAMES:
                        break

    if debug:
        print() 
    return np.concatenate(buffer) if buffer else np.array([], dtype=np.int16)


def save_wav(audio: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())


def discover_node2(timeout: float = 3.0):
    try:
        from zeroconf import Zeroconf, ServiceBrowser
    except ImportError:
        print("⚠️  zeroconf not installed")
        return None

    found = {"endpoint": None}

    class Listener:
        def add_service(self, zc, type_, name):
            info = zc.get_service_info(type_, name)
            if info and info.addresses:
                ip = socket.inet_ntoa(info.addresses[0])
                found["endpoint"] = f"{ip}:{info.port}"

        def update_service(self, *_): pass
        def remove_service(self, *_): pass

    zc = Zeroconf()
    try:
        ServiceBrowser(zc, NODE2_MDNS, Listener())
        deadline = time.time() + timeout
        while time.time() < deadline and not found["endpoint"]:
            time.sleep(0.2)
    finally:
        zc.close()
    return found["endpoint"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--send", action="store_true")
    ap.add_argument("--debug", action="store_true",
                    help="Print per-frame S/_ classification")
    args = ap.parse_args()

    RECORDINGS_DIR.mkdir(exist_ok=True)
    endpoint = None

    if args.send:
        print("🔍 Searching for Node 2 (mDNS)...")
        endpoint = discover_node2()
        if not endpoint:
            print("❌ Node 2 not found — falling back to record-only.")
            args.send = False
        else:
            print(f"✅ Found Node 2: http://{endpoint}")

    print(f"📁 {RECORDINGS_DIR}")
    print(f"⚙️  VAD mode {VAD_MODE}, silence-end {SILENCE_FRAMES*FRAME_MS}ms, "
          f"speech-start {SPEECH_TRIGGER*FRAME_MS}ms")
    print(f"🎙️  Listening... (Ctrl+C to stop)\n")

    import requests
    count = 0

    try:
        while True:
            audio = record_utterance(debug=args.debug)
            duration = len(audio) / SAMPLE_RATE
            if duration < MIN_UTTERANCE_S:
                if not args.debug:
                    print(f"   (ignored: {duration:.2f}s)")
                continue

            count += 1
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = RECORDINGS_DIR / f"utt_{ts}.wav"
            save_wav(audio, path)
            print(f"💾 [{count}] {path.name}  ({duration:.2f}s)")

            if args.send and endpoint:
                try:
                    with open(path, "rb") as fh:
                        r = requests.post(
                            f"http://{endpoint}/transcribe",
                            files={"file": fh}, timeout=30,
                        )
                    print(f"📝 {r.json().get('text', '')!r}")
                except Exception as e:
                    print(f"⚠️  Send failed: {e}")
    except KeyboardInterrupt:
        print(f"\n👋 Stopped after {count} utterance(s).")


if __name__ == "__main__":
    main()