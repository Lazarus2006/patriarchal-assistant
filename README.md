# 👩‍💻 Project Wife: The Local, Offline AI Voice Assistant

> *"The Greatest Patriarchal Project You've Ever Seen."* 
> 
> A fully local, multi-node, voice-activated AI assistant built entirely on legacy hardware. Zero cloud APIs, zero subscriptions, and 100% privacy. It listens, thinks, and executes system commands without ever arguing back.
> 
![Project Wife Banner](banner.png)

---

## 📖 The Concept
While the internet is busy building "Jarvis," we built **"Wife"** — an AI that *actually* gets things done. Built as a comedic yet deeply technical response to the current AI hype, Project Wife is a distributed microservices architecture that repurposes three obsolete laptops into a high-speed, multimodal AI pipeline.

**What makes it special?**
- 🧠 **Implicit Intent:** Say *"I have to make a ppt"* and it opens LibreOffice Impress. You don't need to memorize app names.
- 🗣️ **Hinglish Support:** Understands commands like *"volume badha do"* (increase volume) or *"screen tej hai"* (screen is too bright).
- 🔒 **100% Offline:** The entire STT, LLM, and TTS pipeline runs on a local LAN. No internet required for inference.
- ⚡ **Compound Commands:** *"Open GitHub and play lofi"* is parsed into a multi-step JSON array and executed sequentially.

---

## 🏗️ System Architecture
We distribute the workload across three nodes to avoid CPU thermal throttling and RAM limitations.

| Node | Hardware | Role | Software Stack |
| :--- | :--- | :--- | :--- |
| **Node 0** | ThinkPad T490s (i5 8th Gen, 16GB) | **The Brain** | Qwen 2.5 3B (Ollama), FastAPI, `executor.py`, Piper TTS |
| **Node 1** | HP ProBook 4430s (i5 2nd Gen, 8GB) | **The Monitor** | SSH, `btop`, Avahi *(Jokingly referred to as the "Saas" node)* |
| **Node 2** | Dell Latitude 5580 (i5 7th Gen, 8GB) | **The Ears** | `whisper.cpp`, FastAPI, Zeroconf (mDNS) |

### 🔄 Data Flow
1. **Audio Capture (Node 0):** WebRTC VAD (Voice Activity Detection) records 16kHz audio, ignoring background noise.
2. **Routing via mDNS:** Node 0 automatically discovers Node 2 on the LAN using `_whisper._tcp.local.` (No hardcoded IPs).
3. **STT (Node 2):** Audio is transcribed using a CPU-optimized `whisper.cpp` build.
4. **Inference (Node 0):** The text is sent to the fine-tuned LLM, which outputs strict JSON (e.g., `{"action":"tool", "tool":"open_site", "args":{"site":"youtube"}}`).
5. **Execution (Node 0):** The Python executor runs native OS commands or uses Brave's Chrome DevTools Protocol (CDP) to switch browser tabs.
6. **TTS (Node 0):** Piper generates the spoken confirmation, which is routed to ALSA.

---

## 🧠 The Machine Learning Pipeline

### The LLM: Qwen 2.5 3B Instruct
Running on a CPU-only laptop means we had to be ruthless with our model choice. We chose a 4-bit quantized (Q4_K_M) Qwen 2.5 3B model, balancing general knowledge with a ~10 tokens/sec generation speed.
- **Fine-Tuning:** We curated a dataset of **6,000 synthetic examples** covering tool calls, Hinglish variations, negative constraints, and multi-step commands. 
- **Training:** QLoRA/PEFT trained over 750 steps (Loss: 1.57 → 0.01).
- **Constrained Decoding:** We enforce a strict JSON schema via Ollama's `format` parameter to guarantee the output is always machine-parsable.

### Speech-to-Text: whisper.cpp
To avoid Python dependency hell (PyAV/CUDA issues on Python 3.14), we bypassed `faster-whisper` entirely and compiled `whisper.cpp` from source. By disabling Flash Attention (`-nfa`) and enforcing AVX2 instruction sets, we achieved ~3.8s transcription times on a 15W CPU.

---

## 🚧 Engineering Challenges Overcome
1. **CPU Thermal Throttling:** The T490s (15W TDP) throttled heavily during sustained LLM inference. Offloading STT to Node 2 kept the ThinkPad cool, stabilizing generation times.
2. **Browser Automation:** Traditional Selenium/Playwright was too slow. We utilized Brave's Remote Debugging Protocol (CDP) on port `9222` to switch browser tabs in milliseconds.
3. **Whisper Hallucinations:** Background noise frequently caused Whisper to output `[MUSIC PLAYING]`. We solved this by implementing a phonetic wake-word filter on Node 0 (catching "wife", "life", "wipe") and adding an amplitude floor to the VAD.

---

## 🚀 Future Roadmap
- [ ] **Custom Wake Word:** Replace phonetic matching with a dedicated `openWakeWord` model trained on the wake word.
- [ ] **Node 1 as API Gateway:** Move the FastAPI routing logic to Node 1 to fully realize the microservices architecture.
- [ ] **GPU Acceleration:** Introduce an eGPU (e.g., RTX 3060) to Node 0 to upgrade the LLM to a 7B or 14B model.

---

## ⚙️ Quick Start (High Level)
*(Note: As this is a distributed system, setup requires configuring three separate machines).*

1. **Node 0:** Install Ollama, pull the custom fine-tuned model `home-assistant:3b`, install Piper TTS, and run `voice_pipeline.py`.
2. **Node 2:** Compile `whisper.cpp` (with AVX2 flags), download `ggml-base.en.bin`, and run the `node2_server.py` FastAPI wrapper.
3. Ensure both nodes are on the same Wi-Fi network (mDNS will handle the rest).

---
*Disclaimer: The "Wife" branding is a comedic satire of the "AI Assistant" trope. The code is designed to treat all users with respect, and no actual wives were programmed during the making of this project.*
