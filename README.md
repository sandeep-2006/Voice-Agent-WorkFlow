# Voice-Agent-WorkFlow 🎙️🤖

An end-to-end, real-time voice agent pipeline built with **Silero VAD**, **OpenAI Whisper**, **Groq LLM (Llama 3.3 70B)**, and **Kokoro TTS**.

---

## ⚡ Architecture & Pipeline Flow

```
🎤 Microphone Stream (16kHz)
     │
     ▼
⚡ Silero VAD 6.2.1 (32ms Frame Analysis & Speech Boundary Detection)
     │
     ├── 1. Auto-Speech Trigger: Starts recording automatically when you speak.
     ├── 2. Auto-Silence Cutoff: Stops recording after 1.0s of silence.
     └── 3. Barge-In Interruption: Instantly silences Kokoro TTS if user speaks mid-response.
     │
     ▼
📝 OpenAI Whisper (Local Speech-to-Text with Apple Silicon MPS Acceleration)
     │
     ▼
🚀 Groq LLM Engine (Llama-3.3-70B-Versatile Real-time Token Streaming)
     │
     ▼
🔊 Kokoro TTS Engine (Streaming Studio-Quality Voice Output at 24kHz)
```

---

## ✨ Features

- **Hands-Free Auto VAD**: Silero VAD auto-detects when speech starts and stops.
- **Barge-In Interruption**: Start talking while the AI is speaking, and Kokoro TTS will instantly cut off and listen to your new question.
- **Ultra-Fast LLM Inference**: Powered by Groq's `llama-3.3-70b-versatile`.
- **High-Quality Local Speech Synthesis**: Studio-quality voice generation using Kokoro TTS (`af_heart` voice).
- **Multi-LLM Support**: Supports **Groq**, **OpenAI** (`gpt-4o-mini`), **Google Gemini** (`gemini-2.5-flash`), and **Ollama** (Local LLM).

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10 – 3.12
- `uv` package manager (`brew install uv`)
- `ffmpeg` system package (`brew install ffmpeg`)

### 2. Environment Setup
Create a `.env` file from the example:

```bash
cp .env.example .env
```

Add your **Groq API key** in `.env`:
```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_groq_api_key_here
```
*(Get a free API key at [console.groq.com](https://console.groq.com))*

### 3. Install Dependencies
```bash
uv sync
```

### 4. Run the Voice Agent
```bash
uv run voice-agent2
```

Select Option `1` for **Hands-Free Auto VAD Mode** and start speaking naturally!

---

## 🛠️ Project Structure

```text
voice-agent2/
├── src/
│   └── voice_agent2/
│       ├── __init__.py
│       ├── cli.py          # Interactive Terminal UI & Loop
│       ├── recorder.py     # Microphone Audio Recorder
│       ├── transcriber.py  # OpenAI Whisper Transcriber (STT)
│       ├── llm.py          # LLM Integration (Groq, OpenAI, Gemini, Ollama)
│       ├── tts.py          # Kokoro TTS Engine (Text-to-Speech)
│       └── vad.py          # Silero VAD Speech & Interruption Detector
├── tests/                  # Automated Test Suite
├── pyproject.toml          # UV & Dependencies Configuration
├── .env.example            # Environment Template
└── README.md               # Project Documentation
```

---

## 📜 License
MIT License.
