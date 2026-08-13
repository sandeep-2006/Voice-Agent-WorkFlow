# Voice-Agent-WorkFlow 🎙️🤖

An end-to-end voice agent pipeline built with **OpenAI Whisper**, **Groq LLM (Llama 3.3 70B)**, and **Kokoro TTS**.

---

## ⚡ Architecture & Pipeline Flow

```
🎤 Push-to-Talk Microphone / Audio File
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

- **Whisper Speech-to-Text**: Fast local transcription powered by PyTorch & Apple Silicon MPS.
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
GROQ_API_KEY=your_groq_api_key_here
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

---

## 📜 License
MIT License.
