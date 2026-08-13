"""
Command Line Interface for Voice-to-Text Whisper Agent pipelined into LLM and Kokoro TTS.
"""

import sys
import os
from dotenv import load_dotenv
from .recorder import check_microphone, record_audio_push_to_talk, record_audio_fixed
from .transcriber import WhisperTranscriber
from .llm import LLMEngine
from .tts import KokoroTTSEngine

load_dotenv()

def print_banner():
    print("=" * 68)
    print("   🎙️  VOICE AGENT 2: WHISPER ➔ LLM ➔ KOKORO TTS SPEECH PIPELINE 🔊")
    print("=" * 68)

def main():
    print_banner()

    # Check microphone accessibility
    if not check_microphone():
        print("⚠️ Warning: No active microphone detected! Recording may fail.")

    whisper_model = os.getenv("WHISPER_MODEL", "base")
    llm_provider = os.getenv("LLM_PROVIDER", "auto")

    transcriber = None
    llm_engine = LLMEngine(provider=llm_provider)
    tts_engine = KokoroTTSEngine(enabled=True)

    def get_transcriber():
        nonlocal transcriber
        if transcriber is None:
            transcriber = WhisperTranscriber(model_name=whisper_model)
        return transcriber

    def process_pipeline(user_text: str):
        if not user_text or not user_text.strip():
            print("⚠️ No speech detected.")
            return

        print("\n" + "─" * 60)
        print(f"🗣️  USER (Transcribed Voice):")
        print(f"   \"{user_text.strip()}\"")
        print("─" * 60)

        print("\n" + "═" * 60)
        print(f"🤖 LLM RESPONSE [{llm_engine.provider.upper()}] (Streaming & Speaking...):")

        # Stream LLM tokens to stdout and speak sentences via Kokoro TTS
        llm_stream = llm_engine.generate_response_stream(user_text)
        for chunk in tts_engine.speak_stream(llm_stream):
            print(chunk, end="", flush=True)

        print("\n" + "═" * 60 + "\n")

    while True:
        tts_status = "ON 🔊" if tts_engine.enabled else "OFF 🔇"
        print("\nChoose an option:")
        print(" [1] 🔴 Live Voice -> Whisper -> LLM -> Kokoro TTS (Push-to-Talk)")
        print(" [2] ⏱️  Live Voice -> Whisper -> LLM -> Kokoro TTS (5-second timer)")
        print(" [3] 📁 Transcribe Audio File -> LLM -> Kokoro TTS")
        print(f" [4] 🔊 Toggle TTS Voice Output (Currently: {tts_status})")
        print(f" [5] ⚙️  Settings (Whisper: {whisper_model} | LLM: {llm_engine.provider.upper()})")
        print(" [6] 🧹 Clear LLM Conversation Context")
        print(" [7] ❌ Exit")

        choice = input("\nEnter choice (1-7): ").strip()

        if choice == "1":
            try:
                audio = record_audio_push_to_talk()
                if len(audio) == 0:
                    print("⚠️ No audio recorded.")
                    continue
                t = get_transcriber()
                res = t.transcribe_audio_array(audio)
                text = res.get("text", "").strip()
                process_pipeline(text)
            except Exception as e:
                print(f"❌ Error during voice pipeline: {e}")

        elif choice == "2":
            try:
                sec_str = input("Enter duration in seconds (default 5): ").strip()
                duration = float(sec_str) if sec_str else 5.0
                audio = record_audio_fixed(duration=duration)
                if len(audio) == 0:
                    print("⚠️ No audio recorded.")
                    continue
                t = get_transcriber()
                res = t.transcribe_audio_array(audio)
                text = res.get("text", "").strip()
                process_pipeline(text)
            except Exception as e:
                print(f"❌ Error during voice pipeline: {e}")

        elif choice == "3":
            file_path = input("Enter path to audio file: ").strip().strip("'\"")
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                continue
            try:
                t = get_transcriber()
                res = t.transcribe_file(file_path)
                text = res.get("text", "").strip()
                process_pipeline(text)
            except Exception as e:
                print(f"❌ Error transcribing file: {e}")

        elif choice == "4":
            tts_engine.enabled = not tts_engine.enabled
            new_status = "ENABLED 🔊" if tts_engine.enabled else "DISABLED 🔇"
            print(f"✅ Kokoro TTS Voice Output is now {new_status}")

        elif choice == "5":
            print("\n⚙️ Settings:")
            print(f" Current Whisper Model: {whisper_model}")
            print(f" Current LLM Provider:  {llm_engine.provider.upper()} ({llm_engine.model_name})")
            print(f" Current TTS Engine:   Kokoro TTS ({tts_status})")
            
            print("\nChange LLM Provider:")
            print(" [a] OpenAI")
            print(" [b] Google Gemini")
            print(" [c] Groq")
            print(" [d] Ollama (Local)")
            print(" [e] Back")
            prov_choice = input("Select provider (a-e): ").strip().lower()
            prov_map = {"a": "openai", "b": "gemini", "c": "groq", "d": "ollama"}
            if prov_choice in prov_map:
                new_prov = prov_map[prov_choice]
                llm_engine = LLMEngine(provider=new_prov)
                print(f"✅ LLM Provider switched to {new_prov.upper()}")

        elif choice == "6":
            llm_engine.reset_conversation()

        elif choice == "7":
            print(" 👋 Goodbye!")
            sys.exit(0)

        else:
            print("❌ Invalid choice. Please enter 1-7.")

if __name__ == "__main__":
    main()
