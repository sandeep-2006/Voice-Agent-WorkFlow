"""
Text-to-Speech (TTS) Module using Kokoro TTS for macOS.
Supports real-time sentence streaming, macOS native voice fallback, and Barge-In Speech Interruption.
"""

import os
import re
import sys
import threading
import subprocess
import sounddevice as sd

class KokoroTTSEngine:
    def __init__(self, voice: str = "af_heart", enabled: bool = True):
        self.enabled = enabled
        self.voice = voice
        self.pipeline = None
        self.use_fallback = False
        self._interrupted = threading.Event()

        if not self.enabled:
            print("🔇 Kokoro TTS is disabled.")
            return

        print("🔊 Initializing Kokoro TTS Engine...")
        try:
            from kokoro import KPipeline
            self.pipeline = KPipeline(lang_code='a')
            print(f"✅ Kokoro TTS initialized successfully with voice '{voice}'!")
        except Exception as e:
            print(f"⚠️ Kokoro model initialization notice ({e}). Using macOS native TTS engine fallback.")
            self.use_fallback = True

    def reset_interruption(self):
        self._interrupted.clear()

    def interrupt(self):
        """Immediately stop speech playback and cancel remaining sentences."""
        if not self._interrupted.is_set():
            self._interrupted.set()
            print("\n⚡ [BARGE-IN] Interrupted TTS speech playback!")
            try:
                sd.stop()
            except Exception:
                pass

    @property
    def is_interrupted(self) -> bool:
        return self._interrupted.is_set()

    def _clean_text_for_speech(self, text: str) -> str:
        """Remove markdown syntax and non-speech symbols for clean pronunciation."""
        text = re.sub(r'[*#_`~\[\]()<>═─│]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def speak_sentence(self, sentence: str):
        """Speak a single sentence using Kokoro TTS or macOS fallback."""
        if not self.enabled or self.is_interrupted:
            return

        clean_text = self._clean_text_for_speech(sentence)
        if not clean_text or self.is_interrupted:
            return

        if not self.use_fallback and self.pipeline is not None:
            try:
                for _, _, audio in self.pipeline(clean_text, voice=self.voice, speed=1.0, split_pattern=r'\n+'):
                    if self.is_interrupted:
                        sd.stop()
                        return
                    if audio is not None and len(audio) > 0:
                        sd.play(audio, samplerate=24000)
                        while sd.get_stream().active:
                            if self.is_interrupted:
                                sd.stop()
                                return
                            sd.sleep(50)
                return
            except Exception as e:
                print(f"\n⚠️ Kokoro audio synthesis warning: {e}. Falling back to macOS speech.", file=sys.stderr)
                self.use_fallback = True

        if not self.is_interrupted:
            try:
                subprocess.run(["say", "-r", "195", clean_text], check=False)
            except Exception as e:
                print(f"Error speaking fallback text: {e}", file=sys.stderr)

    def speak_stream(self, generator):
        """
        Receives an LLM token generator, prints tokens live to stdout,
        buffers full sentences, and speaks each sentence as soon as complete.
        Yields tokens to caller.
        """
        self.reset_interruption()
        sentence_buffer = ""
        sentence_end = re.compile(r'([.!?\n])')

        for chunk in generator:
            if self.is_interrupted:
                break

            yield chunk

            if not self.enabled:
                continue

            sentence_buffer += chunk
            
            parts = sentence_end.split(sentence_buffer)
            while len(parts) > 2:
                if self.is_interrupted:
                    break
                complete_sentence = parts[0] + parts[1]
                parts = parts[2:]
                sentence_buffer = "".join(parts)
                self.speak_sentence(complete_sentence)

        if self.enabled and sentence_buffer.strip() and not self.is_interrupted:
            self.speak_sentence(sentence_buffer)
