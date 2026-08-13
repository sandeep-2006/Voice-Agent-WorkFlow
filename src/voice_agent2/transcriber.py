"""
Whisper transcription module for loading models and transcribing audio.
"""

import os
import torch
import numpy as np
import whisper

class WhisperTranscriber:
    def __init__(self, model_name: str = "base"):
        """
        Initialize Whisper transcriber with specified model size.
        Options: 'tiny', 'base', 'small', 'medium', 'large', 'turbo'
        """
        self.model_name = model_name
        self.device = self._detect_device()
        print(f"🤖 Loading Whisper '{model_name}' model on device: {self.device.upper()}...")
        
        # Load whisper model
        self.model = whisper.load_model(model_name, device=self.device)
        print(f"✅ Model '{model_name}' loaded successfully!")

    def _detect_device(self) -> str:
        """Detect available hardware accelerator (MPS for Apple Silicon, CUDA for NVIDIA, CPU fallback)."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def transcribe_audio_array(self, audio_data: np.ndarray, language: str = None) -> dict:
        """
        Transcribe a 1D float32 numpy array of audio (16kHz sample rate).
        """
        if audio_data is None or len(audio_data) == 0:
            return {"text": "", "language": ""}

        # Whisper expects float32 numpy array
        audio_data = audio_data.astype(np.float32)

        print("⚡ Transcribing audio...")
        options = {}
        if language:
            options["language"] = language

        # Run transcribe
        result = self.model.transcribe(audio_data, **options)
        return result

    def transcribe_file(self, file_path: str, language: str = None) -> dict:
        """
        Transcribe an existing audio file (WAV, MP3, M4A, etc.).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        print(f"⚡ Transcribing file: {file_path}...")
        options = {}
        if language:
            options["language"] = language

        result = self.model.transcribe(file_path, **options)
        return result
