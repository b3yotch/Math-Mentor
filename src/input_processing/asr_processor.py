import whisper
import tempfile
import os
import shutil
import warnings
from .schemas import CanonicalInput
from .math_normalizer import MathNormalizer

class ASRProcessor:
    def __init__(self, model_size: str = "base"):
        """
        Initialize Whisper ASR processor.
        Args:
            model_size: 'tiny', 'base', 'small', 'medium', 'large'
        """
        self.is_available = False
        self.normalizer = MathNormalizer()
        self.model = None

        # 1. Check for FFmpeg (CRITICAL for Whisper)
        if shutil.which("ffmpeg"):
            self.is_available = True
        else:
            print("⚠️ FFmpeg not found. Audio processing will be disabled.")
            return

        # 2. Load Model
        try:
            # fp16=False is needed for CPU-only environments (Hugging Face Free Tier)
            # to avoid warnings/errors about half-precision
            print(f"⏳ Loading Whisper model '{model_size}'...")
            self.model = whisper.load_model(model_size)
            print(f"✅ Whisper model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load Whisper model: {e}")
            self.is_available = False

    def process(self, audio_path: str) -> CanonicalInput:
        """Process audio and return canonical input."""
        
        # Graceful failure if setup failed
        if not self.is_available or self.model is None:
            return CanonicalInput(
                input_type="audio",
                raw_file_path=audio_path,
                extracted_text="",
                original_extraction="",
                confidence_score=0.0,
                metadata={
                    "error": "FFmpeg or Whisper model not available on server.",
                    "engine": "whisper"
                }
            )

        try:
            # Transcribe (fp16=False ensures it works on CPU)
            result = self.model.transcribe(audio_path, fp16=False)
            
            raw_text = result["text"].strip()
            normalized_text = self.normalizer.normalize(raw_text)
            
            # Whisper doesn't give a single confidence score.
            # We use language probability as a rough proxy, or default to 1.0 if not found
            # (Note: Real word-level confidence requires more complex logic, usually overkill for this)
            confidence = 1.0 
            
            return CanonicalInput(
                input_type="audio",
                raw_file_path=audio_path,
                extracted_text=normalized_text,
                original_extraction=raw_text,
                confidence_score=confidence,
                metadata={
                    "language": result.get("language", "en"),
                    "raw_transcription": raw_text
                }
            )
            
        except Exception as e:
            return CanonicalInput(
                input_type="audio",
                raw_file_path=audio_path,
                extracted_text="",
                original_extraction="",
                confidence_score=0.0,
                metadata={
                    "error": f"Transcription failed: {str(e)}",
                    "engine": "whisper"
                }
            )