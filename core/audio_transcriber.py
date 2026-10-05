"""
Audio-Backed Speech Transcription Engine using OpenAI Whisper.
Provides high-accuracy speech-to-text as the primary transcription source
when video contains an audio track.

Falls back gracefully when:
1. Video has no audio track (silent video / lip-reading only).
2. Whisper model is not installed or fails to load.
3. Audio is too short or completely silent.
"""

import os
import tempfile
import numpy as np
from typing import Dict, Any, Optional, Tuple


class AudioTranscriber:
    """
    Extracts audio from video files and transcribes using OpenAI Whisper.
    Designed for multi-modal fusion with the visual lip-reading pipeline.
    """

    def __init__(self, model_name: str = "base", device: str = "cpu"):
        """
        Args:
            model_name: Whisper model size ('tiny', 'base', 'small', 'medium', 'large').
                        'base' is recommended for speed+accuracy balance on CPU.
            device: 'cpu' or 'cuda' for GPU acceleration.
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self._init_whisper()

    def _init_whisper(self):
        """Lazily loads the Whisper model. Fails gracefully if not installed."""
        try:
            import whisper
            print(f"[AudioTranscriber] Loading Whisper '{self.model_name}' model...")
            self.model = whisper.load_model(self.model_name, device=self.device)
            print(f"[AudioTranscriber] Whisper '{self.model_name}' model loaded successfully.")
        except ImportError:
            print("[AudioTranscriber] WARNING: 'openai-whisper' not installed. "
                  "Audio transcription disabled. Install with: pip install openai-whisper")
            self.model = None
        except Exception as e:
            print(f"[AudioTranscriber] WARNING: Failed to load Whisper model: {e}")
            self.model = None

    @property
    def is_available(self) -> bool:
        """Returns True if Whisper model is loaded and ready."""
        return self.model is not None

    @staticmethod
    def extract_audio_from_video(video_path: str) -> Optional[str]:
        """
        Extracts the audio track from a video file and saves it as a temporary WAV file.
        Returns the path to the WAV file, or None if the video has no audio.
        """
        try:
            import subprocess

            # Use ffprobe to check if video has an audio stream
            probe_cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                video_path
            ]
            try:
                result = subprocess.run(
                    probe_cmd, capture_output=True, text=True, timeout=10
                )
                if "audio" not in result.stdout.lower():
                    print("[AudioTranscriber] Video has no audio track — visual-only mode.")
                    return None
            except (FileNotFoundError, subprocess.TimeoutExpired):
                # ffprobe not available, try extraction anyway
                pass

            # Extract audio to temporary WAV file using ffmpeg
            audio_path = os.path.join(
                tempfile.gettempdir(),
                f"lipnet_audio_{os.getpid()}.wav"
            )
            extract_cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # 16-bit PCM WAV
                "-ar", "16000",  # 16kHz sample rate (Whisper's native rate)
                "-ac", "1",  # Mono
                audio_path
            ]
            result = subprocess.run(
                extract_cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000:
                return audio_path
            else:
                print("[AudioTranscriber] Audio extraction produced empty/tiny file — likely silent video.")
                return None

        except FileNotFoundError:
            print("[AudioTranscriber] WARNING: ffmpeg not found. Cannot extract audio. "
                  "Install ffmpeg or add it to PATH.")
            return None
        except Exception as e:
            print(f"[AudioTranscriber] Audio extraction error: {e}")
            return None

    def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribes an audio file using Whisper.

        Returns:
            {
                "text": "hello how are you",
                "confidence": 0.95,
                "language": "en",
                "segments": [...],
                "has_audio": True,
                "source": "whisper"
            }
        """
        if self.model is None:
            return self._empty_result("Whisper model not loaded")

        try:
            result = self.model.transcribe(
                audio_path,
                language="en",
                fp16=False,  # CPU-safe
                task="transcribe",
                condition_on_previous_text=False,
                no_speech_threshold=0.5,
                logprob_threshold=-0.8
            )

            text = result.get("text", "").strip().lower()
            segments = result.get("segments", [])

            if not text or len(text) < 1:
                return self._empty_result("Whisper returned empty transcription")

            # Compute average confidence from segment log-probabilities
            if segments:
                avg_logprob = np.mean([
                    seg.get("avg_logprob", -1.0) for seg in segments
                ])
                # Convert log-probability to 0-1 confidence scale
                # Whisper avg_logprob is typically between -1.0 (low) and 0.0 (high)
                confidence = float(np.clip(1.0 + avg_logprob, 0.3, 0.99))

                # Check if Whisper flagged high no_speech_probability
                avg_no_speech = np.mean([
                    seg.get("no_speech_prob", 0.0) for seg in segments
                ])
                if avg_no_speech > 0.7:
                    return self._empty_result("High no-speech probability detected")
            else:
                confidence = 0.75

            return {
                "text": text,
                "confidence": round(confidence, 3),
                "language": result.get("language", "en"),
                "segments": segments,
                "has_audio": True,
                "source": "whisper"
            }

        except Exception as e:
            print(f"[AudioTranscriber] Transcription error: {e}")
            return self._empty_result(str(e))

    def transcribe_video(self, video_path: str) -> Dict[str, Any]:
        """
        End-to-end: extracts audio from video, then transcribes with Whisper.

        Returns same structure as transcribe_audio(), with has_audio=False
        if the video has no audio track.
        """
        if not self.is_available:
            return self._empty_result("Whisper not available")

        if not os.path.exists(video_path):
            return self._empty_result("Video file not found")

        # Step 1: Extract audio track
        audio_path = self.extract_audio_from_video(video_path)
        if audio_path is None:
            return {
                "text": "",
                "confidence": 0.0,
                "language": "en",
                "segments": [],
                "has_audio": False,
                "source": "none",
                "reason": "No audio track in video"
            }

        try:
            # Step 2: Transcribe with Whisper
            result = self.transcribe_audio(audio_path)
            return result
        finally:
            # Cleanup temporary audio file
            try:
                if audio_path and os.path.exists(audio_path):
                    os.remove(audio_path)
            except OSError:
                pass

    @staticmethod
    def _empty_result(reason: str = "") -> Dict[str, Any]:
        """Returns an empty transcription result."""
        return {
            "text": "",
            "confidence": 0.0,
            "language": "en",
            "segments": [],
            "has_audio": False,
            "source": "none",
            "reason": reason
        }
