"""
Synthetic Sample Video Generator for Demo & Testing.
Generates realistic talking-face video clips with moving mouth apertures
encoded with H.264 for native browser playback.
"""

import os
import math
import cv2
import numpy as np
from utils.video_annotator import write_h264_video


class DemoVideoGenerator:
    """Creates synthetic multi-speaker talking videos for testing."""

    @staticmethod
    def draw_face_with_mouth(
        canvas: np.ndarray,
        center_x: int,
        center_y: int,
        face_radius: int,
        mouth_openness: float,
        skin_color: tuple = (180, 200, 230),
        lip_color: tuple = (60, 60, 180)
    ):
        """Draws an animated face with dynamic mouth opening."""
        # Head / Face
        cv2.circle(canvas, (center_x, center_y), face_radius, skin_color, -1)
        cv2.circle(canvas, (center_x, center_y), face_radius, (100, 120, 150), 2)

        # Eyes
        eye_offset_x = int(face_radius * 0.35)
        eye_offset_y = int(face_radius * 0.25)
        cv2.circle(canvas, (center_x - eye_offset_x, center_y - eye_offset_y), 6, (50, 50, 50), -1)
        cv2.circle(canvas, (center_x + eye_offset_x, center_y - eye_offset_y), 6, (50, 50, 50), -1)

        # Nose
        cv2.line(canvas, (center_x, center_y - 5), (center_x, center_y + 15), (100, 120, 150), 2)

        # Mouth (Dynamic Aperture)
        mouth_center_y = center_y + int(face_radius * 0.45)
        mouth_w = int(face_radius * 0.40)
        mouth_h = max(4, int(face_radius * 0.30 * mouth_openness))

        # Outer lips (ellipse)
        cv2.ellipse(
            canvas,
            (center_x, mouth_center_y),
            (mouth_w, mouth_h + 4),
            0, 0, 360,
            lip_color,
            -1
        )
        # Inner mouth cavity
        if mouth_h > 6:
            cv2.ellipse(
                canvas,
                (center_x, mouth_center_y),
                (mouth_w - 6, mouth_h - 2),
                0, 0, 360,
                (20, 20, 40),
                -1
            )
            # Teeth
            cv2.rectangle(
                canvas,
                (center_x - 12, mouth_center_y - mouth_h + 4),
                (center_x + 12, mouth_center_y - 2),
                (240, 240, 240),
                -1
            )

    @classmethod
    def generate_single_speaker_video(
        cls,
        output_path: str = "demo_single_speaker.mp4",
        num_frames: int = 75,
        fps: int = 25
    ) -> str:
        """Generates a 3-second 75-frame single talking person video encoded with H.264."""
        w, h = 640, 480
        frames = []

        for t in range(num_frames):
            frame = np.ones((h, w, 3), dtype=np.uint8) * 235

            # Studio backdrop
            cv2.rectangle(frame, (0, 0), (w, h), (240, 240, 245), -1)
            cv2.putText(
                frame,
                "Single Speaker LipNet Demo Video (75 frames / 25 FPS)",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (80, 80, 80),
                1
            )

            # Oscillating mouth movement (speech cadence)
            speech_pattern = (
                0.5 * (1.0 + math.sin(t * 0.45)) * 
                (0.8 + 0.2 * math.cos(t * 0.2))
            )
            # Slight head natural drift
            head_x = 320 + int(math.sin(t * 0.05) * 4)
            head_y = 230 + int(math.cos(t * 0.08) * 3)

            cls.draw_face_with_mouth(
                frame,
                center_x=head_x,
                center_y=head_y,
                face_radius=110,
                mouth_openness=speech_pattern
            )

            frames.append(frame)

        return write_h264_video(frames, output_path, fps=fps)

    @classmethod
    def generate_multi_speaker_video(
        cls,
        output_path: str = "demo_multi_speaker.mp4",
        num_frames: int = 75,
        fps: int = 25
    ) -> str:
        """Generates a 3-second multi-person video encoded with H.264."""
        w, h = 640, 480
        frames = []

        for t in range(num_frames):
            frame = np.ones((h, w, 3), dtype=np.uint8) * 230

            # Title
            cv2.putText(
                frame,
                "Multi-Speaker Visual Speech Recognition Demo (YOLOv8 + LipNet)",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (60, 60, 60),
                1
            )

            # Speaker 1 (Left) - Actively Talking
            spk1_mouth = 0.5 * (1.0 + math.sin(t * 0.55))
            cls.draw_face_with_mouth(
                frame,
                center_x=180,
                center_y=240,
                face_radius=90,
                mouth_openness=spk1_mouth,
                skin_color=(190, 210, 240)
            )

            # Speaker 2 (Right) - Silent / Listening (Minimal mouth movement)
            spk2_mouth = 0.08 + 0.04 * math.sin(t * 0.1)
            cls.draw_face_with_mouth(
                frame,
                center_x=460,
                center_y=240,
                face_radius=90,
                mouth_openness=spk2_mouth,
                skin_color=(175, 195, 225)
            )

            frames.append(frame)

        return write_h264_video(frames, output_path, fps=fps)
