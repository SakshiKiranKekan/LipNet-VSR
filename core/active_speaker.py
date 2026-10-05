"""
Active Speaker Identification Module (FR-5).
Analyzes facial landmark dynamics and Mouth Aspect Ratio (MAR) variance
to determine if a tracked speaker is actively speaking or silent.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np


class ActiveSpeakerDetector:
    """
    Detects active speaking state by monitoring temporal changes in mouth geometry
    and Mouth Aspect Ratio (MAR) over sliding frame windows.
    """
    def __init__(self, window_size: int = 15, mar_threshold: float = 0.015):
        """
        Args:
            window_size: Number of consecutive frames used to compute motion variance.
            mar_threshold: Minimum variance threshold to classify as active speech.
        """
        self.window_size = window_size
        self.mar_threshold = mar_threshold
        # speaker_history: {speaker_id: [list of recent MAR values]}
        self.speaker_history: Dict[int, List[float]] = {}

    @staticmethod
    def calculate_mar(lip_landmarks: np.ndarray) -> float:
        """
        Computes Mouth Aspect Ratio (MAR) given 2D landmark coordinates.
        Uses key outer and inner lip points:
        - Vertical: Upper lip center, Lower lip center
        - Horizontal: Left corner, Right corner
        Args:
            lip_landmarks: (N, 2) array of normalized or pixel coordinates.
        Returns:
            Computed aspect ratio scalar.
        """
        if len(lip_landmarks) < 4:
            return 0.0

        # Assuming standard landmarks:
        # 0: left corner, 1: right corner, 2: top lip center, 3: bottom lip center
        left_corner = lip_landmarks[0]
        right_corner = lip_landmarks[1]
        top_lip = lip_landmarks[2]
        bottom_lip = lip_landmarks[3]

        horizontal_dist = np.linalg.norm(right_corner - left_corner) + 1e-6
        vertical_dist = np.linalg.norm(bottom_lip - top_lip)

        mar = float(vertical_dist / horizontal_dist)
        return mar

    def update_speaker_mar(self, speaker_id: int, mar_val: float) -> Tuple[bool, float]:
        """
        Updates the sliding history for a speaker and returns whether they are actively speaking.
        Args:
            speaker_id: Tracked unique ID of the speaker.
            mar_val: Current frame MAR.
        Returns:
            Tuple of (is_speaking: bool, motion_score: float)
        """
        if speaker_id not in self.speaker_history:
            self.speaker_history[speaker_id] = []

        history = self.speaker_history[speaker_id]
        history.append(mar_val)
        if len(history) > self.window_size:
            history.pop(0)

        if len(history) < 5:
            # Need minimum frames to establish variance
            return False, 0.0

        # Motion score is the temporal variance of MAR + range
        mar_variance = float(np.var(history))
        mar_range = float(np.ptp(history))  # max - min
        motion_score = mar_variance * 100.0 + mar_range * 0.5

        is_speaking = motion_score > self.mar_threshold
        return is_speaking, motion_score

    def reset(self, speaker_id: Optional[int] = None):
        """Reset history for a specific speaker or all speakers."""
        if speaker_id is not None:
            self.speaker_history.pop(speaker_id, None)
        else:
            self.speaker_history.clear()
