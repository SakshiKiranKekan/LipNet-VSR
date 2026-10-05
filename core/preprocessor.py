"""
High-Precision Video Preprocessing & Lip ROI Extraction Pipeline.
Includes:
1. Affine Horizontal Lip & Face Alignment (Head-tilt invariance).
2. Landmark Coordinate Smoothing (EMA) to eliminate webcam jitter.
3. Adaptive Histogram Equalization (CLAHE) for illumination invariance.
4. Visual Voice Activity Detection (V-VAD) for active speech window extraction.
"""

from typing import List, Tuple, Optional, Dict
import cv2
import numpy as np


# MediaPipe Face Mesh Key Landmarks
LIP_LEFT_CORNER = 61
LIP_RIGHT_CORNER = 291
LIP_TOP_CENTER = 13
LIP_BOTTOM_CENTER = 14
LIP_OUTER_CONTOUR = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    375, 321, 405, 314, 17, 84, 181, 91, 146, 61
]


class VideoPreprocessor:
    """
    Production-grade preprocessor for LipNet Visual Speech Recognition.
    Handles affine alignment, jitter smoothing, CLAHE contrast boost, and V-VAD.
    """
    def __init__(
        self,
        target_frames: int = 75,
        target_height: int = 46,
        target_width: int = 96,
        grayscale: bool = True,
        use_clahe: bool = True
    ):
        self.target_frames = target_frames
        self.target_height = target_height
        self.target_width = target_width
        self.grayscale = grayscale
        self.use_clahe = use_clahe
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        
        # Temporal smoothing state for bounding box / landmark center
        self.prev_center: Optional[Tuple[float, float]] = None
        self.prev_size: Optional[Tuple[float, float]] = None
        self.ema_alpha = 0.65  # Weight for current frame vs history

        # Initialize MediaPipe Face Mesh
        self.face_mesh = None
        self._init_mediapipe()

    def _init_mediapipe(self):
        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=4,
                refine_landmarks=True,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3
            )
        except Exception:
            self.face_mesh = None

    def reset_smoothing(self):
        """Resets temporal smoothing buffers between video streams."""
        self.prev_center = None
        self.prev_size = None

    def _apply_affine_alignment(
        self,
        frame: np.ndarray,
        left_corner: np.ndarray,
        right_corner: np.ndarray,
        top_lip: np.ndarray,
        bottom_lip: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Computes affine transformation to horizontally level and center the mouth.
        Ensures rotational invariance against head tilts.
        """
        h, w = frame.shape[:2]
        
        # Calculate angle of lip corners
        dx = right_corner[0] - left_corner[0]
        dy = right_corner[1] - left_corner[1]
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)

        # Center point between lips
        mouth_center_x = float((left_corner[0] + right_corner[0]) / 2.0)
        mouth_center_y = float((top_lip[1] + bottom_lip[1]) / 2.0)

        # Smooth center across frames to eliminate webcam jitter
        if self.prev_center is not None:
            smooth_cx = self.ema_alpha * mouth_center_x + (1 - self.ema_alpha) * self.prev_center[0]
            smooth_cy = self.ema_alpha * mouth_center_y + (1 - self.ema_alpha) * self.prev_center[1]
        else:
            smooth_cx, smooth_cy = mouth_center_x, mouth_center_y
        self.prev_center = (smooth_cx, smooth_cy)

        # Scale estimation based on mouth width
        lip_width = np.linalg.norm(right_corner - left_corner) + 1e-6
        lip_height = np.linalg.norm(bottom_lip - top_lip)
        mar = float(lip_height / lip_width)

        # Crop dimensions with margin
        crop_w = max(40.0, lip_width * 1.8)
        crop_h = max(24.0, crop_w * (self.target_height / self.target_width))

        if self.prev_size is not None:
            smooth_w = self.ema_alpha * crop_w + (1 - self.ema_alpha) * self.prev_size[0]
            smooth_h = self.ema_alpha * crop_h + (1 - self.ema_alpha) * self.prev_size[1]
        else:
            smooth_w, smooth_h = crop_w, crop_h
        self.prev_size = (smooth_w, smooth_h)

        # Rotation matrix around smoothed mouth center
        rot_mat = cv2.getRotationMatrix2D((smooth_cx, smooth_cy), angle_deg, 1.0)
        
        # Warp full frame to level the mouth
        rotated_frame = cv2.warpAffine(
            frame,
            rot_mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        # Crop horizontally aligned mouth box
        x1 = int(max(0, smooth_cx - smooth_w / 2.0))
        y1 = int(max(0, smooth_cy - smooth_h / 2.0))
        x2 = int(min(w, smooth_cx + smooth_w / 2.0))
        y2 = int(min(h, smooth_cy + smooth_h / 2.0))

        mouth_crop = rotated_frame[y1:y2, x1:x2]
        if mouth_crop.size == 0:
            mouth_crop = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)

        mouth_resized = cv2.resize(
            mouth_crop,
            (self.target_width, self.target_height),
            interpolation=cv2.INTER_CUBIC
        )

        return mouth_resized, mar

    def extract_mouth_roi_and_landmarks(
        self,
        frame: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        """
        Extracts high-precision affine-aligned, CLAHE-enhanced mouth ROI crop.
        """
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1. MediaPipe Face Mesh with Affine Normalization
        if self.face_mesh is not None:
            results = self.face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                left = np.array([landmarks[LIP_LEFT_CORNER].x * w, landmarks[LIP_LEFT_CORNER].y * h])
                right = np.array([landmarks[LIP_RIGHT_CORNER].x * w, landmarks[LIP_RIGHT_CORNER].y * h])
                top = np.array([landmarks[LIP_TOP_CENTER].x * w, landmarks[LIP_TOP_CENTER].y * h])
                bottom = np.array([landmarks[LIP_BOTTOM_CENTER].x * w, landmarks[LIP_BOTTOM_CENTER].y * h])
                key_landmarks = np.array([left, right, top, bottom])

                aligned_mouth, mar = self._apply_affine_alignment(frame, left, right, top, bottom)

                if self.grayscale:
                    gray = cv2.cvtColor(aligned_mouth, cv2.COLOR_BGR2GRAY)
                    if self.use_clahe:
                        gray = self.clahe.apply(gray)
                    return gray, key_landmarks, mar

                return aligned_mouth, key_landmarks, mar

        # 2. Strict Human Landmark Check: If no human face mesh was found, reject frame
        # (Prevents non-human moving objects or background clutter from generating false mouth crops)
        return None, None, None

    def extract_active_speech_segment(
        self,
        frames_list: List[np.ndarray],
        mar_list: Optional[List[float]] = None
    ) -> List[np.ndarray]:
        """
        Visual Voice Activity Detection (V-VAD):
        Trims leading and trailing silence/idle frames and extracts the core speaking window.
        """
        n = len(frames_list)
        if n <= self.target_frames:
            return frames_list

        if mar_list is not None and len(mar_list) == n:
            # Detect speaking energy using MAR deviation from median
            median_mar = np.median(mar_list)
            mar_energy = np.abs(np.array(mar_list) - median_mar)
            
            # Smooth energy with 5-frame moving average
            window = np.ones(5) / 5.0
            smooth_energy = np.convolve(mar_energy, window, mode='same')
            
            threshold = np.percentile(smooth_energy, 40)
            active_indices = np.where(smooth_energy > threshold)[0]
            
            if len(active_indices) >= 15:
                start_idx = max(0, int(active_indices[0] - 4))
                end_idx = min(n, int(active_indices[-1] + 5))
                if (end_idx - start_idx) >= 20:
                    return frames_list[start_idx:end_idx]

        # Fallback: remove 10% from start and end if video is long
        if n > 100:
            trim_margin = int(n * 0.08)
            return frames_list[trim_margin : n - trim_margin]

        return frames_list

    def normalize_sequence(
        self,
        frames_list: List[np.ndarray],
        mar_list: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Standardizes temporal sequence to target_frames (75), applies intensity normalization.
        """
        if not frames_list:
            channels = 1 if self.grayscale else 3
            return np.zeros(
                (self.target_frames, self.target_height, self.target_width, channels),
                dtype=np.float32
            )

        # 1. Extract active speech segment (V-VAD)
        active_frames = self.extract_active_speech_segment(frames_list, mar_list)
        curr_len = len(active_frames)

        # 2. Resample / Interpolate to 75 frames
        if curr_len < self.target_frames:
            padding_count = self.target_frames - curr_len
            last_frame = active_frames[-1]
            padded_frames = active_frames + [last_frame.copy() for _ in range(padding_count)]
        elif curr_len > self.target_frames:
            indices = np.linspace(0, curr_len - 1, self.target_frames, dtype=int)
            padded_frames = [active_frames[idx] for idx in indices]
        else:
            padded_frames = active_frames

        arr = np.array(padded_frames, dtype=np.float32)
        if arr.ndim == 3:  # (T, H, W)
            arr = np.expand_dims(arr, axis=-1)  # (T, H, W, 1)

        # Standard zero-mean unit-variance normalization: (X - mu) / sigma
        mean = np.mean(arr)
        std = np.std(arr) + 1e-6
        normalized = (arr - mean) / std
        return normalized.astype(np.float32)
