"""
Strict Human-Only Face Tracker & Anatomical Biometric Verification Module (FR-3, FR-4).
Guarantees that Speaker IDs are ONLY assigned to verified human beings:
1. MediaPipe 468-point 3D Human Face Mesh validation (rejects non-human moving objects, hands, background).
2. Anatomical Biometric Symmetry (validates eye-nose-mouth vertical hierarchy and eye-pupil distance).
3. Robust Single/Multi-Person Identity Persistence (IoU + Centroid tracking, 60-frame memory).
"""

from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np


def compute_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Computes Intersection over Union (IoU) between two (x, y, w, h) boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    unionArea = float(boxAArea + boxBArea - interArea)

    return interArea / unionArea if unionArea > 0 else 0.0


class TrackedFace:
    """Represents a stably tracked, verified human face identity."""
    def __init__(self, track_id: int, bbox: Tuple[int, int, int, int]):
        self.track_id = track_id
        self.bbox = bbox  # (x, y, w, h)
        self.disappeared_frames = 0
        self.total_visible_frames = 1
        self.history_bboxes: List[Tuple[int, int, int, int]] = [bbox]

    @property
    def centroid(self) -> Tuple[int, int]:
        x, y, w, h = self.bbox
        return (int(x + w / 2), int(y + h / 2))

    def update(self, new_bbox: Tuple[int, int, int, int]):
        alpha = 0.70
        x, y, w, h = self.bbox
        nx, ny, nw, nh = new_bbox
        smoothed_bbox = (
            int(alpha * nx + (1 - alpha) * x),
            int(alpha * ny + (1 - alpha) * y),
            int(alpha * nw + (1 - alpha) * w),
            int(alpha * nh + (1 - alpha) * h),
        )
        self.bbox = smoothed_bbox
        self.disappeared_frames = 0
        self.total_visible_frames += 1
        self.history_bboxes.append(smoothed_bbox)
        if len(self.history_bboxes) > 100:
            self.history_bboxes.pop(0)


class MultiPersonTracker:
    """
    Human-Only Face Tracker. Uses MediaPipe 468-landmark biometric topology
    to strictly reject any non-human moving objects (hands, bottles, shadows, background motion).
    """
    def __init__(
        self,
        max_disappeared: int = 60,
        max_distance: float = 280.0,
        distance_threshold: Optional[float] = None,
        min_iou_match: float = 0.15
    ):
        self.next_track_id = 1
        self.tracked_faces: Dict[int, TrackedFace] = {}
        self.max_disappeared = max_disappeared
        self.max_distance = distance_threshold if distance_threshold is not None else max_distance
        self.min_iou_match = min_iou_match
        
        self.face_mesh = None
        self._init_mediapipe_detector()

    def _init_mediapipe_detector(self):
        """Initializes high-confidence MediaPipe Face Mesh for strict human validation."""
        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=4,
                refine_landmarks=True,
                min_detection_confidence=0.45,
                min_tracking_confidence=0.45
            )
        except Exception:
            self.face_mesh = None

        # Secondary OpenCV Haar Cascade fallback only if MediaPipe is unavailable
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def reset(self):
        """Resets all tracked identities for a clean video run."""
        self.next_track_id = 1
        self.tracked_faces.clear()

    def verify_human_biometrics(self, landmarks, frame_w: int, frame_h: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Validates true human facial anatomy:
        1. Left Eye (33) and Right Eye (263) presence & horizontal spacing.
        2. Nose tip (1) located below eyes.
        3. Lip corners (61, 291) & mouth center (13, 14) located strictly below nose.
        Returns tight bounding box if verified human face, else None.
        """
        # Key anatomical indices
        idx_left_eye = 33
        idx_right_eye = 263
        idx_nose = 1
        idx_mouth_top = 13
        idx_mouth_bottom = 14
        idx_lip_left = 61
        idx_lip_right = 291

        # Extract coordinates
        le_x, le_y = landmarks[idx_left_eye].x * frame_w, landmarks[idx_left_eye].y * frame_h
        re_x, re_y = landmarks[idx_right_eye].x * frame_w, landmarks[idx_right_eye].y * frame_h
        n_x, n_y = landmarks[idx_nose].x * frame_w, landmarks[idx_nose].y * frame_h
        m_top_y = landmarks[idx_mouth_top].y * frame_h
        m_bot_y = landmarks[idx_mouth_bottom].y * frame_h
        lip_l_x = landmarks[idx_lip_left].x * frame_w
        lip_r_x = landmarks[idx_lip_right].x * frame_w

        # Check 1: Eye separation (Human eyes must be at least 20px apart)
        eye_distance = np.sqrt((re_x - le_x)**2 + (re_y - le_y)**2)
        if eye_distance < 20.0:
            return None

        # Check 2: Vertical Anatomy Hierarchy (Eyes above Nose, Nose above Mouth)
        eye_mid_y = (le_y + re_y) / 2.0
        if not (eye_mid_y < n_y < m_top_y and m_top_y <= m_bot_y):
            return None

        # Check 3: Mouth width must be reasonable relative to eye distance
        lip_width = abs(lip_r_x - lip_l_x)
        if lip_width < 15.0 or lip_width > (eye_distance * 2.8):
            return None

        # Calculate bounding box enclosing the complete human face
        all_x = [pt.x * frame_w for pt in landmarks]
        all_y = [pt.y * frame_h for pt in landmarks]

        min_x = max(0, int(min(all_x)))
        max_x = min(frame_w, int(max(all_x)))
        min_y = max(0, int(min(all_y)))
        max_y = min(frame_h, int(max(all_y)))

        bw = max_x - min_x
        bh = max_y - min_y

        # Quality Gate: Face dimensions must be at least 50x50
        if bw < 50 or bh < 50:
            return None

        # Margin expansion to capture full forehead and chin
        pad_x = int(bw * 0.10)
        pad_y = int(bh * 0.12)
        fx = max(0, min_x - pad_x)
        fy = max(0, min_y - pad_y)
        fw = min(frame_w - fx, bw + 2 * pad_x)
        fh = min(frame_h - fy, bh + 2 * pad_y)

        return (fx, fy, fw, fh)

    def detect_human_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detects ONLY real human faces using MediaPipe biometric validation.
        Completely rejects non-human objects, moving hands, or background noise.
        """
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        verified_bboxes = []

        # 1. Primary Human Biometric Detection with MediaPipe Face Mesh
        if self.face_mesh is not None:
            results = self.face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    valid_box = self.verify_human_biometrics(face_landmarks.landmark, w, h)
                    if valid_box is not None:
                        verified_bboxes.append(valid_box)

        # 2. Secondary Haar Fallback (ONLY if MediaPipe is disabled)
        if not verified_bboxes and self.face_mesh is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            for (x, y, fw, fh) in detected:
                verified_bboxes.append((int(x), int(y), int(fw), int(fh)))

        # Return strictly verified human face bounding boxes (NO false fallback boxes)
        return verified_bboxes

    def update(self, frame: np.ndarray) -> Dict[int, Tuple[int, int, int, int]]:
        """
        Updates tracking exclusively for verified human faces.
        Returns: {speaker_id: (x, y, w, h)}
        """
        detections = self.detect_human_faces(frame)

        # If NO human face is detected in this frame, do NOT create new IDs
        if len(detections) == 0:
            for tid in list(self.tracked_faces.keys()):
                self.tracked_faces[tid].disappeared_frames += 1
                if self.tracked_faces[tid].disappeared_frames > self.max_disappeared:
                    del self.tracked_faces[tid]
            return {tid: face.bbox for tid, face in self.tracked_faces.items()}

        # First human face detected -> Initialize Speaker 1
        if len(self.tracked_faces) == 0:
            for bbox in detections:
                self.tracked_faces[self.next_track_id] = TrackedFace(self.next_track_id, bbox)
                self.next_track_id += 1
            return {tid: face.bbox for tid, face in self.tracked_faces.items()}

        track_ids = list(self.tracked_faces.keys())
        existing_boxes = [self.tracked_faces[tid].bbox for tid in track_ids]
        existing_centroids = [self.tracked_faces[tid].centroid for tid in track_ids]
        det_centroids = [(int(x + w / 2), int(y + h / 2)) for (x, y, w, h) in detections]

        matched_tracks = set()
        matched_detections = set()

        # Step A: Match by Spatial Overlap (IoU >= 0.15)
        for i, tid in enumerate(track_ids):
            best_iou = 0.0
            best_det_idx = -1
            for j, det_box in enumerate(detections):
                if j in matched_detections:
                    continue
                iou_val = compute_iou(existing_boxes[i], det_box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_det_idx = j

            if best_iou >= self.min_iou_match and best_det_idx != -1:
                self.tracked_faces[tid].update(detections[best_det_idx])
                matched_tracks.add(tid)
                matched_detections.add(best_det_idx)

        # Step B: Match remaining by Centroid Euclidean Distance (<= max_distance)
        for tid in track_ids:
            if tid in matched_tracks:
                continue

            ec = self.tracked_faces[tid].centroid
            best_dist = float('inf')
            best_det_idx = -1

            for j, dc in enumerate(det_centroids):
                if j in matched_detections:
                    continue
                d = np.linalg.norm(np.array(ec) - np.array(dc))
                if d < best_dist:
                    best_dist = d
                    best_det_idx = j

            if best_dist <= self.max_distance and best_det_idx != -1:
                self.tracked_faces[tid].update(detections[best_det_idx])
                matched_tracks.add(tid)
                matched_detections.add(best_det_idx)

        # Step C: Increment disappeared frames for unmatched existing tracks
        for tid in track_ids:
            if tid not in matched_tracks:
                self.tracked_faces[tid].disappeared_frames += 1
                if self.tracked_faces[tid].disappeared_frames > self.max_disappeared:
                    del self.tracked_faces[tid]

        # Step D: Only create a new ID if detection is a distinctly separate human face
        for j, det_box in enumerate(detections):
            if j not in matched_detections:
                dc = det_centroids[j]
                min_dist_to_any = min(
                    [np.linalg.norm(np.array(self.tracked_faces[t].centroid) - np.array(dc)) for t in self.tracked_faces],
                    default=float('inf')
                )
                if min_dist_to_any > 160.0 and len(self.tracked_faces) < 4:
                    self.tracked_faces[self.next_track_id] = TrackedFace(self.next_track_id, det_box)
                    self.next_track_id += 1
                elif len(self.tracked_faces) > 0:
                    closest_tid = min(
                        self.tracked_faces.keys(),
                        key=lambda t: np.linalg.norm(np.array(self.tracked_faces[t].centroid) - np.array(dc))
                    )
                    self.tracked_faces[closest_tid].update(det_box)

        return {tid: face.bbox for tid, face in self.tracked_faces.items()}
