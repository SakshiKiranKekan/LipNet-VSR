"""
GRID Corpus Dataloader and Alignment File (.align) Parser.
Handles batch generation, temporal padding, and CTC label mapping.
"""

from typing import List, Tuple, Dict, Optional
import os
import glob
import cv2
import numpy as np

from core.vocabulary import Vocabulary, default_vocab


class GridAlignParser:
    """Parses GRID corpus .align files into token sequences and word timestamps."""
    
    @staticmethod
    def parse_align_file(align_path: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Parses a GRID .align file.
        Format: <start_frame_time_samples> <end_frame_time_samples> <word>
        Returns (clean_sentence, list_of_word_segments)
        """
        words = []
        segments = []
        
        if not os.path.exists(align_path):
            return "", []

        with open(align_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    start_t, end_t, word = parts[0], parts[1], parts[2]
                    # Filter out silence ('sil') or empty tokens ('sp')
                    if word.lower() not in ["sil", "sp"]:
                        words.append(word.lower())
                        segments.append({
                            "word": word.lower(),
                            "start": int(start_t),
                            "end": int(end_t)
                        })

        full_sentence = " ".join(words)
        return full_sentence, segments


class GridDatasetLoader:
    """
    Loads video clips and alignments from the GRID corpus directory structure:
    - /data/grid/videos/s1/bbaf2n.mpg
    - /data/grid/alignments/s1/bbaf2n.align
    """
    def __init__(
        self,
        dataset_dir: str,
        vocab: Optional[Vocabulary] = None,
        max_frames: int = 75,
        target_height: int = 46,
        target_width: int = 96
    ):
        self.dataset_dir = dataset_dir
        self.vocab = vocab or default_vocab
        self.max_frames = max_frames
        self.target_height = target_height
        self.target_width = target_width
        self.samples: List[Tuple[str, str]] = []  # (video_path, align_path)
        self._scan_dataset()

    def _scan_dataset(self):
        if not os.path.exists(self.dataset_dir):
            return

        video_files = glob.glob(os.path.join(self.dataset_dir, "**", "*.mpg"), recursive=True) + \
                      glob.glob(os.path.join(self.dataset_dir, "**", "*.mp4"), recursive=True)

        for vpath in video_files:
            base_name = os.path.splitext(os.path.basename(vpath))[0]
            # Search for corresponding .align file
            parent_dir = os.path.dirname(vpath)
            align_candidate = os.path.join(parent_dir, f"{base_name}.align")
            if not os.path.exists(align_candidate):
                # Check sibling alignments directory
                align_candidate = os.path.join(self.dataset_dir, "alignments", f"{base_name}.align")

            if os.path.exists(align_candidate):
                self.samples.append((vpath, align_candidate))

    def __len__(self) -> int:
        return len(self.samples)

    def load_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """
        Loads a single (video_tensor, encoded_labels, input_len, label_len) sample.
        """
        if idx >= len(self.samples):
            raise IndexError("Sample index out of range")

        vpath, apath = self.samples[idx]
        sentence, _ = GridAlignParser.parse_align_file(apath)
        encoded_labels = self.vocab.encode(sentence)

        # Read video frames
        cap = cv2.VideoCapture(vpath)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Center-bottom mouth crop
            h, w = gray.shape
            mouth = gray[int(h*0.6):int(h*0.95), int(w*0.25):int(w*0.75)]
            mouth_resized = cv2.resize(mouth, (self.target_width, self.target_height))
            frames.append(mouth_resized)
        cap.release()

        # Standardize to max_frames
        if len(frames) < self.max_frames:
            frames += [frames[-1]] * (self.max_frames - len(frames))
        else:
            frames = frames[:self.max_frames]

        video_tensor = np.array(frames, dtype=np.float32)
        video_tensor = np.expand_dims(video_tensor, axis=-1)  # (75, 46, 96, 1)

        # Normalize
        mean = np.mean(video_tensor)
        std = np.std(video_tensor) + 1e-6
        video_tensor = (video_tensor - mean) / std

        return (
            video_tensor,
            np.array(encoded_labels, dtype=np.int32),
            self.max_frames,
            len(encoded_labels)
        )
