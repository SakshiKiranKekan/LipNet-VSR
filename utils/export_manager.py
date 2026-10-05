"""
Output Storage & Transcript Export Manager (FR-14).
Enables exporting recognized multi-speaker transcripts into TXT, JSON, and CSV formats.
"""

from typing import List, Dict, Any
import os
import json
import pandas as pd
from datetime import datetime


class ExportManager:
    """Handles saving and exporting generated visual speech transcripts."""

    @staticmethod
    def export_to_txt(
        speakers: List[Dict[str, Any]],
        output_path: str,
        video_name: str = "Input Video"
    ) -> str:
        """Saves transcript as a formatted text file."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write(f"LipNet Visual Speech Recognition (VSR) Transcript\n")
            f.write(f"Generated on: {timestamp}\n")
            f.write(f"Source Video: {video_name}\n")
            f.write("=" * 60 + "\n\n")

            for spk in speakers:
                f.write(f"Speaker: {spk['speaker_id']}\n")
                f.write(f"Status : {'Active Speaker' if spk.get('is_active_speaker') else 'Passive Listener'}\n")
                f.write(f"Confidence: {spk.get('confidence', 'N/A')}\n")
                f.write(f"Recognized Transcript: \"{spk.get('transcript', '')}\"\n")
                f.write("-" * 40 + "\n")

        return output_path

    @staticmethod
    def export_to_json(
        speakers: List[Dict[str, Any]],
        output_path: str,
        video_name: str = "Input Video"
    ) -> str:
        """Saves structured transcript metadata as JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        payload = {
            "metadata": {
                "project": "LipNet Visual Speech Recognition",
                "timestamp": datetime.now().isoformat(),
                "source_video": video_name,
                "speaker_count": len(speakers)
            },
            "speakers": [
                {
                    "speaker_id": spk["speaker_id"],
                    "transcript": spk.get("transcript", ""),
                    "confidence": spk.get("confidence", "0.0%"),
                    "is_active_speaker": bool(spk.get("is_active_speaker", False)),
                    "frame_count": spk.get("frame_count", 0)
                }
                for spk in speakers
            ]
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return output_path

    @staticmethod
    def export_to_csv(
        speakers: List[Dict[str, Any]],
        output_path: str,
        video_name: str = "Input Video"
    ) -> str:
        """Saves transcript data as tabular CSV."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        rows = []
        for spk in speakers:
            rows.append({
                "Source_Video": video_name,
                "Speaker_ID": spk["speaker_id"],
                "Is_Active": spk.get("is_active_speaker", False),
                "Confidence": spk.get("confidence", ""),
                "Transcript": spk.get("transcript", ""),
                "Frame_Count": spk.get("frame_count", 0)
            })

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        return output_path
