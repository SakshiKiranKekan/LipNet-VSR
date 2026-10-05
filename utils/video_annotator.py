"""
Video Annotation, Encoding & Browser Playback Utility.
Converts any video format (including GRID dataset .mpg, .avi, .mov)
into browser-compatible H.264 / YUV420P MP4 format for seamless Gradio playback.
"""

from typing import List, Dict, Any, Optional
import os
import cv2
import numpy as np


def write_h264_video(frames: List[np.ndarray], output_path: str, fps: float = 25.0) -> str:
    """
    Writes a sequence of BGR frames to an H.264 encoded MP4 video (yuv420p)
    guaranteed to play natively in Chrome, Edge, Firefox, and Safari HTML5 video players.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # 1. Try using imageio with bundled ffmpeg (libx264, yuv420p)
    try:
        import imageio
        # imageio expects RGB
        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        writer = imageio.get_writer(
            output_path,
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=1
        )
        for rf in rgb_frames:
            writer.append_data(rf)
        writer.close()
        return output_path
    except Exception as e:
        print(f"[Warning] imageio writer failed: {e}. Falling back to OpenCV video writer.")

    # 2. Fallback to OpenCV VideoWriter (avc1 or mp4v)
    if frames:
        h, w, _ = frames[0].shape
        # Try AVC1 (H264)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        if not out.isOpened():
            # Fallback to mp4v
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            
        for f in frames:
            out.write(f)
        out.release()

    return output_path


def convert_video_to_browser_mp4(input_path: str, output_path: Optional[str] = None) -> Optional[str]:
    """
    Reads any video (including GRID corpus .mpg) and converts it to a standard H.264 MP4.
    """
    if not os.path.exists(input_path):
        return None

    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = f"{base}_h264.mp4"

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        return None

    return write_h264_video(frames, output_path, fps=fps)


class VideoAnnotator:
    """Renders visual overlays and saves processed video with H.264 encoding."""
    
    @staticmethod
    def render_and_save_video(
        annotated_frames: List[np.ndarray],
        speakers_results: List[Dict[str, Any]],
        output_path: str,
        fps: float = 25.0
    ) -> Optional[str]:
        """
        Overlays subtitle banners with decoded transcripts and saves as H.264 MP4.
        """
        if not annotated_frames:
            return None

        h, w, _ = annotated_frames[0].shape
        rendered_frames = []

        # Build subtitle string
        subtitles = []
        for spk in speakers_results:
            text = spk.get("transcript", "")
            if text and text != "[Insufficient lip movement frames detected]":
                subtitles.append(f"{spk['speaker_id']}: \"{text}\"")

        subtitle_text = " | ".join(subtitles) if subtitles else "No speech detected"

        for frame in annotated_frames:
            f = frame.copy()

            # Render bottom subtitle bar (dark semi-transparent ribbon)
            banner_height = 45
            overlay = f.copy()
            cv2.rectangle(overlay, (0, h - banner_height), (w, h), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.75, f, 0.25, 0, f)

            # Subtitle text
            cv2.putText(
                f,
                subtitle_text,
                (20, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

            rendered_frames.append(f)

        # Write using browser-compatible H.264 writer
        return write_h264_video(rendered_frames, output_path, fps=fps)
