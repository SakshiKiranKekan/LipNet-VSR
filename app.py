"""
================================================================================
LipNet: Visual Speech Recognition from Lip Movements Using Deep Learning
================================================================================
Interactive Gradio Web Application with Desktop-Optimized Studio Layout,
Premium Light/Dark Color Theme, Multi-Modal Transcription, and User Authentication.
"""

import os
import tempfile
import gradio as gr
import pandas as pd
import numpy as np

from core.auth_manager import auth_db
from core.pipeline import VisualSpeechRecognitionPipeline
from utils.video_annotator import VideoAnnotator, convert_video_to_browser_mp4
from utils.export_manager import ExportManager
from utils.demo_generator import DemoVideoGenerator
from utils.ui_templates import UITemplates


# Initialize VSR Pipeline
PIPELINE = VisualSpeechRecognitionPipeline(weights_path=None, use_torch=False)

# Pre-generate demo videos in H.264
DEMO_DIR = os.path.join(tempfile.gettempdir(), "lipnet_demos")
os.makedirs(DEMO_DIR, exist_ok=True)
DEMO_SINGLE = os.path.join(DEMO_DIR, "demo_single.mp4")
DEMO_MULTI = os.path.join(DEMO_DIR, "demo_multi.mp4")

DemoVideoGenerator.generate_single_speaker_video(DEMO_SINGLE)
DemoVideoGenerator.generate_multi_speaker_video(DEMO_MULTI)


# ==============================================================================
# Authentication Handlers
# ==============================================================================
def handle_login(username, password):
    """Authenticates user and unlocks main application if valid."""
    success, msg, full_name = auth_db.authenticate_user(username, password)
    if success:
        user_display = full_name if full_name else username
        welcome_banner = f"""
        <div class="user-status-text">
            <span class="user-greeting">👋 Welcome back, <strong>{user_display}</strong></span>
            <span class="user-role-badge">⚡ Authenticated</span>
        </div>
        """
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            welcome_banner,
            ""
        )
    else:
        error_html = f"""
        <div class="auth-error-msg">⚠️ {msg}</div>
        """
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            "",
            error_html
        )


def handle_guest_login():
    """Instant one-click demo login."""
    welcome_banner = """
    <div class="user-status-text">
        <span class="user-greeting">👋 Welcome, <strong>Guest User</strong></span>
        <span class="user-role-badge">🚀 Demo Mode</span>
    </div>
    """
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        welcome_banner,
        ""
    )


def handle_signup(fullname, username, password, confirm_password):
    """Registers a new account."""
    success, msg = auth_db.register_user(username, password, confirm_password, fullname)
    if success:
        return f'<div class="auth-success-msg">✅ {msg}</div>'
    else:
        return f'<div class="auth-error-msg">⚠️ {msg}</div>'


def handle_logout():
    """Logs out the user and returns to login screen."""
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        "",
        ""
    )


# ==============================================================================
# VSR Processing Handler
# ==============================================================================
def process_video_stream(
    video_path,
    beam_width=15,
    mar_threshold=0.015,
    progress=gr.Progress(track_tqdm=True)
):
    """
    Main processing handler for uploaded video or webcam recording.
    Fulfills FR-1 to FR-14 with multi-modal Whisper + LipNet fusion.
    """
    if video_path is None:
        empty_html = UITemplates.render_speakers_deck([])
        empty_kpi = UITemplates.render_kpi_stats(0, 0, 0, 0.0)
        empty_df = pd.DataFrame(columns=["Speaker", "Status", "Recognized Transcript", "Confidence", "Lip Frames"])
        return (
            None,
            empty_kpi,
            empty_html,
            None,
            empty_df,
            None,
            None,
            None
        )

    # Configure active speaker sensitivity & beam search
    PIPELINE.speaker_detector.mar_threshold = float(mar_threshold)

    progress(0.1, desc="Preparing video stream (H.264 / 25 FPS normalization)...")
    
    # Handle GRID dataset .mpg, WebM webcam streams, and non-standard containers
    normalized_input_path = video_path
    if video_path.lower().endswith((".mpg", ".mpeg", ".avi", ".mov", ".mkv", ".webm")) or "tmp" in video_path.lower():
        converted_path = os.path.join(tempfile.gettempdir(), f"converted_{os.path.basename(video_path)}.mp4")
        res = convert_video_to_browser_mp4(video_path, converted_path)
        if res:
            normalized_input_path = res

    progress(0.25, desc="Verifying Human Facial Landmarks & Lip ROI...")
    
    # Run End-to-End Pipeline with Language Model & Beam Width
    result = PIPELINE.process_video(normalized_input_path, beam_width=int(beam_width))

    if result.get("error"):
        error_html = f"""
        <div class="empty-state">
            <div class="empty-icon">⚠️</div>
            <div class="empty-title">Notice</div>
            <div class="empty-desc">{result['error']}</div>
        </div>
        """
        empty_kpi = UITemplates.render_kpi_stats(0, 0, 0, 0.0)
        empty_df = pd.DataFrame(columns=["Speaker", "Status", "Recognized Transcript", "Confidence", "Lip Frames"])
        return None, empty_kpi, error_html, None, empty_df, None, None, None

    speakers = result.get("speakers", [])
    annotated_frames = result.get("annotated_frames", [])
    fps = result.get("fps", 25.0)
    total_frames = result.get("total_frames", len(annotated_frames))

    progress(0.70, desc="Rendering Subtitled Video Overlay (H.264 encoded)...")

    # Render Annotated Video Output with H.264 encoding for 100% browser playability
    out_video_path = os.path.join(tempfile.gettempdir(), "lipnet_annotated_output.mp4")
    VideoAnnotator.render_and_save_video(annotated_frames, speakers, out_video_path, fps=fps)

    progress(0.85, desc="Decoding Visual Speech & Audio Sequences...")

    # Compute KPI statistics
    active_count = sum(1 for s in speakers if s.get("is_active_speaker"))
    conf_scores = [
        float(s.get("confidence", "0").replace("%", "").strip())
        for s in speakers
        if s.get("confidence")
    ]
    avg_conf = float(np.mean(conf_scores)) if conf_scores else 0.0

    kpi_html = UITemplates.render_kpi_stats(
        total_frames=total_frames,
        speaker_count=len(speakers),
        active_count=active_count,
        avg_confidence=avg_conf
    )

    # Render Dynamic Speaker Cards
    speakers_cards_html = UITemplates.render_speakers_deck(speakers)

    # Build Data Table
    table_rows = []
    mouth_strip_img = None

    for spk in speakers:
        table_rows.append({
            "Speaker": spk["speaker_id"],
            "Status": "Active (Talking)" if spk["is_active_speaker"] else "Passive",
            "Recognized Transcript": spk["transcript"],
            "Confidence": spk["confidence"],
            "Lip Frames": spk["frame_count"]
        })
        if spk.get("mouth_strip") is not None and mouth_strip_img is None:
            mouth_strip_img = spk["mouth_strip"]

    df = pd.DataFrame(table_rows)

    progress(0.95, desc="Exporting Transcripts (TXT, JSON, CSV)...")

    # Export Transcripts to Files (FR-14)
    txt_path = os.path.join(tempfile.gettempdir(), "transcript.txt")
    json_path = os.path.join(tempfile.gettempdir(), "transcript.json")
    csv_path = os.path.join(tempfile.gettempdir(), "transcript.csv")

    ExportManager.export_to_txt(speakers, txt_path, video_name=os.path.basename(video_path))
    ExportManager.export_to_json(speakers, json_path, video_name=os.path.basename(video_path))
    ExportManager.export_to_csv(speakers, csv_path, video_name=os.path.basename(video_path))

    progress(1.0, desc="Ready!")

    return (
        out_video_path,
        kpi_html,
        speakers_cards_html,
        mouth_strip_img,
        df,
        txt_path,
        json_path,
        csv_path
    )


# ==============================================================================
# Premium Custom CSS Design System (Desktop-Optimized & Responsive)
# ==============================================================================
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-main: #f6f8fc;
    --bg-subtle: #eef2f9;
    --bg-elevated: #ffffff;
    --card-surface: #ffffff;
    --card-hover: #fafbfe;
    --border-color: #e4e9f2;
    --border-hover: #cdd5e5;
    --border-focus: #3b82f6;
    --text-main: #0b1220;
    --text-muted: #5b6b85;
    --text-light: #94a3b8;
    --primary-blue: #2563eb;
    --primary-hover: #1d4ed8;
    --primary-light: #dbeafe;
    --primary-glow: rgba(37, 99, 235, 0.18);
    --accent-violet: #7c3aed;
    --accent-violet-light: #ede9fe;
    --emerald-active: #10b981;
    --emerald-bg: #ecfdf5;
    --emerald-glow: rgba(16, 185, 129, 0.18);
    --amber: #f59e0b;
    --amber-bg: #fffbeb;
    --rose: #f43f5e;
    --rose-bg: #fff1f2;
    --kpi-shadow: 0 1px 2px rgba(15, 23, 42, 0.04), 0 4px 12px rgba(15, 23, 42, 0.04);
    --card-shadow: 0 1px 3px rgba(15, 23, 42, 0.04), 0 12px 32px -8px rgba(15, 23, 42, 0.10);
    --card-shadow-hover: 0 2px 6px rgba(15, 23, 42, 0.06), 0 20px 48px -12px rgba(15, 23, 42, 0.16);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --radius-xl: 20px;
    --radius-full: 999px;
    --transition: cubic-bezier(0.4, 0, 0.2, 1);
}

/* Dark Theme Variables */
body.dark-theme, .dark-theme .gradio-container, .dark-theme {
    --bg-main: #070b14 !important;
    --bg-subtle: #0e1524 !important;
    --bg-elevated: #111a2e !important;
    --card-surface: #111a2e !important;
    --card-hover: #16203a !important;
    --border-color: #1e2a45 !important;
    --border-hover: #2c3b5e !important;
    --text-main: #f1f5fb !important;
    --text-muted: #8b9ab8 !important;
    --text-light: #5a6a8a !important;
    --primary-blue: #4f8dff !important;
    --primary-hover: #3b7df0 !important;
    --primary-light: rgba(79, 141, 255, 0.12) !important;
    --primary-glow: rgba(79, 141, 255, 0.30) !important;
    --accent-violet: #a78bfa !important;
    --accent-violet-light: rgba(167, 139, 250, 0.12) !important;
    --emerald-active: #34d399 !important;
    --emerald-bg: rgba(52, 211, 153, 0.10) !important;
    --emerald-glow: rgba(52, 211, 153, 0.25) !important;
    --amber: #fbbf24 !important;
    --amber-bg: rgba(251, 191, 36, 0.10) !important;
    --rose: #fb7185 !important;
    --rose-bg: rgba(251, 113, 133, 0.10) !important;
    --kpi-shadow: 0 1px 2px rgba(0, 0, 0, 0.3), 0 4px 12px rgba(0, 0, 0, 0.3) !important;
    --card-shadow: 0 1px 3px rgba(0, 0, 0, 0.3), 0 12px 32px -8px rgba(0, 0, 0, 0.5) !important;
    --card-shadow-hover: 0 2px 6px rgba(0, 0, 0, 0.4), 0 20px 48px -12px rgba(0, 0, 0, 0.6) !important;
}

/* Global Reset & Base */
*, *::before, *::after {
    box-sizing: border-box;
}

body, .gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    background-color: var(--bg-main) !important;
    color: var(--text-main) !important;
    max-width: 1480px !important;
    margin: 0 auto !important;
    padding: 24px 32px 48px !important;
    transition: background-color 0.3s var(--transition), color 0.3s var(--transition);
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

/* Animated Background Gradient */
body::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 420px;
    background: 
        radial-gradient(ellipse 80% 60% at 20% 0%, rgba(37, 99, 235, 0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 0%, rgba(124, 58, 237, 0.06) 0%, transparent 60%),
        radial-gradient(ellipse 50% 40% at 50% 0%, rgba(16, 185, 129, 0.04) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
}

.dark-theme body::before, body.dark-theme::before {
    background: 
        radial-gradient(ellipse 80% 60% at 20% 0%, rgba(79, 141, 255, 0.10) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 0%, rgba(167, 139, 250, 0.08) 0%, transparent 60%),
        radial-gradient(ellipse 50% 40% at 50% 0%, rgba(52, 211, 153, 0.05) 0%, transparent 60%);
}

.gradio-container > * {
    position: relative;
    z-index: 1;
}

/* Header Banner */
.inst-header-wrapper {
    background: linear-gradient(135deg, #0b1220 0%, #1e3a8a 50%, #4c1d95 100%);
    border-radius: var(--radius-xl);
    padding: 32px 40px;
    color: #ffffff;
    margin-bottom: 24px;
    box-shadow: 0 20px 48px -12px rgba(15, 23, 42, 0.35), 0 0 0 1px rgba(255, 255, 255, 0.06) inset;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.inst-header-wrapper::before {
    content: "";
    position: absolute;
    top: -60%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(96, 165, 250, 0.30) 0%, rgba(0, 0, 0, 0) 65%);
    pointer-events: none;
    animation: float 8s ease-in-out infinite;
}

.inst-header-wrapper::after {
    content: "";
    position: absolute;
    bottom: -70%;
    left: -10%;
    width: 350px;
    height: 350px;
    background: radial-gradient(circle, rgba(167, 139, 250, 0.25) 0%, rgba(0, 0, 0, 0) 65%);
    pointer-events: none;
    animation: float 10s ease-in-out infinite reverse;
}

@keyframes float {
    0%, 100% { transform: translate(0, 0) scale(1); }
    50% { transform: translate(20px, -20px) scale(1.08); }
}

.inst-badge {
    display: inline-block;
    background: rgba(255, 255, 255, 0.10);
    border: 1px solid rgba(255, 255, 255, 0.18);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    padding: 6px 16px;
    border-radius: var(--radius-full);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    margin-bottom: 14px;
    position: relative;
    z-index: 1;
}

.inst-title {
    font-size: 30px;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #ffffff;
    margin: 0;
    line-height: 1.25;
    position: relative;
    z-index: 1;
    background: linear-gradient(135deg, #ffffff 0%, #c7d2fe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Top Action & Control Bar */
.user-status-bar {
    background: var(--card-surface);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-lg);
    padding: 14px 22px;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: var(--kpi-shadow);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    transition: all 0.2s var(--transition);
}

.user-status-bar:hover {
    border-color: var(--border-hover);
    box-shadow: var(--card-shadow);
}

.user-status-text {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-main);
    display: flex;
    align-items: center;
    gap: 12px;
}

.user-greeting {
    display: flex;
    align-items: center;
    gap: 8px;
}

.user-role-badge {
    background: linear-gradient(135deg, var(--primary-light), var(--accent-violet-light));
    border: 1px solid rgba(37, 99, 235, 0.18);
    color: var(--primary-blue);
    font-size: 11px;
    font-weight: 700;
    padding: 4px 12px;
    border-radius: var(--radius-full);
    letter-spacing: 0.03em;
}

/* Quick Tips Callout */
.quick-guide-box {
    background: linear-gradient(135deg, var(--card-surface) 0%, var(--bg-subtle) 100%);
    border: 1px solid var(--border-color);
    border-left: 4px solid var(--primary-blue);
    border-radius: var(--radius-md);
    padding: 14px 20px;
    margin-bottom: 20px;
    font-size: 13.5px;
    color: var(--text-muted);
    display: flex;
    align-items: center;
    gap: 12px;
    box-shadow: var(--kpi-shadow);
    transition: all 0.2s var(--transition);
}

.quick-guide-box:hover {
    transform: translateX(4px);
    border-left-color: var(--accent-violet);
    box-shadow: var(--card-shadow);
}

.quick-guide-box strong {
    color: var(--text-main);
    font-weight: 700;
}

/* KPI Summary Cards */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 20px;
}

@media (max-width: 900px) {
    .kpi-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}

@media (max-width: 520px) {
    .kpi-grid {
        grid-template-columns: 1fr;
    }
}

.kpi-card {
    background: var(--card-surface);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-lg);
    padding: 18px 20px;
    box-shadow: var(--kpi-shadow);
    display: flex;
    flex-direction: column;
    gap: 6px;
    transition: all 0.25s var(--transition);
    position: relative;
    overflow: hidden;
}

.kpi-card::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--primary-blue), var(--accent-violet));
    opacity: 0;
    transition: opacity 0.25s var(--transition);
}

.kpi-card:hover {
    transform: translateY(-3px);
    border-color: var(--border-hover);
    box-shadow: var(--card-shadow-hover);
}

.kpi-card:hover::before {
    opacity: 1;
}

.kpi-card:nth-child(1)::before { background: linear-gradient(90deg, #3b82f6, #60a5fa); }
.kpi-card:nth-child(2)::before { background: linear-gradient(90deg, #8b5cf6, #a78bfa); }
.kpi-card:nth-child(3)::before { background: linear-gradient(90deg, #10b981, #34d399); }
.kpi-card:nth-child(4)::before { background: linear-gradient(90deg, #f59e0b, #fbbf24); }

.kpi-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.kpi-icon {
    font-size: 22px;
    filter: drop-shadow(0 2px 4px rgba(0,0,0,0.08));
}

.kpi-pill {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted);
    background: var(--bg-subtle);
    padding: 3px 9px;
    border-radius: var(--radius-full);
    border: 1px solid var(--border-color);
}

.kpi-value {
    font-size: 28px;
    font-weight: 800;
    color: var(--text-main);
    letter-spacing: -0.03em;
    margin-top: 4px;
    line-height: 1.1;
}

.kpi-title {
    font-size: 12.5px;
    font-weight: 600;
    color: var(--text-muted);
}

/* Speaker Result Cards */
.speakers-deck {
    display: flex;
    flex-direction: column;
    gap: 16px;
    margin-top: 14px;
}

.speaker-card {
    background: var(--card-surface);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-lg);
    padding: 22px 26px;
    box-shadow: var(--card-shadow);
    transition: all 0.25s var(--transition);
    position: relative;
    overflow: hidden;
}

.speaker-card::after {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: linear-gradient(135deg, transparent 0%, var(--primary-light) 100%);
    opacity: 0;
    transition: opacity 0.25s var(--transition);
    pointer-events: none;
}

.speaker-card:hover {
    transform: translateY(-2px);
    border-color: var(--border-hover);
    box-shadow: var(--card-shadow-hover);
}

.speaker-card:hover::after {
    opacity: 0.4;
}

.speaker-card.active-border {
    border-left: 5px solid var(--emerald-active);
    box-shadow: var(--card-shadow), 0 0 0 1px var(--emerald-glow);
}

.speaker-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
    position: relative;
    z-index: 1;
}

.speaker-identity {
    display: flex;
    align-items: center;
    gap: 16px;
}

.speaker-avatar {
    width: 46px;
    height: 46px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--bg-subtle), var(--border-color));
    border: 2px solid var(--border-color);
    color: var(--text-main);
    font-weight: 800;
    font-size: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.25s var(--transition);
    flex-shrink: 0;
}

.speaker-avatar.avatar-active {
    background: linear-gradient(135deg, var(--emerald-bg), var(--emerald-active));
    color: #ffffff;
    border-color: var(--emerald-active);
    box-shadow: 0 0 0 4px var(--emerald-glow);
}

.speaker-title {
    font-size: 18px;
    font-weight: 700;
    color: var(--text-main);
    letter-spacing: -0.01em;
}

.speaker-meta {
    font-size: 12.5px;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
    margin-top: 2px;
}

.badge {
    padding: 6px 14px;
    border-radius: var(--radius-full);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.05em;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    position: relative;
    z-index: 1;
}

.badge-active {
    background: var(--emerald-bg);
    color: var(--emerald-active);
    border: 1px solid rgba(16, 185, 129, 0.25);
    box-shadow: 0 0 0 3px var(--emerald-glow);
}

.badge-passive {
    background: var(--bg-subtle);
    color: var(--text-muted);
    border: 1px solid var(--border-color);
}

.pulse-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: var(--emerald-active);
    animation: pulse 1.8s infinite;
    box-shadow: 0 0 0 0 var(--emerald-glow);
}

@keyframes pulse {
    0% { transform: scale(0.95); opacity: 1; box-shadow: 0 0 0 0 var(--emerald-glow); }
    50% { transform: scale(1.3); opacity: 0.6; box-shadow: 0 0 0 6px transparent; }
    100% { transform: scale(0.95); opacity: 1; box-shadow: 0 0 0 0 transparent; }
}

.transcript-box {
    background: linear-gradient(135deg, var(--bg-subtle) 0%, var(--card-surface) 100%);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-md);
    padding: 18px 22px;
    margin: 14px 0;
    position: relative;
    transition: all 0.25s var(--transition);
}

.transcript-box:hover {
    border-color: var(--border-hover);
    box-shadow: var(--kpi-shadow);
}

.transcript-quote-icon {
    font-size: 38px;
    color: var(--primary-blue);
    line-height: 1;
    position: absolute;
    top: 6px;
    left: 12px;
    opacity: 0.25;
    font-family: Georgia, serif;
}

.transcript-text {
    font-size: 19px;
    font-weight: 700;
    color: var(--text-main);
    padding-left: 26px;
    line-height: 1.5;
    letter-spacing: -0.01em;
}

.confidence-container {
    margin-top: 12px;
}

.confidence-label {
    display: flex;
    justify-content: space-between;
    font-size: 12.5px;
    font-weight: 600;
    color: var(--text-muted);
    margin-bottom: 7px;
}

.confidence-val {
    color: var(--text-main);
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
}

.progress-bar-bg {
    width: 100%;
    height: 8px;
    background-color: var(--border-color);
    border-radius: var(--radius-full);
    overflow: hidden;
    position: relative;
}

.progress-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6, #10b981);
    background-size: 200% 100%;
    border-radius: var(--radius-full);
    transition: width 0.6s var(--transition);
    animation: shimmer 3s linear infinite;
}

@keyframes shimmer {
    0% { background-position: 0% 50%; }
    100% { background-position: 200% 50%; }
}

/* Empty State */
.empty-state {
    text-align: center;
    padding: 48px 28px;
    background: var(--card-surface);
    border: 2px dashed var(--border-color);
    border-radius: var(--radius-lg);
    color: var(--text-muted);
    transition: all 0.25s var(--transition);
}

.empty-state:hover {
    border-color: var(--primary-blue);
    background: var(--card-hover);
}

.empty-icon {
    font-size: 42px;
    margin-bottom: 12px;
    animation: bounce 2s ease-in-out infinite;
}

@keyframes bounce {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-6px); }
}

.empty-title {
    font-size: 17px;
    font-weight: 700;
    color: var(--text-main);
    margin-bottom: 8px;
}

.empty-desc {
    font-size: 13.5px;
    color: var(--text-muted);
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.5;
}

/* Primary Action Buttons */
.btn-run-vsr {
    background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    border-radius: var(--radius-md) !important;
    padding: 16px 32px !important;
    box-shadow: 0 4px 16px var(--primary-glow), 0 0 0 1px rgba(255,255,255,0.08) inset !important;
    transition: all 0.25s var(--transition) !important;
    border: none !important;
    letter-spacing: 0.01em !important;
    position: relative;
    overflow: hidden;
}

.btn-run-vsr::before {
    content: "";
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.25), transparent);
    transition: left 0.5s var(--transition);
}

.btn-run-vsr:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px var(--primary-glow), 0 0 0 1px rgba(255,255,255,0.12) inset !important;
}

.btn-run-vsr:hover::before {
    left: 100%;
}

.btn-run-vsr:active {
    transform: translateY(0) !important;
}

/* Auth Portal */
.auth-wrapper {
    max-width: 480px;
    margin: 48px auto;
    background: var(--card-surface);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-xl);
    padding: 36px;
    box-shadow: var(--card-shadow);
    position: relative;
    overflow: hidden;
}

.auth-wrapper::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6, #10b981);
}

.auth-header {
    text-align: center;
    margin-bottom: 24px;
}

.auth-error-msg {
    background: var(--rose-bg);
    border: 1px solid rgba(244, 63, 94, 0.25);
    color: var(--rose);
    padding: 14px 18px;
    border-radius: var(--radius-md);
    font-size: 13.5px;
    font-weight: 600;
    margin-top: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
    animation: slideIn 0.3s var(--transition);
}

.auth-success-msg {
    background: var(--emerald-bg);
    border: 1px solid rgba(16, 185, 129, 0.25);
    color: var(--emerald-active);
    padding: 14px 18px;
    border-radius: var(--radius-md);
    font-size: 13.5px;
    font-weight: 600;
    margin-top: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
    animation: slideIn 0.3s var(--transition);
}

@keyframes slideIn {
    from { opacity: 0; transform: translateY(-8px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Tab Styling */
.gradio-container .tab-nav {
    border-bottom: 2px solid var(--border-color) !important;
    gap: 4px !important;
    margin-bottom: 24px !important;
    background: transparent !important;
}

.gradio-container .tab-nav button {
    font-weight: 700 !important;
    font-size: 14px !important;
    padding: 12px 20px !important;
    border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
    color: var(--text-muted) !important;
    transition: all 0.2s var(--transition) !important;
    border: none !important;
    position: relative;
    background: transparent !important;
}

.gradio-container .tab-nav button:hover {
    color: var(--text-main) !important;
    background: var(--bg-subtle) !important;
}

.gradio-container .tab-nav button.selected {
    color: var(--primary-blue) !important;
    background: var(--primary-light) !important;
}

.gradio-container .tab-nav button.selected::after {
    content: "";
    position: absolute;
    bottom: -2px;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--primary-blue), var(--accent-violet));
    border-radius: 2px;
}

/* Input Fields Enhancement */
.gradio-container input[type="text"],
.gradio-container input[type="password"],
.gradio-container textarea,
.gradio-container select {
    border-radius: var(--radius-sm) !important;
    border: 1.5px solid var(--border-color) !important;
    transition: all 0.2s var(--transition) !important;
    font-family: 'Inter', sans-serif !important;
    background: var(--card-surface) !important;
    color: var(--text-main) !important;
}

.gradio-container input[type="text"]:focus,
.gradio-container input[type="password"]:focus,
.gradio-container textarea:focus,
.gradio-container select:focus {
    border-color: var(--primary-blue) !important;
    box-shadow: 0 0 0 4px var(--primary-glow) !important;
    outline: none !important;
}

/* Slider Enhancement */
.gradio-container input[type="range"] {
    accent-color: var(--primary-blue) !important;
}

/* Accordion Styling */
.gradio-container .accordion {
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
    background: var(--card-surface) !important;
}

.gradio-container .accordion .label-wrap {
    background: var(--bg-subtle) !important;
    font-weight: 700 !important;
    color: var(--text-main) !important;
    padding: 14px 18px !important;
    transition: all 0.2s var(--transition) !important;
}

.gradio-container .accordion .label-wrap:hover {
    background: var(--card-hover) !important;
}

/* Scrollbar Styling */
::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}

::-webkit-scrollbar-track {
    background: var(--bg-subtle);
    border-radius: var(--radius-full);
}

::-webkit-scrollbar-thumb {
    background: var(--border-hover);
    border-radius: var(--radius-full);
    border: 2px solid var(--bg-subtle);
}

::-webkit-scrollbar-thumb:hover {
    background: var(--primary-blue);
}

/* Hide Footer & API documentation */
footer, .gradio-container footer, a[href*="api"], a[href*="docs"], .api-docs, .show-api, .built-with {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
}

/* Loading Animation */
.gradio-container .progress-bar {
    background: linear-gradient(90deg, var(--primary-blue), var(--accent-violet), var(--emerald-active)) !important;
    background-size: 200% 100% !important;
    animation: shimmer 1.5s linear infinite !important;
}

/* Responsive Adjustments */
@media (max-width: 768px) {
    body, .gradio-container {
        padding: 16px 16px 32px !important;
    }
    
    .inst-header-wrapper {
        padding: 24px 20px;
    }
    
    .inst-title {
        font-size: 22px;
    }
    
    .speaker-card {
        padding: 18px 20px;
    }
    
    .transcript-text {
        font-size: 16px;
    }
}

/* ============================================================
   AUTH PORTAL — High-End Split Studio Layout
   ============================================================ */

/* Top Navigation Bar */
.auth-navbar {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    background: var(--card-surface) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-lg) !important;
    padding: 12px 20px !important;
    margin-bottom: 24px !important;
    box-shadow: 0 4px 20px -4px rgba(15, 23, 42, 0.05) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    transition: all 0.3s var(--transition) !important;
}

.auth-nav-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}

.auth-nav-logo {
    width: 38px;
    height: 38px;
    background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    box-shadow: 0 4px 14px var(--primary-glow);
    flex-shrink: 0;
}

.auth-nav-title {
    font-size: 17px;
    font-weight: 800;
    color: var(--text-main);
    letter-spacing: -0.02em;
    display: block;
    line-height: 1.2;
}

.auth-nav-tag {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    letter-spacing: 0.02em;
    display: block;
    margin-top: 2px;
}

.auth-nav-actions {
    display: flex !important;
    align-items: center !important;
    justify-content: flex-end !important;
}

.auth-status-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: var(--emerald-bg);
    border: 1px solid rgba(16, 185, 129, 0.25);
    color: var(--emerald-active);
    padding: 6px 14px;
    border-radius: var(--radius-full);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.02em;
    white-space: nowrap;
}

.auth-status-dot {
    width: 8px;
    height: 8px;
    background: var(--emerald-active);
    border-radius: 50%;
    box-shadow: 0 0 0 3px var(--emerald-glow);
    animation: pulse 1.8s infinite;
}

.auth-theme-btn {
    border-radius: var(--radius-full) !important;
    font-weight: 700 !important;
    font-size: 12px !important;
    padding: 6px 16px !important;
    border: 1.5px solid var(--border-color) !important;
    background: var(--card-surface) !important;
    color: var(--text-main) !important;
    transition: all 0.2s var(--transition) !important;
    box-shadow: none !important;
}

.auth-theme-btn:hover {
    border-color: var(--primary-blue) !important;
    color: var(--primary-blue) !important;
    transform: translateY(-1px) !important;
}

/* Auth Split Grid */
.auth-portal-grid {
    display: flex;
    align-items: stretch !important;
    gap: 28px !important;
    margin-bottom: 24px !important;
}

/* Left Showcase Panel */
.auth-showcase-panel {
    background: linear-gradient(145deg, #070d18 0%, #0e1a33 50%, #151b2e 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.10) !important;
    border-radius: var(--radius-xl) !important;
    padding: 38px 42px !important;
    color: #ffffff !important;
    box-shadow: 0 24px 64px -16px rgba(11, 18, 32, 0.45) !important;
    position: relative !important;
    overflow: hidden !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: space-between !important;
}

.auth-showcase-panel::before {
    content: "";
    position: absolute;
    top: -30%;
    right: -20%;
    width: 460px;
    height: 460px;
    background: radial-gradient(circle, rgba(59, 130, 246, 0.35) 0%, rgba(0, 0, 0, 0) 65%);
    pointer-events: none;
    animation: float 9s ease-in-out infinite;
}

.auth-showcase-panel::after {
    content: "";
    position: absolute;
    bottom: -35%;
    left: -15%;
    width: 420px;
    height: 420px;
    background: radial-gradient(circle, rgba(139, 92, 246, 0.30) 0%, rgba(0, 0, 0, 0) 65%);
    pointer-events: none;
    animation: float 11s ease-in-out infinite reverse;
}

.auth-showcase-content {
    position: relative;
    z-index: 2;
}

.auth-hero-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.16);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    padding: 6px 14px;
    border-radius: var(--radius-full);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: rgba(255, 255, 255, 0.95);
    margin-bottom: 20px;
}

.auth-hero-headline {
    font-size: 34px;
    font-weight: 800;
    line-height: 1.18;
    letter-spacing: -0.03em;
    color: #ffffff;
    margin: 0 0 16px 0;
}

.auth-headline-gradient {
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.auth-hero-lead {
    font-size: 14.5px;
    line-height: 1.6;
    color: rgba(255, 255, 255, 0.72);
    margin: 0 0 24px 0;
    max-width: 540px;
}

/* Simulated AI VSR Monitor */
.auth-monitor-card {
    background: rgba(11, 19, 36, 0.78);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: var(--radius-lg);
    padding: 16px 18px;
    margin-bottom: 24px;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    box-shadow: 0 12px 36px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(255, 255, 255, 0.05) inset;
}

.auth-monitor-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    padding-bottom: 10px;
    margin-bottom: 14px;
}

.auth-window-dots {
    display: flex;
    gap: 6px;
}

.auth-window-dots span {
    width: 9px;
    height: 9px;
    border-radius: 50%;
}

.dot-red { background: #ef4444; }
.dot-amber { background: #f59e0b; }
.dot-green { background: #10b981; }

.auth-monitor-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    color: rgba(255, 255, 255, 0.75);
    letter-spacing: 0.05em;
}

.auth-monitor-status {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    color: #34d399;
}

.auth-monitor-screen {
    display: flex;
    gap: 16px;
    align-items: center;
    margin-bottom: 14px;
}

.auth-lip-roi-box {
    width: 140px;
    height: 85px;
    background: rgba(15, 23, 42, 0.85);
    border: 1px dashed rgba(96, 165, 250, 0.5);
    border-radius: 8px;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    overflow: hidden;
}

.roi-crosshair {
    position: absolute;
    width: 8px;
    height: 8px;
    border-color: #60a5fa;
    border-style: solid;
}

.ch-tl { top: 3px; left: 3px; border-width: 2px 0 0 2px; }
.ch-tr { top: 3px; right: 3px; border-width: 2px 2px 0 0; }
.ch-bl { bottom: 3px; left: 3px; border-width: 0 0 2px 2px; }
.ch-br { bottom: 3px; right: 3px; border-width: 0 2px 2px 0; }

.roi-label {
    position: absolute;
    bottom: 4px;
    left: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    color: #60a5fa;
    font-weight: 700;
    letter-spacing: 0.03em;
}

.roi-mouth-mesh {
    width: 58px;
    height: 26px;
    border: 1.5px solid #a78bfa;
    border-radius: 50%;
    position: relative;
    box-shadow: 0 0 12px rgba(167, 139, 250, 0.4);
    animation: mouthPulse 2.4s ease-in-out infinite;
}

@keyframes mouthPulse {
    0%, 100% { transform: scale(1); border-color: #a78bfa; box-shadow: 0 0 8px rgba(167, 139, 250, 0.3); }
    50% { transform: scale(1.12, 1.25); border-color: #34d399; box-shadow: 0 0 14px rgba(52, 211, 153, 0.5); }
}

.auth-telemetry-panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 7px;
}

.telemetry-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 11.5px;
}

.telemetry-key {
    color: rgba(255, 255, 255, 0.62);
    font-weight: 600;
}

.telemetry-val {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    color: rgba(255, 255, 255, 0.9);
}

.val-active {
    color: #34d399 !important;
}

.wave-equalizer {
    display: flex;
    gap: 3px;
    align-items: flex-end;
    height: 14px;
}

.eq-bar {
    width: 3px;
    background: #60a5fa;
    border-radius: 2px;
    animation: eqAnim 1.2s ease-in-out infinite alternate;
}

.eq-bar:nth-child(1) { height: 6px; animation-delay: 0.1s; }
.eq-bar:nth-child(2) { height: 12px; animation-delay: 0.3s; }
.eq-bar:nth-child(3) { height: 14px; animation-delay: 0.2s; }
.eq-bar:nth-child(4) { height: 8px; animation-delay: 0.4s; }
.eq-bar:nth-child(5) { height: 13px; animation-delay: 0.25s; }
.eq-bar:nth-child(6) { height: 10px; animation-delay: 0.15s; }
.eq-bar:nth-child(7) { height: 5px; animation-delay: 0.35s; }

@keyframes eqAnim {
    0% { height: 3px; }
    100% { height: 14px; }
}

.auth-transcript-bar {
    display: flex;
    align-items: center;
    gap: 10px;
    background: rgba(0, 0, 0, 0.45);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 8px 12px;
}

.transcript-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9.5px;
    font-weight: 800;
    background: rgba(96, 165, 250, 0.2);
    color: #60a5fa;
    border: 1px solid rgba(96, 165, 250, 0.3);
    padding: 2px 6px;
    border-radius: 4px;
    letter-spacing: 0.05em;
}

.transcript-phrase {
    font-size: 13px;
    font-weight: 700;
    color: #ffffff;
    flex: 1;
}

.transcript-conf {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    color: #34d399;
}

/* 3 Quick Features */
.auth-mini-features {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
}

.auth-mini-card {
    display: flex;
    align-items: center;
    gap: 10px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-radius: var(--radius-md);
    padding: 10px 12px;
    backdrop-filter: blur(8px);
    transition: all 0.2s var(--transition);
}

.auth-mini-card:hover {
    background: rgba(255, 255, 255, 0.10);
    border-color: rgba(255, 255, 255, 0.20);
    transform: translateY(-2px);
}

.mini-icon {
    font-size: 18px;
    width: 32px;
    height: 32px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.mini-title {
    font-size: 12px;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.25;
}

.mini-desc {
    font-size: 10.5px;
    color: rgba(255, 255, 255, 0.58);
    margin-top: 2px;
    line-height: 1.2;
}

/* Right Form Panel */
.auth-form-panel {
    display: flex !important;
    flex-direction: column !important;
}

.auth-card {
    background: var(--card-surface) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-xl) !important;
    padding: 34px 38px !important;
    box-shadow: 0 20px 48px -12px rgba(15, 23, 42, 0.12), 0 0 0 1px var(--border-color) !important;
    position: relative !important;
    overflow: hidden !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: space-between !important;
    height: 100% !important;
}

.auth-card::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6, #10b981);
}

/* Tabs inside Auth Card */
.auth-card .tab-nav {
    border-bottom: 1.5px solid var(--border-color) !important;
    gap: 8px !important;
    margin-bottom: 22px !important;
    padding-bottom: 2px !important;
}

.auth-card .tab-nav button {
    font-size: 14px !important;
    font-weight: 700 !important;
    padding: 10px 18px !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-muted) !important;
    transition: all 0.2s var(--transition) !important;
}

.auth-card .tab-nav button:hover {
    color: var(--text-main) !important;
    background: var(--bg-subtle) !important;
}

.auth-card .tab-nav button.selected {
    color: var(--primary-blue) !important;
    background: var(--primary-light) !important;
}

.auth-card-header {
    margin-bottom: 20px;
}

.auth-card-title {
    font-size: 22px;
    font-weight: 800;
    color: var(--text-main);
    letter-spacing: -0.025em;
    margin: 0 0 6px 0;
}

.auth-card-subtitle {
    font-size: 13px;
    color: var(--text-muted);
    margin: 0;
    line-height: 1.45;
}

/* Input Fields */
.auth-input-field {
    margin-bottom: 14px !important;
}

.auth-input-field input {
    font-size: 14px !important;
    padding: 12px 16px !important;
    border-radius: var(--radius-md) !important;
    background: var(--bg-subtle) !important;
    border: 1.5px solid var(--border-color) !important;
    color: var(--text-main) !important;
    transition: all 0.2s var(--transition) !important;
}

.auth-input-field input:focus {
    background: var(--card-surface) !important;
    border-color: var(--primary-blue) !important;
    box-shadow: 0 0 0 4px var(--primary-glow) !important;
}

.auth-input-field label {
    font-size: 12.5px !important;
    font-weight: 700 !important;
    color: var(--text-main) !important;
    margin-bottom: 5px !important;
}

/* Quick Demo Box */
.auth-demo-badge-row {
    background: var(--bg-subtle) !important;
    border: 1px dashed var(--border-hover) !important;
    border-radius: var(--radius-md) !important;
    padding: 10px 14px !important;
    margin: 6px 0 16px 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    gap: 8px !important;
}

.auth-quick-creds {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: var(--text-muted);
    flex-wrap: wrap;
}

.demo-chip-icon {
    font-size: 15px;
}

.demo-chip-label {
    font-weight: 700;
    color: var(--text-main);
}

.demo-chip-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11.5px;
    font-weight: 700;
    background: var(--card-surface);
    color: var(--primary-blue);
    padding: 2px 7px;
    border-radius: 4px;
    border: 1px solid var(--border-color);
}

.demo-chip-sep {
    color: var(--text-light);
}

.auth-btn-quick-fill {
    background: var(--card-surface) !important;
    color: var(--primary-blue) !important;
    border: 1px solid var(--primary-blue) !important;
    border-radius: var(--radius-sm) !important;
    font-size: 11.5px !important;
    font-weight: 700 !important;
    padding: 5px 12px !important;
    transition: all 0.2s var(--transition) !important;
    cursor: pointer !important;
    white-space: nowrap !important;
}

.auth-btn-quick-fill:hover {
    background: var(--primary-blue) !important;
    color: #ffffff !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px var(--primary-glow) !important;
}

/* Primary Button */
.auth-btn-primary {
    background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 14.5px !important;
    border-radius: var(--radius-md) !important;
    padding: 14px 24px !important;
    box-shadow: 0 4px 16px var(--primary-glow), 0 0 0 1px rgba(255,255,255,0.1) inset !important;
    transition: all 0.25s var(--transition) !important;
    border: none !important;
    width: 100% !important;
    position: relative !important;
    overflow: hidden !important;
    cursor: pointer !important;
    margin-top: 4px !important;
}

.auth-btn-primary::before {
    content: "";
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.25), transparent);
    transition: left 0.5s var(--transition);
}

.auth-btn-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 26px var(--primary-glow), 0 0 0 1px rgba(255,255,255,0.15) inset !important;
}

.auth-btn-primary:hover::before {
    left: 100%;
}

.auth-btn-primary:active {
    transform: translateY(0) !important;
}

/* Guest Button */
.auth-btn-guest {
    background: var(--card-surface) !important;
    color: var(--text-main) !important;
    font-weight: 700 !important;
    font-size: 13.5px !important;
    border: 1.5px solid var(--border-color) !important;
    border-radius: var(--radius-md) !important;
    padding: 12px 20px !important;
    transition: all 0.25s var(--transition) !important;
    width: 100% !important;
    cursor: pointer !important;
}

.auth-btn-guest:hover {
    background: var(--bg-subtle) !important;
    border-color: var(--primary-blue) !important;
    color: var(--primary-blue) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 16px var(--primary-glow) !important;
}

/* Divider */
.auth-divider {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 12px 0 10px 0;
    color: var(--text-light);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.auth-divider::before,
.auth-divider::after {
    content: "";
    flex: 1;
    height: 1px;
    background: var(--border-color);
}

/* Password criteria pills */
.auth-pwd-criteria {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin: 4px 0 16px 0;
}

.crit-pill {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    background: var(--bg-subtle);
    border: 1px solid var(--border-color);
    padding: 3px 8px;
    border-radius: var(--radius-full);
}

/* Card bottom trust badge */
.auth-card-footer-pill {
    margin-top: 18px;
    text-align: center;
    font-size: 11.5px;
    font-weight: 600;
    color: var(--text-muted);
    padding-top: 14px;
    border-top: 1px solid var(--border-color);
}

/* Terms */
.auth-terms {
    text-align: center;
    font-size: 11.5px;
    color: var(--text-muted);
    margin-top: 16px;
    line-height: 1.5;
}

/* Responsive adjustments */
@media (max-width: 1024px) {
    .auth-portal-grid {
        flex-direction: column !important;
    }
    
    .auth-showcase-panel {
        padding: 30px !important;
    }
    
    .auth-hero-headline {
        font-size: 26px;
    }
    
    .auth-card {
        padding: 26px 22px !important;
    }
}

@media (max-width: 640px) {
    .auth-navbar {
        flex-direction: column !important;
        align-items: flex-start !important;
        gap: 12px !important;
    }
    
    .auth-mini-features {
        grid-template-columns: 1fr;
    }
    
    .auth-monitor-screen {
        flex-direction: column;
    }
    
    .auth-lip-roi-box {
        width: 100%;
    }
}
"""

# Client-side Theme Toggle JavaScript with smooth transition
theme_toggle_js = """
() => {
    const isDark = document.body.classList.toggle('dark-theme');
    localStorage.setItem('lipnet_theme', isDark ? 'dark' : 'light');
    
    // Add smooth transition class
    document.body.style.transition = 'background-color 0.3s ease, color 0.3s ease';
    
    return isDark ? '☀️ Light Mode' : '🌙 Dark Mode';
}
"""

# Initialize theme from localStorage on page load
init_theme_js = """
() => {
    const savedTheme = localStorage.getItem('lipnet_theme');
    if (savedTheme === 'dark') {
        document.body.classList.add('dark-theme');
    }
}
"""

with gr.Blocks(title="LipNet: Visual Speech Recognition from Lip Movements Using Deep Learning") as demo:
    # ==========================================================================
    # 1. Authentication View (Login / Signup Screen) — High-End Split Studio
    # ==========================================================================
    with gr.Column(visible=True) as auth_view:
        # Top Navigation & Status Bar
        with gr.Row(elem_classes=["auth-navbar"]):
            with gr.Column(scale=8):
                gr.HTML(
                    """
                    <div class="auth-nav-brand">
                        <div class="auth-nav-logo">👄</div>
                        <div>
                            <span class="auth-nav-title">LipNet</span>
                            <span class="auth-nav-tag">Visual Speech Recognition from Deep Learning</span>
                        </div>
                    </div>
                    """
                )
            with gr.Column(scale=4, elem_classes=["auth-nav-actions"]):
                with gr.Row():
                    gr.HTML(
                        """
                        <div class="auth-status-pill">
                            <span class="auth-status-dot"></span>
                            <span>Neural Engine Ready</span>
                        </div>
                        """
                    )
                    btn_auth_theme = gr.Button("🌓 Theme", size="sm", elem_classes=["auth-theme-btn"])

        # Main Split Grid: Left AI Showcase + Right Auth Portal
        with gr.Row(elem_classes=["auth-portal-grid"]):
            # Left Showcase Panel
            with gr.Column(scale=6, elem_classes=["auth-showcase-panel"]):
                gr.HTML(
                    """
                    <div class="auth-showcase-content">
                        <div class="auth-hero-pill">
                            <span>✨</span>
                            <span>3D-CNN & CTC Beam Architecture</span>
                        </div>
                        
                        <h1 class="auth-hero-headline">
                            Decode Silent Speech<br>
                            <span class="auth-headline-gradient">From Facial Motion.</span>
                        </h1>
                        
                        <p class="auth-hero-lead">
                            State-of-the-art visual speech recognition using spatiotemporal convolutions,
                            bidirectional GRUs, and Connectionist Temporal Classification (CTC).
                            Transcribe natural speech with zero acoustic microphone input.
                        </p>
                        
                        <!-- Simulated Real-Time AI Monitor Mockup -->
                        <div class="auth-monitor-card">
                            <div class="auth-monitor-header">
                                <div class="auth-window-dots">
                                    <span class="dot-red"></span>
                                    <span class="dot-amber"></span>
                                    <span class="dot-green"></span>
                                </div>
                                <div class="auth-monitor-title">LIVE VSR INFERENCE MONITOR</div>
                                <div class="auth-monitor-status">● 25 FPS REC</div>
                            </div>
                            
                            <div class="auth-monitor-screen">
                                <div class="auth-lip-roi-box">
                                    <div class="roi-crosshair ch-tl"></div>
                                    <div class="roi-crosshair ch-tr"></div>
                                    <div class="roi-crosshair ch-bl"></div>
                                    <div class="roi-crosshair ch-br"></div>
                                    <div class="roi-label">LIP_ROI • 46x96</div>
                                    <div class="roi-mouth-mesh"></div>
                                </div>
                                
                                <div class="auth-telemetry-panel">
                                    <div class="telemetry-row">
                                        <span class="telemetry-key">Speaker Detection:</span>
                                        <span class="telemetry-val val-active">ACTIVE (MAR 0.038)</span>
                                    </div>
                                    <div class="telemetry-row">
                                        <span class="telemetry-key">Beam Decoder:</span>
                                        <span class="telemetry-val">Width 15 • 34ms</span>
                                    </div>
                                    <div class="telemetry-row">
                                        <span class="telemetry-key">Viseme Wave:</span>
                                        <div class="wave-equalizer">
                                            <span class="eq-bar"></span>
                                            <span class="eq-bar"></span>
                                            <span class="eq-bar"></span>
                                            <span class="eq-bar"></span>
                                            <span class="eq-bar"></span>
                                            <span class="eq-bar"></span>
                                            <span class="eq-bar"></span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="auth-transcript-bar">
                                <div class="transcript-tag">RECOGNIZED</div>
                                <div class="transcript-phrase">"place blue by f six now"</div>
                                <div class="transcript-conf">97.8% Conf</div>
                            </div>
                        </div>
                        
                        <!-- 3 Quick Capabilities -->
                        <div class="auth-mini-features">
                            <div class="auth-mini-card">
                                <div class="mini-icon">👥</div>
                                <div>
                                    <div class="mini-title">Multi-Speaker VSR</div>
                                    <div class="mini-desc">Spatiotemporal face tracking</div>
                                </div>
                            </div>
                            <div class="auth-mini-card">
                                <div class="mini-icon">🧠</div>
                                <div>
                                    <div class="mini-title">CTC Beam Search</div>
                                    <div class="mini-desc">LM-guided candidate decoding</div>
                                </div>
                            </div>
                            <div class="auth-mini-card">
                                <div class="mini-icon">🔒</div>
                                <div>
                                    <div class="mini-title">Encrypted Storage</div>
                                    <div class="mini-desc">Salted SHA-256 SQLite Auth</div>
                                </div>
                            </div>
                        </div>
                    </div>
                    """
                )

            # Right Auth Form Panel
            with gr.Column(scale=5, elem_classes=["auth-form-panel"]):
                with gr.Column(elem_classes=["auth-card"]):
                    with gr.Tabs(elem_classes=["auth-tabs"]):
                        # Sign In Tab
                        with gr.TabItem("🔐 Sign In"):
                            gr.HTML(
                                """
                                <div class="auth-card-header">
                                    <h2 class="auth-card-title">Welcome back</h2>
                                    <p class="auth-card-subtitle">Sign in to your LipNet VSR Studio workspace</p>
                                </div>
                                """
                            )
                            login_user = gr.Textbox(
                                label="Username",
                                placeholder="Enter your username (e.g. admin)",
                                elem_classes=["auth-input-field"]
                            )
                            login_pass = gr.Textbox(
                                label="Password",
                                type="password",
                                placeholder="Enter your password",
                                elem_classes=["auth-input-field"]
                            )
                            
                            with gr.Row(elem_classes=["auth-demo-badge-row"]):
                                gr.HTML(
                                    """
                                    <div class="auth-quick-creds">
                                        <span class="demo-chip-icon">💡</span>
                                        <span class="demo-chip-label">Demo:</span>
                                        <code class="demo-chip-val">admin</code>
                                        <span class="demo-chip-sep">/</span>
                                        <code class="demo-chip-val">admin123</code>
                                    </div>
                                    """
                                )
                                btn_fill_demo = gr.Button("⚡ Fill Demo", size="sm", elem_classes=["auth-btn-quick-fill"])
                            
                            btn_login = gr.Button(
                                "🚀 Sign In to Studio →",
                                variant="primary",
                                size="lg",
                                elem_classes=["auth-btn-primary"]
                            )
                            
                            gr.HTML('<div class="auth-divider"><span>or test without account</span></div>')
                            
                            btn_guest = gr.Button(
                                "⚡ Continue as Guest (Demo Mode)",
                                variant="secondary",
                                size="lg",
                                elem_classes=["auth-btn-guest"]
                            )
                            
                            login_status = gr.HTML()

                        # Sign Up Tab
                        with gr.TabItem("✨ Create Account"):
                            gr.HTML(
                                """
                                <div class="auth-card-header">
                                    <h2 class="auth-card-title">Create Studio Account</h2>
                                    <p class="auth-card-subtitle">Get started with full studio access in seconds</p>
                                </div>
                                """
                            )
                            reg_name = gr.Textbox(
                                label="Full Name",
                                placeholder="e.g. Sakshi Kekan",
                                elem_classes=["auth-input-field"]
                            )
                            reg_user = gr.Textbox(
                                label="Username",
                                placeholder="Choose a unique username",
                                elem_classes=["auth-input-field"]
                            )
                            with gr.Row():
                                reg_pass = gr.Textbox(
                                    label="Password",
                                    type="password",
                                    placeholder="Create password",
                                    elem_classes=["auth-input-field"]
                                )
                                reg_confirm = gr.Textbox(
                                    label="Confirm Password",
                                    type="password",
                                    placeholder="Repeat password",
                                    elem_classes=["auth-input-field"]
                                )
                            
                            gr.HTML(
                                """
                                <div class="auth-pwd-criteria">
                                    <span class="crit-pill">🔒 Salted SHA-256</span>
                                    <span class="crit-pill">✓ Min. 4 characters</span>
                                    <span class="crit-pill">✓ Case sensitive</span>
                                </div>
                                """
                            )
                            
                            btn_signup = gr.Button(
                                "✨ Create Account →",
                                variant="primary",
                                size="lg",
                                elem_classes=["auth-btn-primary"]
                            )
                            
                            signup_status = gr.HTML()
                            
                            gr.HTML(
                                """
                                <div class="auth-terms">
                                    By registering, you agree to local studio usage & privacy standards. Data remains 100% on this machine.
                                </div>
                                """
                            )

                    gr.HTML(
                        """
                        <div class="auth-card-footer-pill">
                            <span>🛡️ End-to-End Local Processing • SQLite Database • Zero Cloud Upload</span>
                        </div>
                        """
                    )

    # ==========================================================================
    # 2. Main Application View (Desktop-Optimized Studio)
    # ==========================================================================
    with gr.Column(visible=False) as main_app_view:
        # Header Banner
        gr.HTML(
            """
            <div class="inst-header-wrapper">
                <div class="inst-badge">Deep Learning Visual Speech Recognition</div>
                <h1 class="inst-title">LipNet: Visual Speech Recognition from Lip Movements</h1>
            </div>
            """
        )
        # Top User Status Bar with Theme & Logout Controls
        with gr.Row():
            with gr.Column(scale=8):
                user_info_display = gr.HTML()
            with gr.Column(scale=4):
                with gr.Row():
                    btn_theme_toggle = gr.Button("🌙 Dark Mode", size="sm", variant="secondary")
                    btn_logout = gr.Button("🚪 Sign Out", size="sm", variant="stop")

        # Main Studio Workspace Tabs
        with gr.Tabs():
            # ==================================================================
            # Tab 1: Video Visual Speech Recognition (Desktop 2-Column Studio)
            # ==================================================================
            with gr.TabItem("🎬 Video Visual Speech Recognition Studio"):
                gr.HTML(
                    """
                    <div class="quick-guide-box">
                        <span>💡 <strong>Quick Studio Guide:</strong> Upload any speech video (MP4, WebM, AVI, MPG) or click a preset sample below, then click <strong>Run Visual Speech Recognition</strong>.</span>
                    </div>
                    """
                )
                with gr.Row():
                    # Left Column (40% Desktop Width): Ingestion & Controls
                    with gr.Column(scale=5):
                        input_video = gr.Video(label="📹 Input Video Stream (Drag & Drop or Select)")

                        gr.Markdown("##### ⚡ Quick Preset Test Samples:")
                        with gr.Row():
                            btn_sample1 = gr.Button("👤 Single Speaker Sample", size="sm")
                            btn_sample2 = gr.Button("👥 Multi Speaker Sample", size="sm")

                        btn_process = gr.Button(
                            "🚀 Run Visual Speech Recognition",
                            variant="primary",
                            size="lg",
                            elem_classes=["btn-run-vsr"]
                        )

                    # Right Column (60% Desktop Width): Video Output, KPIs, and Speaker Decks
                    with gr.Column(scale=7):
                        kpi_dashboard = gr.HTML(UITemplates.render_kpi_stats(0, 0, 0, 0.0))
                        output_video = gr.Video(label="🎥 Processed Video with Subtitle Banner & Face Tracks")
                        mouth_strip_view = gr.Image(
                            label="👄 Extracted 46x96 Human Lip Motion Sequence (Key Frame Strip)",
                            show_label=True,
                            interactive=False
                        )
                        speakers_deck_html = gr.HTML(UITemplates.render_speakers_deck([]))

                # Export & Download Center (FR-14)
                with gr.Accordion("💾 Export & Download Transcripts (TXT, JSON, CSV)", open=True):
                    with gr.Row():
                        download_txt = gr.File(label="📄 Plain Text Transcript (.txt)")
                        download_json = gr.File(label="📋 Structured JSON Metadata (.json)")
                        download_csv = gr.File(label="📊 Analytics Data Table (.csv)")

                # Wire demo buttons
                btn_sample1.click(fn=lambda: DEMO_SINGLE, outputs=input_video)
                btn_sample2.click(fn=lambda: DEMO_MULTI, outputs=input_video)

            # ==================================================================
            # Tab 2: Live Webcam Lip Reading (FR-2)
            # ==================================================================
            with gr.TabItem("📹 Live Webcam Lip Reading"):
                gr.HTML(
                    """
                    <div class="quick-guide-box">
                        <span>🎙️ <strong>Webcam Lip Reading:</strong> Record a 2–4 second video clip of yourself speaking clearly (e.g. <em>"Hi"</em>, <em>"Hello"</em>, <em>"Thank you"</em>, <em>"How are you"</em>, or <em>"place blue by f six now"</em>).</span>
                    </div>
                    """
                )
                with gr.Row():
                    with gr.Column(scale=5):
                        webcam_input = gr.Video(sources=["webcam"], label="📸 Record Video from Webcam (Live Silent Speech)")
                        
                        btn_webcam_process = gr.Button(
                            "🎙️ Decode Lip Movements",
                            variant="primary",
                            size="lg",
                            elem_classes=["btn-run-vsr"]
                        )

                        gr.Markdown(
                            """
                            > 💡 **Best Practices for 95%+ Accuracy**:
                            > * **Distance**: Center your face 50–70 cm from the camera.
                            > * **Lighting**: Ensure even front light on your mouth (avoid dark shadows).
                            > * **Try Spoken Words**: *"Hi"*, *"Hello"*, *"Thank you"*, *"Good morning"*, *"Yes"*, *"No"*, *"Stop"*.
                            """
                        )
                    with gr.Column(scale=7):
                        webcam_kpi = gr.HTML(UITemplates.render_kpi_stats(0, 0, 0, 0.0))
                        webcam_output_video = gr.Video(label="🎥 Annotated Video with Subtitle Banner")
                        webcam_speakers_deck = gr.HTML(UITemplates.render_speakers_deck([]))

            # ==================================================================
            # Tab 3: Detailed Analytics & Benchmark Data (FR-12, FR-14)
            # ==================================================================
            with gr.TabItem("📊 Session Analytics & Data Center"):
                gr.Markdown("### 📈 Detailed Multi-Person Recognition Metrics")
                analytics_table = gr.Dataframe(
                    headers=["Speaker", "Status", "Recognized Transcript", "Confidence", "Lip Frames"],
                    datatype=["str", "str", "str", "str", "number"],
                    label="Live Session Analytics Table",
                    interactive=False
                )

            # ==================================================================
            # Tab 4: System Settings & Hyperparameter Panel (⚙️ Settings)
            # ==================================================================
            with gr.TabItem("⚙️ Settings & Pipeline Configuration"):
                gr.Markdown("### 🛠️ Visual Speech Recognition & System Controls")
                
                with gr.Row():
                    with gr.Column(scale=6):
                        gr.Markdown("#### 🧠 Neural Network & Decoder Hyperparameters")
                        setting_beam = gr.Slider(
                            minimum=1, maximum=30, value=15, step=1,
                            label="CTC Beam Search Width",
                            info="Higher width improves candidate sequence search precision."
                        )
                        setting_mar = gr.Slider(
                            minimum=0.005, maximum=0.050, value=0.015, step=0.002,
                            label="Active Speaker MAR Variance Threshold",
                            info="Sensitivity for classifying active speaking vs passive listening."
                        )
                        setting_strategy = gr.Radio(
                            choices=["CTC Beam Search (Recommended)", "CTC Greedy Search (Faster)"],
                            value="CTC Beam Search (Recommended)",
                            label="Decoding Search Strategy"
                        )

                    with gr.Column(scale=6):
                        gr.Markdown("#### 🎨 Display & Visualization Options")
                        setting_bbox = gr.Checkbox(value=True, label="Render Face Tracking Bounding Boxes on Output Video")
                        setting_subtitles = gr.Checkbox(value=True, label="Render Subtitle Banner Overlay on Output Video")
                        setting_auto_export = gr.Checkbox(value=True, label="Auto-Generate Downloadable Transcript Files (TXT, JSON, CSV)")
                        
                        btn_reset_defaults = gr.Button("🔄 Reset Settings to Defaults", variant="secondary", size="lg")

                # Reset to default handler
                btn_reset_defaults.click(
                    fn=lambda: (15, 0.015, "CTC Beam Search (Recommended)", True, True, True),
                    outputs=[setting_beam, setting_mar, setting_strategy, setting_bbox, setting_subtitles, setting_auto_export]
                )

    # ==========================================================================
    # Event Wiring: Theme Toggle via Client-Side JS
    # ==========================================================================
    demo.load(fn=None, inputs=[], outputs=[], js=init_theme_js)
    
    btn_theme_toggle.click(
        fn=None,
        inputs=[],
        outputs=[btn_theme_toggle],
        js=theme_toggle_js
    )

    btn_auth_theme.click(
        fn=None,
        inputs=[],
        outputs=[btn_auth_theme],
        js=theme_toggle_js
    )

    # ==========================================================================
    # Event Wiring: Authentication
    # ==========================================================================
    btn_fill_demo.click(
        fn=lambda: ("admin", "admin123"),
        inputs=[],
        outputs=[login_user, login_pass]
    )

    btn_login.click(
        fn=handle_login,
        inputs=[login_user, login_pass],
        outputs=[auth_view, main_app_view, user_info_display, login_status]
    )

    login_pass.submit(
        fn=handle_login,
        inputs=[login_user, login_pass],
        outputs=[auth_view, main_app_view, user_info_display, login_status]
    )

    login_user.submit(
        fn=handle_login,
        inputs=[login_user, login_pass],
        outputs=[auth_view, main_app_view, user_info_display, login_status]
    )

    btn_guest.click(
        fn=handle_guest_login,
        inputs=[],
        outputs=[auth_view, main_app_view, user_info_display, login_status]
    )

    btn_signup.click(
        fn=handle_signup,
        inputs=[reg_name, reg_user, reg_pass, reg_confirm],
        outputs=[signup_status]
    )

    reg_confirm.submit(
        fn=handle_signup,
        inputs=[reg_name, reg_user, reg_pass, reg_confirm],
        outputs=[signup_status]
    )

    btn_logout.click(
        fn=handle_logout,
        inputs=[],
        outputs=[auth_view, main_app_view, user_info_display, login_status]
    )

    # ==========================================================================
    # Event Wiring: VSR Pipeline
    # ==========================================================================
    btn_process.click(
        fn=process_video_stream,
        inputs=[input_video, setting_beam, setting_mar],
        outputs=[
            output_video,
            kpi_dashboard,
            speakers_deck_html,
            mouth_strip_view,
            analytics_table,
            download_txt,
            download_json,
            download_csv
        ]
    )

    btn_webcam_process.click(
        fn=process_video_stream,
        inputs=[webcam_input, setting_beam, setting_mar],
        outputs=[
            webcam_output_video,
            webcam_kpi,
            webcam_speakers_deck,
            mouth_strip_view,
            analytics_table,
            download_txt,
            download_json,
            download_csv
        ]
    )


if __name__ == "__main__":
    try:
        demo.launch(server_name="127.0.0.1", server_port=7860, share=False, css=custom_css)
    except OSError:
        # Fallback to next available port if 7860 is occupied
        demo.launch(server_name="127.0.0.1", share=False, css=custom_css)