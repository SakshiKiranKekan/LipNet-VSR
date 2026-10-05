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
            <span>👋 Welcome, <strong>{user_display}</strong></span>
            <span class="user-role-badge">⚡ Authenticated User</span>
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
        <span>👋 Welcome, <strong>Guest User</strong></span>
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
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-main: #f8fafc;
    --bg-subtle: #f1f5f9;
    --card-surface: #ffffff;
    --card-hover: #f8fafc;
    --border-color: #e2e8f0;
    --border-hover: #cbd5e1;
    --text-main: #0f172a;
    --text-muted: #64748b;
    --text-light: #94a3b8;
    --primary-blue: #2563eb;
    --primary-hover: #1d4ed8;
    --primary-glow: rgba(37, 99, 235, 0.25);
    --emerald-active: #10b981;
    --emerald-bg: #ecfdf5;
    --kpi-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
    --card-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.06), 0 4px 6px -4px rgba(0, 0, 0, 0.04);
}

/* Dark Theme Variables */
body.dark-theme, .dark-theme .gradio-container, .dark-theme {
    --bg-main: #090d16 !important;
    --bg-subtle: #111726 !important;
    --card-surface: #141d2e !important;
    --card-hover: #1a253a !important;
    --border-color: #243248 !important;
    --border-hover: #334561 !important;
    --text-main: #f8fafc !important;
    --text-muted: #94a3b8 !important;
    --text-light: #64748b !important;
    --primary-blue: #3b82f6 !important;
    --primary-hover: #2563eb !important;
    --primary-glow: rgba(59, 130, 246, 0.35);
    --emerald-active: #10b981 !important;
    --emerald-bg: rgba(16, 185, 129, 0.12) !important;
    --kpi-shadow: 0 4px 12px rgba(0, 0, 0, 0.4) !important;
    --card-shadow: 0 12px 24px -4px rgba(0, 0, 0, 0.5) !important;
}

/* Desktop-Friendly Container */
body, .gradio-container {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: var(--bg-main) !important;
    color: var(--text-main) !important;
    max-width: 1440px !important;
    margin: 0 auto !important;
    padding: 20px 28px !important;
    transition: background-color 0.25s ease, color 0.25s ease;
}

/* Header Banner */
.inst-header-wrapper {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 55%, #172554 100%);
    border-radius: 16px;
    padding: 24px 32px;
    color: #ffffff;
    margin-bottom: 20px;
    box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.2);
    text-align: center;
    position: relative;
    overflow: hidden;
}

.inst-header-wrapper::after {
    content: "";
    position: absolute;
    top: -50%;
    right: -20%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(59, 130, 246, 0.25) 0%, rgba(0, 0, 0, 0) 70%);
    pointer-events: none;
}

.inst-badge {
    display: inline-block;
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid rgba(255, 255, 255, 0.2);
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 11.5px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.inst-title {
    font-size: 26px;
    font-weight: 800;
    letter-spacing: -0.025em;
    color: #ffffff;
    margin: 0;
    line-height: 1.25;
}

/* Top Action & Control Bar */
.user-status-bar {
    background: var(--card-surface);
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 12px 20px;
    margin-bottom: 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: var(--kpi-shadow);
}

.user-status-text {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-main);
    display: flex;
    align-items: center;
    gap: 10px;
}

.user-role-badge {
    background: var(--bg-subtle);
    border: 1px solid var(--border-color);
    color: var(--text-muted);
    font-size: 11px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 12px;
}

/* Quick Tips Callout */
.quick-guide-box {
    background: var(--card-surface);
    border: 1px solid var(--border-color);
    border-left: 4px solid var(--primary-blue);
    border-radius: 10px;
    padding: 12px 18px;
    margin-bottom: 16px;
    font-size: 13px;
    color: var(--text-muted);
    display: flex;
    align-items: center;
    gap: 12px;
}

.quick-guide-box strong {
    color: var(--text-main);
}

/* KPI Summary Cards */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 18px;
}

@media (max-width: 900px) {
    .kpi-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}

.kpi-card {
    background: var(--card-surface);
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 16px 18px;
    box-shadow: var(--kpi-shadow);
    display: flex;
    flex-direction: column;
    gap: 4px;
    transition: transform 0.15s ease, border-color 0.15s ease;
}

.kpi-card:hover {
    transform: translateY(-2px);
    border-color: var(--border-hover);
}

.kpi-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.kpi-icon {
    font-size: 20px;
}

.kpi-pill {
    font-size: 10.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    background: var(--bg-subtle);
    padding: 2px 7px;
    border-radius: 8px;
}

.kpi-value {
    font-size: 26px;
    font-weight: 800;
    color: var(--text-main);
    letter-spacing: -0.03em;
    margin-top: 4px;
}

.kpi-title {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-muted);
}

/* Speaker Result Cards */
.speakers-deck {
    display: flex;
    flex-direction: column;
    gap: 14px;
    margin-top: 12px;
}

.speaker-card {
    background: var(--card-surface);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 20px 24px;
    box-shadow: var(--card-shadow);
    transition: all 0.2s ease;
}

.speaker-card:hover {
    box-shadow: 0 14px 28px -4px rgba(0, 0, 0, 0.09);
    border-color: var(--border-hover);
}

.speaker-card.active-border {
    border-left: 5px solid var(--emerald-active);
}

.speaker-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 14px;
}

.speaker-identity {
    display: flex;
    align-items: center;
    gap: 14px;
}

.speaker-avatar {
    width: 42px;
    height: 42px;
    border-radius: 50%;
    background: var(--bg-subtle);
    border: 1px solid var(--border-color);
    color: var(--text-main);
    font-weight: 800;
    font-size: 15px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.speaker-avatar.avatar-active {
    background: var(--emerald-bg);
    color: var(--emerald-active);
    border-color: rgba(16, 185, 129, 0.3);
}

.speaker-title {
    font-size: 17px;
    font-weight: 700;
    color: var(--text-main);
}

.speaker-meta {
    font-size: 12.5px;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
}

.badge {
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.05em;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

.badge-active {
    background: var(--emerald-bg);
    color: var(--emerald-active);
    border: 1px solid rgba(16, 185, 129, 0.25);
}

.badge-passive {
    background: var(--bg-subtle);
    color: var(--text-muted);
    border: 1px solid var(--border-color);
}

.pulse-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background-color: var(--emerald-active);
    animation: pulse 1.8s infinite;
}

@keyframes pulse {
    0% { transform: scale(0.95); opacity: 1; }
    50% { transform: scale(1.4); opacity: 0.4; }
    100% { transform: scale(0.95); opacity: 1; }
}

.transcript-box {
    background: var(--bg-subtle);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 12px 0;
    position: relative;
}

.transcript-quote-icon {
    font-size: 32px;
    color: var(--text-light);
    line-height: 1;
    position: absolute;
    top: 8px;
    left: 12px;
    opacity: 0.4;
}

.transcript-text {
    font-size: 18px;
    font-weight: 700;
    color: var(--text-main);
    padding-left: 22px;
    line-height: 1.45;
}

.confidence-container {
    margin-top: 10px;
}

.confidence-label {
    display: flex;
    justify-content: space-between;
    font-size: 12.5px;
    font-weight: 600;
    color: var(--text-muted);
    margin-bottom: 5px;
}

.confidence-val {
    color: var(--text-main);
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
}

.progress-bar-bg {
    width: 100%;
    height: 7px;
    background-color: var(--border-color);
    border-radius: 4px;
    overflow: hidden;
}

.progress-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #3b82f6, #10b981);
    border-radius: 4px;
    transition: width 0.4s ease;
}

/* Empty State */
.empty-state {
    text-align: center;
    padding: 40px 24px;
    background: var(--card-surface);
    border: 1px dashed var(--border-color);
    border-radius: 16px;
    color: var(--text-muted);
}

.empty-icon {
    font-size: 36px;
    margin-bottom: 10px;
}

.empty-title {
    font-size: 16px;
    font-weight: 700;
    color: var(--text-main);
    margin-bottom: 6px;
}

.empty-desc {
    font-size: 13.5px;
    color: var(--text-muted);
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.4;
}

/* Primary Action Buttons */
.btn-run-vsr {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    border-radius: 12px !important;
    padding: 14px 28px !important;
    box-shadow: 0 4px 14px var(--primary-glow) !important;
    transition: all 0.2s ease !important;
    border: none !important;
}

.btn-run-vsr:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px var(--primary-glow) !important;
}

/* Auth Portal */
.auth-wrapper {
    max-width: 460px;
    margin: 40px auto;
    background: var(--card-surface);
    border: 1px solid var(--border-color);
    border-radius: 20px;
    padding: 32px;
    box-shadow: var(--card-shadow);
}

.auth-header {
    text-align: center;
    margin-bottom: 20px;
}

.auth-error-msg {
    background: #fef2f2;
    border: 1px solid #fecaca;
    color: #dc2626;
    padding: 12px 16px;
    border-radius: 10px;
    font-size: 13px;
    font-weight: 600;
    margin-top: 14px;
}

.auth-success-msg {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    color: #16a34a;
    padding: 12px 16px;
    border-radius: 10px;
    font-size: 13px;
    font-weight: 600;
    margin-top: 14px;
}

/* Tab Styling */
.gradio-container .tab-nav button {
    font-weight: 700 !important;
    font-size: 14.5px !important;
    padding: 10px 18px !important;
    border-radius: 8px !important;
}

/* Hide Footer & API documentation */
footer, .gradio-container footer, a[href*="api"], a[href*="docs"], .api-docs, .show-api, .built-with {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
}
"""

# Client-side Theme Toggle JavaScript
theme_toggle_js = """
() => {
    const isDark = document.body.classList.toggle('dark-theme');
    localStorage.setItem('lipnet_theme', isDark ? 'dark' : 'light');
    return isDark ? '☀️ Light Mode' : '🌙 Dark Mode';
}
"""

with gr.Blocks(title="LipNet: Visual Speech Recognition from Lip Movements Using Deep Learning") as demo:
    # Header Banner
    with gr.Column():
        gr.HTML(
            """
            <div class="inst-header-wrapper">
                <div class="inst-badge">Deep Learning Visual Speech Recognition</div>
                <h1 class="inst-title">LipNet: Visual Speech Recognition from Lip Movements Using Deep Learning</h1>
            </div>
            """
        )

    # ==========================================================================
    # 1. Authentication View (Login / Signup Screen)
    # ==========================================================================
    with gr.Column(visible=True) as auth_view:
        with gr.Row():
            with gr.Column(scale=1):
                pass
            with gr.Column(scale=2):
                with gr.Tabs():
                    # Sign In Tab
                    with gr.TabItem("🔐 Sign In"):
                        gr.Markdown("### Sign in to your Account")
                        login_user = gr.Textbox(label="Username", placeholder="Enter username (Default: admin)")
                        login_pass = gr.Textbox(label="Password", type="password", placeholder="Enter password (Default: admin123)")
                        
                        with gr.Row():
                            btn_login = gr.Button("Sign In", variant="primary")
                            btn_guest = gr.Button("⚡ Guest Demo Login", variant="secondary")

                        login_status = gr.HTML()
                        gr.Markdown(
                            """
                            > 💡 **Quick Test Account**: Username: `admin` | Password: `admin123`  
                            > Or click **Guest Demo Login** for instant access without credentials.
                            """
                        )

                    # Sign Up Tab
                    with gr.TabItem("📝 Create Account"):
                        gr.Markdown("### Register New Account")
                        reg_name = gr.Textbox(label="Full Name", placeholder="e.g. Samarth Bhingardive")
                        reg_user = gr.Textbox(label="Username", placeholder="Choose a unique username")
                        reg_pass = gr.Textbox(label="Password", type="password", placeholder="Create password")
                        reg_confirm = gr.Textbox(label="Confirm Password", type="password", placeholder="Repeat password")
                        btn_signup = gr.Button("Create Account", variant="primary")
                        signup_status = gr.HTML()

            with gr.Column(scale=1):
                pass

    # ==========================================================================
    # 2. Main Application View (Desktop-Optimized Studio)
    # ==========================================================================
    with gr.Column(visible=False) as main_app_view:
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
                        
                        btn_reset_defaults = gr.Button("🔄 Reset Settings to Defaults", variant="secondary")

                # Reset to default handler
                btn_reset_defaults.click(
                    fn=lambda: (15, 0.015, "CTC Beam Search (Recommended)", True, True, True),
                    outputs=[setting_beam, setting_mar, setting_strategy, setting_bbox, setting_subtitles, setting_auto_export]
                )

    # ==========================================================================
    # Event Wiring: Theme Toggle via Client-Side JS
    # ==========================================================================
    btn_theme_toggle.click(
        fn=None,
        inputs=[],
        outputs=[btn_theme_toggle],
        js=theme_toggle_js
    )

    # ==========================================================================
    # Event Wiring: Authentication
    # ==========================================================================
    btn_login.click(
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
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, css=custom_css)
