"""
UI Templates & Custom Component Renderers for Gradio.
Generates polished, responsive HTML cards, KPI statistics, speaker badges, and desktop layouts.
"""

from typing import List, Dict, Any, Optional
import html


class UITemplates:
    """Renders custom modern HTML components for the LipNet VSR interface."""

    @staticmethod
    def render_speaker_card(speaker: Dict[str, Any]) -> str:
        """Renders an individual speaker transcript card with visual badges and metrics."""
        speaker_id = html.escape(speaker.get("speaker_id", "Speaker"))
        transcript = html.escape(speaker.get("transcript", "No speech detected"))
        is_active = speaker.get("is_active_speaker", False)
        confidence_str = speaker.get("confidence", "0.0%")
        frame_count = speaker.get("frame_count", 0)

        # Parse confidence float
        try:
            conf_val = float(confidence_str.replace("%", "").strip())
        except ValueError:
            conf_val = 0.0

        status_badge = (
            '<span class="badge badge-active"><span class="pulse-dot"></span> ACTIVE SPEAKER</span>'
            if is_active
            else '<span class="badge badge-passive">PASSIVE LISTENER</span>'
        )

        card_html = f"""
        <div class="speaker-card {'active-border' if is_active else ''}">
            <div class="speaker-card-header">
                <div class="speaker-identity">
                    <div class="speaker-avatar {'avatar-active' if is_active else ''}">
                        {speaker_id[0] if speaker_id else 'S'}{speaker_id[-1] if speaker_id and speaker_id[-1].isdigit() else ''}
                    </div>
                    <div>
                        <div class="speaker-title">{speaker_id}</div>
                        <div class="speaker-meta">{frame_count} verified human lip frames</div>
                    </div>
                </div>
                <div>
                    {status_badge}
                </div>
            </div>

            <div class="transcript-box">
                <div class="transcript-quote-icon">“</div>
                <div class="transcript-text">{transcript}</div>
            </div>

            <div class="card-footer">
                <div class="confidence-container">
                    <div class="confidence-label">
                        <span>Recognition Confidence</span>
                        <span class="confidence-val">{confidence_str}</span>
                    </div>
                    <div class="progress-bar-bg">
                        <div class="progress-bar-fill" style="width: {min(100.0, max(5.0, conf_val))}%;"></div>
                    </div>
                </div>
            </div>
        </div>
        """
        return card_html

    @staticmethod
    def render_speakers_deck(speakers: List[Dict[str, Any]]) -> str:
        """Renders a grid of speaker cards with polished empty states."""
        if not speakers:
            return """
            <div class="empty-state">
                <div class="empty-icon">🎙️</div>
                <div class="empty-title">Ready for Visual Speech Recognition</div>
                <div class="empty-desc">Upload a video or record with your webcam, then click "Run Visual Speech Recognition" to view real-time decoded speech.</div>
            </div>
            """

        cards = [UITemplates.render_speaker_card(spk) for spk in speakers]
        return f'<div class="speakers-deck">{"".join(cards)}</div>'

    @staticmethod
    def render_kpi_stats(
        total_frames: int,
        speaker_count: int,
        active_count: int,
        avg_confidence: float
    ) -> str:
        """Renders top metric KPI summary counters."""
        return f"""
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-header">
                    <span class="kpi-icon">🎞️</span>
                    <span class="kpi-pill">Frames</span>
                </div>
                <div class="kpi-value">{total_frames}</div>
                <div class="kpi-title">Sequence Frames</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-header">
                    <span class="kpi-icon">👤</span>
                    <span class="kpi-pill">Faces</span>
                </div>
                <div class="kpi-value">{speaker_count}</div>
                <div class="kpi-title">Human Speakers</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-header">
                    <span class="kpi-icon">🗣️</span>
                    <span class="kpi-pill">Active</span>
                </div>
                <div class="kpi-value">{active_count}</div>
                <div class="kpi-title">Active Speakers</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-header">
                    <span class="kpi-icon">🎯</span>
                    <span class="kpi-pill">Precision</span>
                </div>
                <div class="kpi-value">{avg_confidence:.1f}%</div>
                <div class="kpi-title">Avg Confidence</div>
            </div>
        </div>
        """
