"""
================================================================================
UI Templates & Custom Component Renderers for LipNet-VSR
================================================================================
Provides modern, accessible, professional HTML components:
- Interactive Sticky Navbar with Theme Switcher & GitHub link
- Hero Section with Animated Neural Lip Graphics & CTAs
- 6-Feature Grid Showcase
- 6-Step "How It Works" Visual Pipeline
- Deep-Dive "About & Architecture" Section with Real-World Applications
- Technology Stack Section
- High-Impact Prediction Cards with Copy-to-Clipboard & Progress Bars
- Real-Time KPI Metric Cards & Processing Skeleton Loaders
"""

from typing import List, Dict, Any, Optional
import html


class UITemplates:
    """Renders custom modern HTML components for the LipNet-VSR web application."""

    @staticmethod
    def render_navbar(user_display: str = "Guest User", is_auth: bool = False) -> str:
        """Renders modern sticky navigation bar."""
        auth_badge = (
            f'<span class="nav-user-pill">👤 {html.escape(user_display)}</span>'
            if is_auth
            else '<span class="nav-user-pill">⚡ Guest Mode</span>'
        )
        return f"""
        <header class="app-navbar">
            <div class="nav-container">
                <a href="#home" class="nav-brand">
                    <div class="brand-icon-wrapper">
                        <span class="brand-icon">👄</span>
                        <span class="brand-pulse"></span>
                    </div>
                    <div class="brand-text">
                        <span class="brand-name">LipNet <span class="brand-highlight">AI</span></span>
                        <span class="brand-sub">Visual Speech Recognition</span>
                    </div>
                </a>

                <nav class="nav-links">
                    <a href="#home" class="nav-link">Home</a>
                    <a href="#studio" class="nav-link">Lip Reading</a>
                    <a href="#how-it-works" class="nav-link">How It Works</a>
                    <a href="#about" class="nav-link">About</a>
                    <a href="https://github.com/SakshiKiranKekan/LipNet-VSR" target="_blank" rel="noopener noreferrer" class="nav-link nav-github">
                        <svg class="github-icon" viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
                            <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/>
                        </svg>
                        <span>GitHub</span>
                    </a>
                </nav>

                <div class="nav-actions">
                    {auth_badge}
                    <a href="#studio" class="btn-nav-primary">Try Now</a>
                </div>
            </div>
        </header>
        """

    @staticmethod
    def render_hero_section() -> str:
        """Renders landing hero section with animated visual preview and statistics."""
        return """
        <section id="home" class="hero-section">
            <div class="hero-container">
                <div class="hero-content">
                    <div class="hero-badge">
                        <span class="badge-sparkle">✨</span>
                        <span>State-of-the-Art Deep Learning & Computer Vision</span>
                    </div>

                    <h1 class="hero-title">
                        AI-Powered <span class="gradient-text">Lip Reading</span>
                    </h1>

                    <p class="hero-subtitle">
                        Convert silent lip movements into meaningful text in real time using 3D Spatiotemporal Convolutional Neural Networks, Bi-directional GRUs, and Biometric Face Mesh Tracking.
                    </p>

                    <div class="hero-cta-group">
                        <a href="#studio" class="hero-btn-primary">
                            <span>Start Lip Reading</span>
                            <span class="btn-arrow">→</span>
                        </a>
                        <a href="#how-it-works" class="hero-btn-secondary">
                            <span>How It Works</span>
                        </a>
                    </div>

                    <div class="hero-stats">
                        <div class="hero-stat-item">
                            <span class="stat-num">95.2%</span>
                            <span class="stat-lbl">GRID Corpus Accuracy</span>
                        </div>
                        <div class="hero-stat-divider"></div>
                        <div class="hero-stat-item">
                            <span class="stat-num">468</span>
                            <span class="stat-lbl">Facial Biometric Landmarks</span>
                        </div>
                        <div class="hero-stat-divider"></div>
                        <div class="hero-stat-item">
                            <span class="stat-num">&lt; 150ms</span>
                            <span class="stat-lbl">Inference Latency</span>
                        </div>
                    </div>
                </div>

                <div class="hero-visual">
                    <div class="visual-card">
                        <div class="visual-header">
                            <div class="traffic-dots">
                                <span class="dot dot-red"></span>
                                <span class="dot dot-yellow"></span>
                                <span class="dot dot-green"></span>
                            </div>
                            <span class="visual-badge">LIVE INFERENCE PREVIEW</span>
                        </div>

                        <div class="visual-body">
                            <div class="lip-mesh-container">
                                <div class="lip-scan-line"></div>
                                <div class="lip-mesh-graphic">
                                    <svg viewBox="0 0 200 120" class="lip-svg">
                                        <!-- Outer lip contour -->
                                        <path d="M 20,60 Q 60,30 100,45 Q 140,30 180,60 Q 140,90 100,75 Q 60,90 20,60 Z" class="lip-outer-path"/>
                                        <!-- Inner lip contour -->
                                        <path d="M 40,60 Q 70,50 100,55 Q 130,50 160,60 Q 130,70 100,65 Q 70,70 40,60 Z" class="lip-inner-path"/>
                                        <!-- Biometric landmark dots -->
                                        <circle cx="20" cy="60" r="3" class="landmark-dot landmark-corner"/>
                                        <circle cx="60" cy="38" r="2.5" class="landmark-dot"/>
                                        <circle cx="100" cy="45" r="3" class="landmark-dot landmark-center"/>
                                        <circle cx="140" cy="38" r="2.5" class="landmark-dot"/>
                                        <circle cx="180" cy="60" r="3" class="landmark-dot landmark-corner"/>
                                        <circle cx="140" cy="82" r="2.5" class="landmark-dot"/>
                                        <circle cx="100" cy="75" r="3" class="landmark-dot landmark-center"/>
                                        <circle cx="60" cy="82" r="2.5" class="landmark-dot"/>
                                    </svg>
                                </div>
                            </div>

                            <div class="visual-output-preview">
                                <div class="preview-tag">Predicted Transcript</div>
                                <div class="preview-text">" hello world "</div>
                                <div class="preview-meta">
                                    <span class="preview-conf">Confidence: <strong>94.8%</strong></span>
                                    <span class="preview-status">● VERIFIED</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>
        """

    @staticmethod
    def render_features_grid() -> str:
        """Renders 6 modern interactive feature cards."""
        return """
        <section class="features-section">
            <div class="section-container">
                <div class="section-header">
                    <span class="section-badge">CORE CAPABILITIES</span>
                    <h2 class="section-title">Engineered for Precision & Real-Time Performance</h2>
                    <p class="section-desc">State-of-the-art computer vision algorithms fused with deep recurrent spatiotemporal architectures.</p>
                </div>

                <div class="features-grid">
                    <!-- Feature 1 -->
                    <div class="feature-card">
                        <div class="feature-icon-box icon-purple">
                            <span>🧠</span>
                        </div>
                        <h3 class="feature-title">3D-CNN & BiGRU Architecture</h3>
                        <p class="feature-text">Spatiotemporal 3D convolutions capture fine-grained lip dynamics, while 2-layer Bi-directional GRUs model phonetic temporal dependencies.</p>
                    </div>

                    <!-- Feature 2 -->
                    <div class="feature-card">
                        <div class="feature-icon-box icon-blue">
                            <span>⚡</span>
                        </div>
                        <h3 class="feature-title">Real-Time Video Processing</h3>
                        <p class="feature-text">High-throughput frame pipeline with adaptive spatiotemporal interpolation, 2D affine lip leveling, and CLAHE contrast normalization.</p>
                    </div>

                    <!-- Feature 3 -->
                    <div class="feature-card">
                        <div class="feature-icon-box icon-cyan">
                            <span>👁️</span>
                        </div>
                        <h3 class="feature-title">468-Point MediaPipe Mesh</h3>
                        <p class="feature-text">High-confidence 3D facial topology tracking isolates the 46×96 mouth region of interest while rejecting non-human background noise.</p>
                    </div>

                    <!-- Feature 4 -->
                    <div class="feature-card">
                        <div class="feature-icon-box icon-emerald">
                            <span>🗣️</span>
                        </div>
                        <h3 class="feature-title">Multi-Speaker Diarization</h3>
                        <p class="feature-text">Persistent IoU and spatial tracking with Mouth Aspect Ratio (MAR) variance detects who is actively talking versus passively listening.</p>
                    </div>

                    <!-- Feature 5 -->
                    <div class="feature-card">
                        <div class="feature-icon-box icon-amber">
                            <span>👄</span>
                        </div>
                        <h3 class="feature-title">Kinematic Viseme Engine</h3>
                        <p class="feature-text">Translates 3D vertical aperture, bilabial closures (P, B, M), and syllable cadence directly into continuous speech distributions.</p>
                    </div>

                    <!-- Feature 6 -->
                    <div class="feature-card">
                        <div class="feature-icon-box icon-indigo">
                            <span>🎯</span>
                        </div>
                        <h3 class="feature-title">CTC Language Model Rescorer</h3>
                        <p class="feature-text">Beam Search CTC decoding rescored against N-gram language models and English dictionaries to eliminate character repetitions.</p>
                    </div>
                </div>
            </div>
        </section>
        """

    @staticmethod
    def render_how_it_works_section() -> str:
        """Renders 6-step visual pipeline with connecting visual flow."""
        return """
        <section id="how-it-works" class="how-section">
            <div class="section-container">
                <div class="section-header">
                    <span class="section-badge">PIPELINE ARCHITECTURE</span>
                    <h2 class="section-title">How LipNet Visual Speech Recognition Works</h2>
                    <p class="section-desc">From raw pixel stream to decoded sentence in six synchronized spatiotemporal stages.</p>
                </div>

                <div class="steps-grid">
                    <div class="step-card">
                        <div class="step-num">01</div>
                        <div class="step-icon">📹</div>
                        <h3 class="step-title">Video Ingestion</h3>
                        <p class="step-text">Accepts webcam streams or video files (MP4, MPG, AVI, WebM) and standardizes to 25 FPS.</p>
                    </div>

                    <div class="step-card">
                        <div class="step-num">02</div>
                        <div class="step-icon">👤</div>
                        <h3 class="step-title">Biometric Face Mesh</h3>
                        <p class="step-text">MediaPipe 468 landmarks verify human facial geometry (eyes, nose, mouth triangle hierarchy).</p>
                    </div>

                    <div class="step-card">
                        <div class="step-num">03</div>
                        <div class="step-icon">👄</div>
                        <h3 class="step-title">Lip ROI Extraction</h3>
                        <p class="step-text">Isolates 46×96 mouth crop with 2D Affine leveling and CLAHE adaptive histogram equalization.</p>
                    </div>

                    <div class="step-card">
                        <div class="step-num">04</div>
                        <div class="step-icon">🎞️</div>
                        <h3 class="step-title">Sequence Resampling</h3>
                        <p class="step-text">Visual VAD extracts active speech windows and resamples to uniform 75-timestep tensor sequence.</p>
                    </div>

                    <div class="step-card">
                        <div class="step-num">05</div>
                        <div class="step-icon">🧠</div>
                        <h3 class="step-title">Deep Learning Inference</h3>
                        <p class="step-text">3D-CNN layers extract spatiotemporal features, and BiGRU units compute temporal character logits.</p>
                    </div>

                    <div class="step-card">
                        <div class="step-num">06</div>
                        <div class="step-icon">📝</div>
                        <h3 class="step-title">CTC Transcript & Subtitles</h3>
                        <p class="step-text">Beam Search decoding generates clean text, confidence metrics, and annotated video subtitle tracks.</p>
                    </div>
                </div>
            </div>
        </section>
        """

    @staticmethod
    def render_about_section() -> str:
        """Renders About project, technology stack, and real-world applications."""
        return """
        <section id="about" class="about-section">
            <div class="section-container">
                <div class="section-header">
                    <span class="section-badge">ABOUT PROJECT</span>
                    <h2 class="section-title">Automated Lip Reading & Visual Speech Recognition</h2>
                    <p class="section-desc">Pioneering silent communication and assistive technology through deep spatiotemporal neural networks.</p>
                </div>

                <div class="about-grid">
                    <div class="about-card">
                        <h3 class="about-card-title">📖 What is LipNet?</h3>
                        <p class="about-card-text">
                            LipNet is an end-to-end deep learning model that maps sequences of video frames of human lip movements directly to text sequences. Unlike traditional phoneme-based approaches, LipNet performs sentence-level sequence-to-sequence visual speech recognition without requiring audio signals.
                        </p>
                    </div>

                    <div class="about-card">
                        <h3 class="about-card-title">🎯 Key Problem Solved</h3>
                        <p class="about-card-text">
                            Human lip-reading is notoriously difficult (commercial lip readers achieve only ~12-20% accuracy due to viseme ambiguities). LipNet leverages spatiotemporal convolutions, recurrent temporal memory, and language model beam rescoring to achieve over 95% accuracy on benchmark corpora.
                        </p>
                    </div>
                </div>

                <!-- Applications Grid -->
                <div class="apps-container">
                    <h3 class="apps-heading">🌍 Real-World Applications</h3>
                    <div class="apps-grid">
                        <div class="app-item">
                            <span class="app-icon">♿</span>
                            <div class="app-info">
                                <h4 class="app-title">Assistive Technology & Dysarthria</h4>
                                <p class="app-desc">Empowers individuals with vocal cord impairment, laryngectomy, or severe ALS to communicate silently.</p>
                            </div>
                        </div>

                        <div class="app-item">
                            <span class="app-icon">🔇</span>
                            <div class="app-info">
                                <h4 class="app-title">High-Noise Speech Recognition</h4>
                                <p class="app-desc">Overcomes extreme acoustic noise in industrial manufacturing, cockpits, airports, and construction sites.</p>
                            </div>
                        </div>

                        <div class="app-item">
                            <span class="app-icon">🔒</span>
                            <div class="app-info">
                                <h4 class="app-title">Silent & Covert Operations</h4>
                                <p class="app-desc">Enables silent PIN/password entry, secure authentication, and tactical communications without acoustic leakage.</p>
                            </div>
                        </div>

                        <div class="app-item">
                            <span class="app-icon">🕵️</span>
                            <div class="app-info">
                                <h4 class="app-title">Forensic & Silent Video Analysis</h4>
                                <p class="app-desc">Transcribes silent security CCTV footage, historical archives, and noisy broadcast television.</p>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Tech Stack Pills -->
                <div class="tech-stack-wrapper">
                    <h3 class="tech-heading">🛠️ Technology Stack</h3>
                    <div class="tech-pills">
                        <span class="tech-pill">🐍 Python 3.11</span>
                        <span class="tech-pill">👁️ OpenCV 4.10</span>
                        <span class="tech-pill">🧠 TensorFlow / Keras</span>
                        <span class="tech-pill">⚡ PyTorch</span>
                        <span class="tech-pill">📐 MediaPipe 468 Mesh</span>
                        <span class="tech-pill">🎙️ OpenAI Whisper</span>
                        <span class="tech-pill">🌐 Gradio 4.x</span>
                        <span class="tech-pill">📊 NumPy 1.26</span>
                        <span class="tech-pill">🗄️ SQLite3 Auth</span>
                    </div>
                </div>
            </div>
        </section>
        """

    @staticmethod
    def render_speaker_card(speaker: Dict[str, Any]) -> str:
        """Renders an individual speaker transcript card with visual badges, copy button, and progress metrics."""
        speaker_id = html.escape(speaker.get("speaker_id", "Speaker 1"))
        raw_transcript = speaker.get("transcript", "No speech detected")
        transcript = html.escape(raw_transcript)
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

        escaped_js_text = raw_transcript.replace("'", "\\'").replace('"', '&quot;')

        card_html = f"""
        <div class="speaker-card {'active-border' if is_active else ''}">
            <div class="speaker-card-header">
                <div class="speaker-identity">
                    <div class="speaker-avatar {'avatar-active' if is_active else ''}">
                        {speaker_id[0] if speaker_id else 'S'}{speaker_id[-1] if speaker_id and speaker_id[-1].isdigit() else '1'}
                    </div>
                    <div>
                        <div class="speaker-title">{speaker_id}</div>
                        <div class="speaker-meta">{frame_count} verified human lip frames</div>
                    </div>
                </div>
                <div class="header-actions">
                    {status_badge}
                </div>
            </div>

            <div class="transcript-box">
                <div class="transcript-quote-icon">“</div>
                <div class="transcript-text">{transcript}</div>
                <button class="btn-copy-transcript" onclick="navigator.clipboard.writeText('{escaped_js_text}'); this.innerText='✓ Copied!';" title="Copy to clipboard">
                    📋 Copy Text
                </button>
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
                <div class="empty-desc">Upload a video or select a dataset sample from the left panel, then click "Start Lip Reading Prediction" to view real-time decoded speech.</div>
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
