"""
app.py
------
Streamlit dark-themed dashboard for the Speech Performance Evaluator.
Run with:  streamlit run app.py
"""

import os
import tempfile
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go

# Modules folder
from modules.audio_loader      import load_audio
from modules.acoustic_analyser import (
    extract_pitch_stability,
    extract_speech_rate,
    extract_energy_consistency,
    extract_pause_ratio,
)
from modules.transcriber import transcribe_audio

# Root-level evaluator
from speech_evaluator import (
    SpeechEvaluator,
    ContextConfig,
    AcousticFeatures,
    SemanticFeatures,
    extract_sentiment_score,
    extract_keyword_density,
    extract_clarity_score,
)


# ===========================================================================
# Page config
# ===========================================================================

st.set_page_config(
    page_title="Speech Evaluator",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ===========================================================================
# Dark theme CSS
# ===========================================================================

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Space+Grotesk:wght@300;500;700&display=swap');

  html, body, [class*="css"] {
    background-color: #0D0F14 !important;
    color: #E2E8F0 !important;
    font-family: 'Space Grotesk', sans-serif;
  }

  .stApp { background-color: #0D0F14; max-width: 680px; margin: 0 auto; }

  .metric-card {
    background: linear-gradient(135deg, #1A1E2E 0%, #141824 100%);
    border: 1px solid #2D3452;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.5rem 0;
  }

  .score-ring {
    font-family: 'JetBrains Mono', monospace;
    font-size: 4rem;
    font-weight: 700;
    text-align: center;
    background: linear-gradient(135deg, #60A5FA, #A78BFA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }

  .grade-badge {
    display: inline-block;
    padding: 4px 16px;
    border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    font-weight: 700;
  }

  .feedback-item {
    border-left: 3px solid #3B82F6;
    padding: 0.4rem 0.8rem;
    margin: 0.3rem 0;
    background: #161B2E;
    border-radius: 0 8px 8px 0;
    font-size: 0.85rem;
    color: #94A3B8;
  }

  .stFileUploader {
    background: #1A1E2E;
    border-radius: 12px;
    border: 1px dashed #3B82F6;
  }

  .stSelectbox > div > div {
    background: #1A1E2E !important;
    border-color: #2D3452 !important;
  }

  .stButton > button {
    background: linear-gradient(135deg, #3B82F6, #8B5CF6);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    width: 100%;
    font-size: 1rem;
    letter-spacing: 0.05em;
  }

  h1 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem !important;
    background: linear-gradient(90deg, #60A5FA, #A78BFA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }

  .section-label {
    color: #64748B;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.3rem;
  }
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# Radar chart
# ===========================================================================

def draw_radar_chart(dimension_scores: dict, context_name: str):
    categories = [k.replace("_", " ").title() for k in dimension_scores]
    values     = list(dimension_scores.values())

    # Close the polygon
    cats   = categories + [categories[0]]
    vals   = values     + [values[0]]

    fig = go.Figure()

    # Reference ring at 70
    fig.add_trace(go.Scatterpolar(
        r         = [70] * len(cats),
        theta     = cats,
        fill      = "toself",
        fillcolor = "rgba(59,130,246,0.05)",
        line      = dict(color="#3B82F6", width=1, dash="dot"),
        name      = "Target (70)",
    ))

    # Actual scores
    fig.add_trace(go.Scatterpolar(
        r         = vals,
        theta     = cats,
        fill      = "toself",
        fillcolor = "rgba(139,92,246,0.2)",
        line      = dict(color="#A78BFA", width=2.5),
        name      = "Your Score",
    ))

    fig.update_layout(
        polar=dict(
            bgcolor      = "rgba(0,0,0,0)",
            radialaxis   = dict(
                visible   = True,
                range     = [0, 100],
                gridcolor = "#1E2438",
                linecolor = "#1E2438",
                tickfont  = dict(color="#475569", size=9),
                tickvals  = [20, 40, 60, 80, 100],
            ),
            angularaxis  = dict(
                gridcolor = "#1E2438",
                linecolor = "#2D3452",
                tickfont  = dict(color="#94A3B8", size=10),
            ),
        ),
        showlegend    = True,
        legend        = dict(font=dict(color="#94A3B8", size=10), bgcolor="#0D0F14"),
        paper_bgcolor = "#0D0F14",
        plot_bgcolor  = "#0D0F14",
        margin        = dict(l=40, r=40, t=40, b=20),
        title         = dict(
            text = f"<b>{context_name}</b>",
            font = dict(color="#64748B", size=11),
            x    = 0.5,
        ),
    )
    return fig


# ===========================================================================
# Grade colours
# ===========================================================================

GRADE_COLORS = {
    "A": "#22C55E",
    "B": "#60A5FA",
    "C": "#F59E0B",
    "D": "#F97316",
    "F": "#EF4444",
}


# ===========================================================================
# Main app
# ===========================================================================

def main():

    st.markdown("# 🎙️ Speech Evaluator")
    st.markdown(
        '<p style="color:#475569; font-size:0.85rem;">'
        "Multi-context acoustic &amp; semantic analysis"
        "</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # -- Context selector -----------------------------------------------------
    available = ContextConfig.available_contexts()
    st.markdown('<p class="section-label">Evaluation Context</p>', unsafe_allow_html=True)
    selected = st.selectbox(
        "",
        options          = available,
        format_func      = lambda x: x.replace("_", " ").title(),
        label_visibility = "collapsed",
    )

    cfg = ContextConfig(selected)
    st.markdown(
        f'<p style="color:#64748B; font-size:0.8rem; margin-top:-0.5rem;">'
        f"{cfg.description}</p>",
        unsafe_allow_html=True,
    )

    # -- File uploader --------------------------------------------------------
    st.markdown("")
    st.markdown('<p class="section-label">Upload Audio</p>', unsafe_allow_html=True)
    audio_file = st.file_uploader(
        "",
        type             = ["wav", "mp3", "m4a", "ogg", "flac"],
        label_visibility = "collapsed",
    )

    if audio_file:
        st.audio(audio_file, format="audio/wav")

    st.markdown("")
    run = st.button("▶  Analyse Speech")

    # -- Analysis pipeline ----------------------------------------------------
    if run and audio_file:

        suffix = Path(audio_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name

        try:
            with st.spinner("Transcribing with Whisper..."):
                transcript = transcribe_audio(tmp_path)

            with st.spinner("Extracting acoustic features..."):
                audio_data = load_audio(tmp_path)
                word_count = len(transcript.split())
                acoustic   = AcousticFeatures(
                    pitch_stability    = extract_pitch_stability(audio_data),
                    speech_rate        = extract_speech_rate(word_count, audio_data.duration),
                    energy_consistency = extract_energy_consistency(audio_data),
                    pause_ratio        = extract_pause_ratio(audio_data),
                )

            with st.spinner("Analysing semantics..."):
                semantic = SemanticFeatures(
                    sentiment_score = extract_sentiment_score(transcript),
                    keyword_density = extract_keyword_density(transcript, selected),
                    clarity_score   = extract_clarity_score(transcript),
                )

        finally:
            os.unlink(tmp_path)  # always clean up the temp file

        # -- Evaluate ---------------------------------------------------------
        evaluator = SpeechEvaluator(selected)
        result    = evaluator.evaluate(acoustic, semantic)

        # -- Render results ---------------------------------------------------
        st.divider()
        st.markdown("### Results")

        col_score, col_grade = st.columns([2, 1])

        with col_score:
            st.markdown(
                f'<div class="score-ring">{result.final_score:.0f}</div>'
                '<p style="text-align:center; color:#475569; font-size:0.8rem;">/ 100</p>',
                unsafe_allow_html=True,
            )

        with col_grade:
            color = GRADE_COLORS.get(result.grade, "#94A3B8")
            st.markdown(f"""
            <div style="display:flex; align-items:center;
                        justify-content:center; height:100%;">
              <div class="grade-badge"
                   style="background:{color}22; color:{color};
                          border:1px solid {color}44;
                          font-size:2.5rem; padding:12px 28px;">
                {result.grade}
              </div>
            </div>""", unsafe_allow_html=True)

        # Radar chart
        st.plotly_chart(
            draw_radar_chart(result.dimension_scores, result.context_name),
            use_container_width=True,
        )

        # Dimension breakdown bars
        st.markdown('<p class="section-label">Dimension Breakdown</p>', unsafe_allow_html=True)
        for feature, score in result.dimension_scores.items():
            bar_color = (
                "#22C55E" if score >= 80 else
                "#F59E0B" if score >= 60 else
                "#EF4444"
            )
            st.markdown(f"""
            <div class="metric-card">
              <div style="display:flex; justify-content:space-between;
                          margin-bottom:6px;">
                <span style="font-size:0.85rem;">
                  {feature.replace("_", " ").title()}
                </span>
                <span style="font-family:'JetBrains Mono',monospace;
                             color:{bar_color};">
                  {score:.0f}
                </span>
              </div>
              <div style="background:#1E2438; border-radius:4px; height:6px;">
                <div style="width:{score}%; background:{bar_color};
                            height:6px; border-radius:4px;"></div>
              </div>
            </div>""", unsafe_allow_html=True)

        # Coaching feedback
        st.markdown("")
        st.markdown('<p class="section-label">Coaching Feedback</p>', unsafe_allow_html=True)
        for feature, tip in result.feedback.items():
            st.markdown(
                f'<div class="feedback-item">💡 {tip}</div>',
                unsafe_allow_html=True,
            )

        # Raw transcript
        with st.expander("📝 Transcript"):
            st.markdown(
                f'<p style="color:#94A3B8; font-size:0.85rem; line-height:1.7;">'
                f"{transcript}</p>",
                unsafe_allow_html=True,
            )

    elif run and not audio_file:
        st.warning("Please upload an audio file first.")


if __name__ == "__main__":
    main()