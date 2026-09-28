"""
frontend/streamlit_app.py
──────────────────────────
Streamlit UI for the Medical Prescription Analyser.

Calls the FastAPI backend at http://localhost:8000/predict
and renders results in a rich, interactive dashboard.

Run:
    streamlit run frontend/streamlit_app.py
"""

import io
import json
import time

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Medical Prescription Analyser",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS – dark premium theme
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* Import Google Font */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* ── Hero gradient header ─────────────────────────────────── */
  .hero-header {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
  }
  .hero-header h1 {
    color: #ffffff;
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
  }
  .hero-header p {
    color: #94b4c1;
    font-size: 1.05rem;
    margin: 0.6rem 0 0;
  }

  /* ── Metric cards ─────────────────────────────────────────── */
  .metric-card {
    background: linear-gradient(145deg, #1a2a3a, #1e3448);
    border: 1px solid #2d4a6a;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }
  .metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.4);
  }
  .metric-card .label {
    font-size: 0.75rem;
    color: #7fa8c9;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
    margin-bottom: 0.4rem;
  }
  .metric-card .value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #4fc3f7;
  }

  /* ── Entity badges ───────────────────────────────────────── */
  .entity-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 0.2rem;
    letter-spacing: 0.3px;
  }
  .badge-DRUG      { background: #1a3a5c; color: #4fc3f7; border: 1px solid #1e6091; }
  .badge-DOSAGE    { background: #1a3a1a; color: #66bb6a; border: 1px solid #2e7d32; }
  .badge-STRENGTH  { background: #3a1a1a; color: #ef9a9a; border: 1px solid #c62828; }
  .badge-ROUTE     { background: #3a2a1a; color: #ffa726; border: 1px solid #e65100; }
  .badge-FREQUENCY { background: #2a1a3a; color: #ce93d8; border: 1px solid #6a1b9a; }
  .badge-DURATION  { background: #1a3a3a; color: #80deea; border: 1px solid #00838f; }
  .badge-FORM      { background: #2a2a1a; color: #fff176; border: 1px solid #f57f17; }
  .badge-DEFAULT   { background: #2a2a2a; color: #bdbdbd; border: 1px solid #424242; }

  /* ── OCR text box ────────────────────────────────────────── */
  .ocr-box {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1.2rem;
    font-family: 'Courier New', monospace;
    font-size: 0.9rem;
    color: #e6edf3;
    white-space: pre-wrap;
    line-height: 1.7;
    max-height: 300px;
    overflow-y: auto;
  }

  /* ── Section titles ──────────────────────────────────────── */
  .section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #e0e0e0;
    margin: 1.5rem 0 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, #2d4a6a, transparent);
    margin-left: 0.5rem;
  }

  /* ── Upload area ─────────────────────────────────────────── */
  [data-testid="stFileUploader"] {
    border: 2px dashed #2d4a6a !important;
    border-radius: 12px !important;
    background: #0f1923 !important;
    padding: 1rem !important;
  }

  /* ── Sidebar ─────────────────────────────────────────────── */
  [data-testid="stSidebar"] {
    background: #0d1117;
  }

  /* ── Confidence bar ─────────────────────────────────────── */
  .conf-bar-wrap {
    background: #1a2332;
    border-radius: 6px;
    height: 8px;
    margin-top: 4px;
    overflow: hidden;
  }
  .conf-bar-fill {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, #1565c0, #4fc3f7);
    transition: width 0.5s ease;
  }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000"

LABEL_COLORS = {
    "DRUG": "badge-DRUG",
    "CHEMICAL": "badge-DRUG",
    "DOSAGE": "badge-DOSAGE",
    "STRENGTH": "badge-STRENGTH",
    "ROUTE": "badge-ROUTE",
    "FREQUENCY": "badge-FREQUENCY",
    "DURATION": "badge-DURATION",
    "FORM": "badge-FORM",
}

LABEL_ICONS = {
    "DRUG": "💊",
    "CHEMICAL": "🧪",
    "DOSAGE": "⚖️",
    "STRENGTH": "💪",
    "ROUTE": "🛤️",
    "FREQUENCY": "🔁",
    "DURATION": "⏱️",
    "FORM": "💉",
}

# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def badge_html(label: str, text: str) -> str:
    css = LABEL_COLORS.get(label.upper(), "badge-DEFAULT")
    icon = LABEL_ICONS.get(label.upper(), "🔖")
    return f'<span class="entity-badge {css}">{icon} {text}</span>'


def check_backend() -> bool:
    try:
        r = httpx.get(f"{API_BASE}/", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def conf_bar(conf: float) -> str:
    pct = int(conf * 100)
    color = "#4fc3f7" if conf >= 0.8 else "#ffa726" if conf >= 0.5 else "#ef5350"
    return (
        f'<div style="font-size:0.75rem; color:#7fa8c9;">{pct}%</div>'
        f'<div class="conf-bar-wrap">'
        f'<div class="conf-bar-fill" style="width:{pct}%; background:{color};"></div>'
        f'</div>'
    )


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    backend_ok = check_backend()
    status_icon = "🟢" if backend_ok else "🔴"
    st.markdown(f"**API Status:** {status_icon} {'Online' if backend_ok else 'Offline'}")
    if not backend_ok:
        st.warning("Start the backend:\n```\nuvicorn app.main:app --reload\n```")

    st.divider()

    if backend_ok:
        try:
            info = httpx.get(f"{API_BASE}/models/status", timeout=3).json()
            st.markdown("**Active Models**")
            st.json(info)
        except Exception:
            pass

    st.divider()
    st.markdown("**About**")
    st.markdown(
        "Medical Prescription Analyser v2.0  \n"
        "Engine: **Vision-Language Model (VLM)**  \n"
        "Backend: FastAPI + Ollama"
    )

# ──────────────────────────────────────────────────────────────────────────────
# Hero header
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-header">
  <h1>💊 Medical Prescription Analyser</h1>
  <p>Upload a handwritten prescription image — our AI reads it and extracts structured medical data.</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Upload & Main Layout
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-title">📤 Upload Prescription</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Drag & drop or browse",
    type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
    label_visibility="collapsed",
)

if uploaded:
    # Split into Image (Left) and Results (Right) to avoid vertical scrolling
    col_image, col_results = st.columns([1, 1.5], gap="large")
    
    with col_image:
        st.markdown('<div class="section-title">🖼️ Original Image</div>', unsafe_allow_html=True)
        img = Image.open(uploaded)
        # Display image; Streamlit will scale it to the column width
        st.image(img, use_container_width=True)
        analyse_btn = st.button("🔬 Analyse Prescription", use_container_width=True, type="primary")

    with col_results:
        if analyse_btn:
            if not backend_ok:
                st.error("Backend is offline. Start with: `uvicorn app.main:app --reload`")
            else:
                with st.spinner("🔬 Running OCR + NLP pipeline …"):
                    uploaded.seek(0)
                    files = {"file": (uploaded.name, uploaded.read(), uploaded.type)}
                    try:
                        t0 = time.perf_counter()
                        response = httpx.post(f"{API_BASE}/predict", files=files, timeout=120)
                        elapsed = round(time.perf_counter() - t0, 2)

                        if response.status_code != 200:
                            st.error(f"Backend error {response.status_code}: {response.text}")
                            st.stop()

                        data = response.json()

                    except httpx.TimeoutException:
                        st.error("Request timed out (120 s). The model might still be loading — try again.")
                        st.stop()
                    except Exception as exc:
                        st.error(f"Unexpected error: {exc}")
                        st.stop()

                st.success(f"✅ Analysed in **{data.get('processing_time_seconds', elapsed)} s**")
                
                # ── Summary metrics ────────────────────────────────────────────
                prescription = data.get("prescription", {})
                medicines = prescription.get("medicines", [])
                doctor_notes = prescription.get("doctor_notes")

                n_drugs = len(medicines)

                m1, m2 = st.columns(2)
                m1.markdown(
                    f'<div class="metric-card">'
                    f'<div class="label">Medicines Extracted</div>'
                    f'<div class="value">{n_drugs}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                
                has_notes = "Yes" if doctor_notes else "No"
                m2.markdown(
                    f'<div class="metric-card">'
                    f'<div class="label">Doctor Notes Found</div>'
                    f'<div class="value">{has_notes}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                st.markdown("")  # spacer

                # ── Results tabs ───────────────────────────────────────────────
                tab_table, tab_notes, tab_raw = st.tabs([
                    "📋 Structured Table", "📝 Doctor Notes", "🔧 Raw JSON"
                ])

                # ─ Tab 1: Structured table ─────────────────────────────────────
                with tab_table:
                    if medicines:
                        df = pd.DataFrame(medicines)
                        # Capitalize column names for display
                        df.columns = [str(c).replace("_", " ").title() for c in df.columns]
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No structured rows to display.")

                # ─ Tab 2: Doctor Notes ─────────────────────────────────────────
                with tab_notes:
                    if doctor_notes:
                        st.markdown('<div class="section-title">Additional Instructions</div>',
                                    unsafe_allow_html=True)
                        st.markdown(f'<div class="ocr-box">{doctor_notes}</div>',
                                    unsafe_allow_html=True)
                    else:
                        st.warning("No additional notes were extracted from this prescription.")

                # ─ Tab 3: Raw JSON ─────────────────────────────────────────────
                with tab_raw:
                    st.json(data)

# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#4a6a8a; font-size:0.8rem;">'
    "Medical Prescription Analyser · Vision-Language Model · FastAPI · Streamlit"
    "</p>",
    unsafe_allow_html=True,
)
