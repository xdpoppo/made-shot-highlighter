import streamlit as st
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from simple_make import detect
from make_detector import dedup_makes
from clip_extractor import extract_all
import cv2

st.set_page_config(page_title="Huge Highlights", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #0b0d12;
    }

    #MainMenu, footer, header { visibility: hidden; }

    .block-container {
        padding-top: 3rem;
        max-width: 980px;
    }

    .brand {
        display: flex;
        align-items: baseline;
        gap: 0.6rem;
        margin-bottom: 0.25rem;
    }
    .brand-mark {
        width: 10px;
        height: 10px;
        border-radius: 2px;
        background: linear-gradient(135deg, #ff6a3d, #ff3d77);
        display: inline-block;
        margin-right: 0.4rem;
    }
    .brand h1 {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #f5f6f8;
        margin: 0;
    }
    .subtitle {
        color: #7d8394;
        font-size: 1rem;
        margin-bottom: 2.4rem;
        font-weight: 400;
    }

    section[data-testid="stFileUploaderDropzone"] {
        background: #12151c;
        border: 1.5px dashed #2a2f3d;
        border-radius: 14px;
        transition: border-color 0.15s ease;
    }
    section[data-testid="stFileUploaderDropzone"]:hover {
        border-color: #ff5a63;
    }
    section[data-testid="stFileUploaderDropzone"] button {
        background: #1c2029;
        color: #e8e9ed;
        border: 1px solid #2a2f3d;
        border-radius: 8px;
    }
    div[data-testid="stFileUploaderDropzoneInstructions"] span,
    div[data-testid="stFileUploaderDropzoneInstructions"] small {
        color: #8b91a1 !important;
    }

    div[data-testid="stMetric"] {
        background: #12151c;
        border: 1px solid #1e222c;
        border-radius: 12px;
        padding: 1rem 1.2rem;
    }
    div[data-testid="stMetricLabel"] {
        color: #7d8394;
    }
    div[data-testid="stMetricValue"] {
        color: #f5f6f8;
    }

    .highlight-card {
        background: #12151c;
        border: 1px solid #1e222c;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1.1rem;
    }
    .highlight-title {
        color: #f5f6f8;
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 0.15rem;
    }
    .highlight-time {
        color: #ff5a63;
        font-weight: 600;
        font-size: 0.9rem;
    }

    .section-label {
        color: #f5f6f8;
        font-weight: 700;
        font-size: 1.15rem;
        margin: 2rem 0 1rem 0;
        letter-spacing: -0.01em;
    }

    .status-line {
        color: #7d8394;
        font-size: 0.85rem;
        margin-top: -1.6rem;
        margin-bottom: 2rem;
    }

    div.stButton > button {
        background: linear-gradient(135deg, #ff6a3d, #ff3d77);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
    }
    div.stDownloadButton > button {
        background: #1c2029;
        color: #e8e9ed;
        border: 1px solid #2a2f3d;
        border-radius: 8px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="brand"><span class="brand-mark"></span><h1>Huge Highlights</h1></div>
<div class="subtitle">Upload full game footage. Get every made shot cut into its own clip.</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["mp4", "mov", "avi", "mkv"], label_visibility="collapsed")

if uploaded_file is not None:
    work_dir = tempfile.mkdtemp(prefix="huge_highlights_")
    src_path = os.path.join(work_dir, "source.mp4")
    with open(src_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    clips_dir = os.path.join(work_dir, "clips")
    frames_dir = os.path.join(work_dir, "frames")

    try:
        with st.spinner("Scanning footage for made shots..."):
            makes, fps = detect(src_path, out_dir=frames_dir, verbose=False)
            makes = dedup_makes(makes, min_gap_s=7.0)

        if makes:
            with st.spinner(f"Cutting {len(makes)} highlight clip(s)..."):
                clip_paths = extract_all(src_path, makes, clips_dir, pre=6.0, post=2.5)
        else:
            clip_paths = []

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Made shots found", len(makes))
        with col2:
            st.metric("Video length", f"{(makes[-1]['time'] if makes else 0):.0f}s scanned")

        if makes:
            st.markdown('<div class="section-label">Highlights</div>', unsafe_allow_html=True)

            for i, (make, clip_path) in enumerate(zip(makes, clip_paths), 1):
                t = make["time"]
                mins, secs = int(t // 60), t % 60

                st.markdown(
                    f'<div class="highlight-card">'
                    f'<div class="highlight-title">Make {i}</div>'
                    f'<div class="highlight-time">{mins}:{secs:05.2f}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if os.path.exists(clip_path):
                    st.video(clip_path)
                    with open(clip_path, "rb") as f:
                        st.download_button(
                            f"Download clip {i}",
                            data=f.read(),
                            file_name=f"huge_highlights_make_{i:02d}.mp4",
                            mime="video/mp4",
                            key=f"dl_{i}",
                        )
        else:
            st.info("No made shots detected in this clip.")

    finally:
        pass  # work_dir cleaned up on next run / process exit; keep files alive for st.video/downloads this render
