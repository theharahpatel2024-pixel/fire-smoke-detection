"""
app.py - Streamlit web interface for Fire and Smoke Detection
Run:  streamlit run app.py
"""
import os
import tempfile

import cv2
import numpy as np
import streamlit as st

import detector as dt

st.set_page_config(page_title="Fire & Smoke Detection", page_icon="🔥", layout="wide")
st.title("🔥 Fire and Smoke Detection")
st.caption("IPCV PBL mini project | Classical image processing with OpenCV (HSV thresholding + morphology + contours)")

# ---------------- Sidebar: parameters ----------------
st.sidebar.header("Settings")
detect_fire = st.sidebar.checkbox("Detect Fire", True)
detect_smoke = st.sidebar.checkbox("Detect Smoke", True)

st.sidebar.subheader("Fire (HSV)")
fire_h_max = st.sidebar.slider("Hue max (0 to value)", 10, 60, dt.DEFAULTS["fire_h_max"])
fire_s_min = st.sidebar.slider("Saturation min", 0, 255, dt.DEFAULTS["fire_s_min"])
fire_v_min = st.sidebar.slider("Value (brightness) min", 0, 255, dt.DEFAULTS["fire_v_min"])

st.sidebar.subheader("Smoke (HSV)")
smoke_s_max = st.sidebar.slider("Saturation max", 0, 120, dt.DEFAULTS["smoke_s_max"])
smoke_v_min = st.sidebar.slider("Value min", 0, 255, dt.DEFAULTS["smoke_v_min"])
smoke_v_max = st.sidebar.slider("Value max", 0, 255, dt.DEFAULTS["smoke_v_max"])

st.sidebar.subheader("Common")
min_area_pct = st.sidebar.slider("Minimum area (% of image)", 0.1, 10.0, dt.DEFAULTS["min_area_pct"], 0.1)
blur = st.sidebar.slider("Gaussian blur kernel", 1, 15, dt.DEFAULTS["blur"], 2)
kernel = st.sidebar.slider("Morphology kernel", 1, 15, dt.DEFAULTS["kernel"], 2)

params = dict(fire_h_max=fire_h_max, fire_s_min=fire_s_min, fire_v_min=fire_v_min,
              smoke_s_max=smoke_s_max, smoke_v_min=smoke_v_min, smoke_v_max=smoke_v_max,
              min_area_pct=min_area_pct, blur=blur, kernel=kernel)


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


tab_img, tab_vid, tab_about = st.tabs(["📷 Image", "🎞️ Video", "ℹ️ How it works"])

# ---------------- Image tab ----------------
with tab_img:
    up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"], key="img")
    if up is not None:
        data = np.frombuffer(up.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            st.error("Could not read this image. Please upload a valid JPG/PNG.")
        elif not (detect_fire or detect_smoke):
            st.warning("Enable at least one of Fire / Smoke in the sidebar.")
        else:
            r = dt.detect(img, params, detect_fire=detect_fire, detect_smoke=detect_smoke)
            if r["fire_boxes"]:
                st.error(f"🔥 {r['label']}")
            elif r["smoke_boxes"]:
                st.warning(f"💨 {r['label']}")
            else:
                st.success(f"✅ {r['label']}")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Fire area", f"{r['fire_pct']:.2f} %")
            m2.metric("Smoke area", f"{r['smoke_pct']:.2f} %")
            m3.metric("Fire regions", len(r["fire_boxes"]))
            m4.metric("Smoke regions", len(r["smoke_boxes"]))

            c1, c2, c3 = st.columns(3)
            c1.image(rgb(r["original"]), caption="Original", width="stretch")
            combined = cv2.merge([r["smoke_mask"], np.zeros_like(r["fire_mask"]), r["fire_mask"]])
            c2.image(rgb(combined), caption="Mask (red = fire, blue = smoke)", width="stretch")
            c3.image(rgb(r["annotated"]), caption="Result", width="stretch")

            ok, buf = cv2.imencode(".png", r["annotated"])
            if ok:
                st.download_button("Download result image", buf.tobytes(), "result.png", "image/png")
    else:
        st.info("Upload an image to start. Use the sidebar sliders to tune the detection live.")

# ---------------- Video tab ----------------
with tab_vid:
    st.caption("Smoke is confirmed with frame differencing (it must also be moving). Keep the video short; max 300 frames are processed.")
    upv = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"], key="vid")
    if upv is not None:
        if not (detect_fire or detect_smoke):
            st.warning("Enable at least one of Fire / Smoke in the sidebar.")
        elif st.button("Run detection on video"):
            suffix = os.path.splitext(upv.name)[1] or ".mp4"
            tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp_in.write(upv.read())
            tmp_in.close()
            tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            bar = st.progress(0.0, text="Processing frames...")
            try:
                s = dt.process_video(tmp_in.name, tmp_out, params, detect_fire, detect_smoke,
                                     progress_cb=lambda v: bar.progress(v, text="Processing frames..."))
                bar.empty()
                if s["frames"] == 0:
                    st.error("No frames could be read from this video.")
                else:
                    a, b, c = st.columns(3)
                    a.metric("Frames processed", s["frames"])
                    b.metric("Frames with fire", s["fire_frames"])
                    c.metric("Frames with smoke", s["smoke_frames"])
                    if s["samples"]:
                        cols = st.columns(len(s["samples"]))
                        for col, fr in zip(cols, s["samples"]):
                            col.image(rgb(fr), width="stretch")
                    with open(tmp_out, "rb") as f:
                        st.download_button("Download result video", f.read(), "result.mp4", "video/mp4")
            except Exception as e:
                bar.empty()
                st.error(f"Video processing failed: {e}")
            finally:
                for pth in (tmp_in.name, tmp_out):
                    try:
                        os.remove(pth)
                    except OSError:
                        pass

# ---------------- About tab ----------------
with tab_about:
    st.markdown("""
**Pipeline**
1. Resize and Gaussian blur (noise removal)
2. BGR to HSV conversion
3. Colour thresholding with `cv2.inRange`
   - Fire: red-orange-yellow hue, high saturation, high brightness
   - Smoke: low saturation (grey), medium brightness
4. Morphological opening and closing (remove noise, fill holes)
5. Contours, minimum-area filter and bounding boxes
6. Decision: label + area percentage
7. Video: frame differencing to confirm that smoke is moving

**Limitations**
- Colour based, so orange clothes, sunsets or lights may give false fire detection
- Grey walls, clouds or fog may give false smoke detection
- Thresholds depend on lighting; tune them with the sidebar sliders
- Future work: CNN / YOLO based detection (Exp 8, 9)
""")
