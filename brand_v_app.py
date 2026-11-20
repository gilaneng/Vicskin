"""Brand V - face contour line scanning (standalone Streamlit app).
Run: streamlit run brand_v_app.py   (needs moire_contour.py beside it)
"""
import cv2
import numpy as np
import streamlit as st

import moire_contour as mc

st.set_page_config(page_title="Brand V Contour Scan", layout="wide")
st.title("Brand V - Face Contour Line Scan")
st.caption("Front-facing photo, same distance, same lighting, head straight.")


def load(upload):
    if upload is None:
        return None, None
    img = cv2.imdecode(np.frombuffer(upload.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    return img, mc.get_landmarks(img)


def check(name, img, pts):
    if pts is None:
        st.error(f"{name}: no face detected.")
        return False
    for msg in mc.quality_check(img):
        st.warning(f"{name}: {msg}")
    ok, pose = mc.pose_ok(pts)
    if not ok:
        st.warning(f"{name}: head turned/tilted (yaw {pose['yaw']:.1f}, "
                   f"roll {pose['roll']:.1f}). Retake for accuracy.")
    return True


lines_n = st.sidebar.slider("Contour density (lines across face)", 12, 48, 24)
mode = st.sidebar.radio("Mode", ["Single scan", "Before / After", "Symmetry"])

col1, col2 = st.columns(2)
before = col1.file_uploader("Photo" if mode != "Before / After" else "Before",
                            type=["jpg", "jpeg", "png"], key="b")
after = col2.file_uploader("After", type=["jpg", "jpeg", "png"], key="a") \
    if mode == "Before / After" else None

b_img, b_pts = load(before)

if b_img is not None and check("Photo", b_img, b_pts):
    if mode == "Single scan":
        ov, _, _ = mc.render_contour(b_img, b_pts, lines_n, color=(255, 255, 255))
        c1, c2 = st.columns(2)
        c1.image(b_img[..., ::-1], caption="Original")
        c2.image(ov[..., ::-1], caption="Contour lines")
    elif mode == "Symmetry":
        st.image(mc.symmetry_map(b_img, b_pts, lines_n)[..., ::-1],
                 caption="Red = left side, Green = mirrored right side")
    else:
        a_img, a_pts = load(after)
        if a_img is not None and check("After", a_img, a_pts):
            comp, rep = mc.compare_contours(b_img, b_pts, a_img, a_pts, lines_n)
            st.image(comp[..., ::-1], caption="Red = before, Green = after")
            m1, m2 = st.columns(2)
            m1.metric("Jaw lift (% face height)",
                      f"{rep['jaw_lift_pct_face_height']:.1f}%")
            m2.metric("Mean depth change", f"{rep['mean_abs_depth_change_pct']:.1f}%")
else:
    st.info("Upload a photo to begin.")
