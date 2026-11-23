"""Brand V - face contour line scanning (standalone Streamlit app).
Run: streamlit run brand_v_app.py   (needs moire_contour.py beside it)
"""
import base64

import cv2
import numpy as np
import streamlit as st

import moire_contour as mc

st.set_page_config(page_title="Brand V Contour Scan", layout="wide")
st.title("Brand V - Face Contour Line Scan")
st.caption("Front-facing photo, same distance, same lighting, head straight.  [build 6]")


MAX_SIDE = 1000  # phone photos are huge; shrinking keeps the server fast


def load(upload):
    if upload is None:
        return None, None
    img = cv2.imdecode(np.frombuffer(upload.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None, None
    scale = MAX_SIDE / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img, mc.get_landmarks(img)


def show(container, bgr, caption):
    """Embed the image directly in the page (base64) so it never depends on
    Streamlit's temporary media storage."""
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        container.error("Could not build the result image.")
        return
    b64 = base64.b64encode(buf.tobytes()).decode()
    container.markdown(
        f'<img src="data:image/jpeg;base64,{b64}" style="width:100%;border-radius:8px">'
        f'<p style="color:gray;font-size:0.85em;margin-top:4px">{caption}</p>',
        unsafe_allow_html=True)
    kwargs = dict(file_name="contour.jpg", mime="image/jpeg", key=f"dl_{caption}")
    try:
        # newer Streamlit: don't rerun the app when the button is tapped
        container.download_button("Save image", buf.tobytes(), on_click="ignore",
                                  **kwargs)
    except TypeError:
        container.download_button("Save image", buf.tobytes(), **kwargs)
    container.caption("Tip: if the button does nothing, press and hold the image "
                      "and choose Save to Photos / Add to Photos.")


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
show_grid = st.sidebar.checkbox("Show grid", value=False)
grid_rows = st.sidebar.slider("Grid rows (forehead to chin)", 4, 20, 10,
                              disabled=not show_grid)
grid_cols = st.sidebar.slider("Grid columns", 2, 16, 8, disabled=not show_grid)
grid_hex = st.sidebar.color_picker("Grid colour", "#FFFFFF", disabled=not show_grid)
grid_mid = st.sidebar.checkbox("Show midline", value=True, disabled=not show_grid)


def with_grid(img, pts):
    if not show_grid:
        return img
    r, g, b = (int(grid_hex[i:i + 2], 16) for i in (1, 3, 5))
    return mc.draw_grid(img, pts, grid_rows, grid_cols, color=(b, g, r),
                        midline=grid_mid)

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
        show(c1, b_img, "Original")
        show(c2, with_grid(ov, b_pts), "Contour lines")
    elif mode == "Symmetry":
        show(st, with_grid(mc.symmetry_map(b_img, b_pts, lines_n), b_pts),
             "Red = left side, Green = mirrored right side")
    else:
        a_img, a_pts = load(after)
        if a_img is not None and check("After", a_img, a_pts):
            comp, rep = mc.compare_contours(b_img, b_pts, a_img, a_pts, lines_n)
            show(st, with_grid(comp, b_pts), "Red = before, Green = after")
            m1, m2 = st.columns(2)
            m1.metric("Jaw lift (% face height)",
                      f"{rep['jaw_lift_pct_face_height']:.1f}%")
            m2.metric("Mean depth change", f"{rep['mean_abs_depth_change_pct']:.1f}%")
else:
    st.info("Upload a photo to begin.")
