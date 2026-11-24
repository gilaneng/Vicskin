"""Brand V - face contour analysis (standalone Streamlit app).
Needs moire_contour.py beside it; theme lives in .streamlit/config.toml."""
import base64
import json

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

import moire_contour as mc

st.set_page_config(page_title="Brand V | Contour Analysis", page_icon="✨",
                   layout="centered", initial_sidebar_state="collapsed")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600&family=Inter:wght@400;500;600&display=swap');
html, body, .stApp { font-family: 'Inter', sans-serif; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.2rem; max-width: 760px; }
.hero { background: linear-gradient(135deg,#4A2C3A 0%,#8C4A5E 55%,#B76E79 100%);
  border-radius: 20px; padding: 28px 22px; color: #fff; text-align: center;
  margin-bottom: 18px; box-shadow: 0 8px 24px rgba(74,44,58,.25); }
.hero .brand { font-family: 'Playfair Display', serif; font-size: 2.2rem;
  letter-spacing: .35em; margin: 0; padding-left: .35em; }
.hero .sub { font-size: .78rem; letter-spacing: .28em; text-transform: uppercase;
  opacity: .85; margin-top: 6px; }
.card { background: #fff; border-radius: 16px; padding: 12px;
  box-shadow: 0 2px 14px rgba(74,44,58,.10); margin: 10px 0; }
.card img { width: 100%; border-radius: 12px; display: block; }
.cap { color: #8a7a7f; font-size: .82rem; margin-top: 8px; text-align: center; }
.chips { display: flex; gap: 8px; justify-content: center; margin-top: 8px; flex-wrap: wrap; }
.chip { font-size: .78rem; padding: 4px 10px; border-radius: 999px;
  background: #F3E9E4; color: #4A2C3A; }
.dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; }
.metrics { display: flex; gap: 12px; margin: 12px 0; }
.metric { flex: 1; background: #fff; border-radius: 16px; padding: 14px; text-align: center;
  box-shadow: 0 2px 14px rgba(74,44,58,.10); border-top: 4px solid #B76E79; }
.metric .v { font-family: 'Playfair Display', serif; font-size: 1.9rem; color: #4A2C3A; }
.metric .l { font-size: .72rem; color: #8a7a7f; text-transform: uppercase; letter-spacing: .08em; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { border-radius: 999px; padding: 8px 16px; background: #F3E9E4; }
.stTabs [aria-selected="true"] { background: #B76E79; }
.stTabs [aria-selected="true"] p { color: #fff; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
.stButton > button, .stDownloadButton > button { border-radius: 12px; background: #B76E79;
  color: #fff; border: 0; padding: .55rem 1rem; width: 100%; }
.stDownloadButton > button:hover, .stButton > button:hover { background: #8C4A5E; color: #fff; }
.guide { background: #F3E9E4; border-radius: 14px; padding: 14px 16px; color: #4A2C3A;
  font-size: .9rem; line-height: 1.6; }
.note { font-size: .78rem; color: #8a7a7f; text-align: center; margin-top: 22px; }
</style>
<div class="hero"><div class="brand">BRAND V</div>
<div class="sub">Face Contour Analysis</div></div>""", unsafe_allow_html=True)

SHARE_HTML = """
<button id="b" style="width:100%;padding:12px;border-radius:12px;border:1px solid #B76E79;
background:#fff;color:#8C4A5E;font-size:15px;font-family:sans-serif">Share / Save to Photos</button>
<div id="m" style="font:13px sans-serif;color:#a33;margin-top:6px"></div>
<script>
const b64 = "__B64__";
document.getElementById("b").onclick = async () => {
  const m = document.getElementById("m");
  try {
    const bin = atob(b64), arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
    const file = new File([arr], "contour.jpg", {type: "image/jpeg"});
    if (navigator.canShare && navigator.canShare({files: [file]})) {
      await navigator.share({files: [file], title: "Brand V contour"});
    } else {
      m.textContent = "Share not available here - press and hold the image instead.";
    }
  } catch (e) {
    if (e.name !== "AbortError")
      m.textContent = "Could not open the share sheet - press and hold the image instead.";
  }
};
</script>
"""

MAX_SIDE = 1000

with st.expander("Scan settings"):
    lines_n = st.slider("Contour density", 12, 48, 24)
    show_grid = st.checkbox("Show grid", value=False)
    g1, g2 = st.columns(2)
    grid_rows = g1.slider("Grid rows", 4, 20, 10, disabled=not show_grid)
    grid_cols = g2.slider("Grid columns", 2, 16, 8, disabled=not show_grid)
    g3, g4 = st.columns(2)
    grid_hex = g3.color_picker("Grid colour", "#FFFFFF", disabled=not show_grid)
    grid_mid = g4.checkbox("Show midline", value=True, disabled=not show_grid)
    t1, t2 = st.columns(2)
    grid_dx = t1.slider("Grid left / right", -30, 30, 0, disabled=not show_grid)
    grid_dy = t2.slider("Grid up / down", -30, 30, 0, disabled=not show_grid)
    t3, t4 = st.columns(2)
    grid_size = t3.slider("Grid size %", 60, 150, 100, disabled=not show_grid)
    grid_tilt = t4.slider("Grid tilt (degrees)", -25, 25, 0, disabled=not show_grid)


def with_grid(img, pts):
    if not show_grid:
        return img
    r, g, b = (int(grid_hex[i:i + 2], 16) for i in (1, 3, 5))
    return mc.draw_grid(img, pts, grid_rows, grid_cols, color=(b, g, r),
                        midline=grid_mid, dx=grid_dx / 100, dy=grid_dy / 100,
                        scale=grid_size / 100, rot_deg=grid_tilt)


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


def check(name, img, pts):
    if pts is None:
        st.error(f"{name}: no face detected.")
        return False
    for msg in mc.quality_check(img):
        st.warning(f"{name}: {msg}")
    ok, pose = mc.pose_ok(pts)
    if not ok:
        st.warning(f"{name}: head turned/tilted. Retake for accuracy.")
    return True


def labelled_pair(left, right):
    """Join two equal-size images with a divider and BEFORE / AFTER labels."""
    h = left.shape[0]
    out = np.hstack([left, np.full((h, 8, 3), 255, np.uint8), right])
    for text, x in (("BEFORE", 14), ("AFTER", left.shape[1] + 22)):
        for col, th in (((60, 40, 74), 5), ((255, 255, 255), 2)):
            cv2.putText(out, text, (x, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, th,
                        cv2.LINE_AA)
    return out


def jpg(bgr):
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes() if ok else None


GRID_HTML = """<!DOCTYPE html><html><body style="margin:0">
<div id="w" style="font-family:sans-serif">
<canvas id="c" style="width:100%;border-radius:12px;touch-action:none;display:block;user-select:none;-webkit-user-select:none"></canvas>
<div style="display:flex;gap:8px;margin-top:8px">
<button id="reset" style="flex:1;padding:11px;border-radius:12px;border:1px solid #B76E79;background:#fff;color:#8C4A5E;font-size:15px">Reset grid</button>
<button id="save" style="flex:1;padding:11px;border-radius:12px;border:0;background:#B76E79;color:#fff;font-size:15px">Save / Share</button></div>
</div>
<script>
const P = __PARAMS__;
const img = new Image();
img.onload = start;
img.src = "data:image/jpeg;base64,__B64__";
function start() {
  const c = document.getElementById("c"), ctx = c.getContext("2d");
  c.width = img.width; c.height = img.height;
  const g0 = Object.assign({}, P.g); let g = Object.assign({}, g0);
  const E = 1.1, PW = P.w + P.gap, HR = P.w / 38, BASE = Math.max(1.5, P.w / 450);
  const rot = (x, y, a) => [x*Math.cos(a) - y*Math.sin(a), x*Math.sin(a) + y*Math.cos(a)];
  const pt = (x, y) => { const r = rot(x, y, g.a); return [g.cx + r[0], g.cy + r[1]]; };
  const dist = (a, b) => Math.hypot(a[0]-b[0], a[1]-b[1]);
  const cornerH = () => pt(g.hw*E, g.hh*E);
  const rotH = () => pt(0, -g.hh*E - HR*2.2);
  function line(ox, x1, y1, x2, y2, lw) {
    const A = pt(x1, y1), B = pt(x2, y2);
    ctx.lineWidth = lw; ctx.beginPath();
    ctx.moveTo(ox + A[0], A[1]); ctx.lineTo(ox + B[0], B[1]); ctx.stroke();
  }
  function grid(ox) {
    ctx.strokeStyle = P.color; ctx.globalAlpha = 0.75;
    for (let k = 0; k <= P.rows; k++) { const t = -g.hh + 2*g.hh*k/P.rows; line(ox, -g.hw*E, t, g.hw*E, t, BASE); }
    for (let k = 0; k <= P.cols; k++) {
      const s = -g.hw + 2*g.hw*k/P.cols;
      line(ox, s, -g.hh*E, s, g.hh*E, (P.midline && P.cols % 2 == 0 && k == P.cols/2) ? BASE*2 : BASE);
    }
    if (P.midline && P.cols % 2 == 1) line(ox, 0, -g.hh*E, 0, g.hh*E, BASE*2);
    ctx.globalAlpha = 1;
  }
  function handles(ox) {
    const top = pt(0, -g.hh*E), rh = rotH(), ch = cornerH();
    ctx.strokeStyle = "#B76E79"; ctx.lineWidth = BASE;
    ctx.beginPath(); ctx.moveTo(ox + top[0], top[1]); ctx.lineTo(ox + rh[0], rh[1]); ctx.stroke();
    [rh, ch].forEach(h => {
      ctx.beginPath(); ctx.arc(ox + h[0], h[1], HR, 0, 7);
      ctx.fillStyle = "#fff"; ctx.fill(); ctx.lineWidth = BASE*1.6; ctx.stroke();
    });
  }
  function labels() {
    ctx.font = "bold " + Math.round(P.w/28) + "px sans-serif"; ctx.lineJoin = "round";
    [["BEFORE", 14], ["AFTER", PW + 14]].forEach(t => {
      ctx.strokeStyle = "#3C284A"; ctx.lineWidth = 5; ctx.strokeText(t[0], t[1], P.w/16);
      ctx.fillStyle = "#fff"; ctx.fillText(t[0], t[1], P.w/16);
    });
  }
  function render(withHandles) {
    ctx.drawImage(img, 0, 0);
    [0, PW].forEach(ox => { grid(ox); if (withHandles) handles(ox); });
    labels();
    try {
      const needH = document.getElementById("w").offsetHeight + 16, fe = window.frameElement;
      if (fe && Math.abs(fe.offsetHeight - needH) > 3) fe.style.height = needH + "px";
    } catch (e) {}
  }
  const pos = ev => { const r = c.getBoundingClientRect();
    return [(ev.clientX - r.left) * c.width / r.width, (ev.clientY - r.top) * c.height / r.height]; };
  let mode = null, last = null;
  const local = p => [p[0] - (p[0] >= PW ? PW : 0), p[1]];
  c.onpointerdown = ev => {
    ev.preventDefault();
    c.setPointerCapture(ev.pointerId);
    const p = pos(ev), q = local(p); mode = "move"; last = p;
    if (dist(q, cornerH()) < HR*1.8) mode = "size";
    else if (dist(q, rotH()) < HR*1.8) mode = "rot";
  };
  c.onpointermove = ev => {
    if (!mode) return;
    const p = pos(ev), q = local(p);
    if (mode == "move") { g.cx += p[0] - last[0]; g.cy += p[1] - last[1]; }
    else if (mode == "size") {
      const d = rot(q[0] - g.cx, q[1] - g.cy, -g.a);
      g.hw = Math.max(20, Math.abs(d[0]) / E); g.hh = Math.max(20, Math.abs(d[1]) / E);
    } else g.a = Math.atan2(q[1] - g.cy, q[0] - g.cx) + Math.PI/2;
    last = p; render(true);
  };
  c.onpointerup = c.onpointercancel = () => { mode = null; };
  document.getElementById("reset").onclick = () => { g = Object.assign({}, g0); render(true); };
  document.getElementById("save").onclick = () => {
    render(false);
    c.toBlob(b => {
      const f = new File([b], "brand-v-grid.jpg", {type: "image/jpeg"});
      if (navigator.canShare && navigator.canShare({files: [f]})) navigator.share({files: [f]}).catch(() => {});
      else { const a = document.createElement("a"); a.href = URL.createObjectURL(b); a.download = "brand-v-grid.jpg"; a.click(); }
      render(true);
    }, "image/jpeg", 0.92);
  };
  render(true);
}
</script></body></html>
"""


def interactive_pair(left, right, pts):
    """Side-by-side photos on a canvas whose grid can be dragged (move), resized
    (corner handle) and rotated (top handle). Runs in the browser."""
    h, w = left.shape[:2]
    gap = 8
    pair = np.hstack([left, np.full((h, gap, 3), 255, np.uint8), right])
    data = jpg(pair)
    if data is None:
        st.error("Could not build the result image.")
        return
    r, g, b = (int(grid_hex[i:i + 2], 16) for i in (1, 3, 5))
    params = {"w": int(w), "h": int(h), "gap": gap, "rows": int(grid_rows),
              "cols": int(grid_cols), "color": grid_hex, "midline": bool(grid_mid),
              "g": mc.grid_geometry(pts, grid_dx / 100, grid_dy / 100,
                                  grid_size / 100, grid_tilt)}
    html = (GRID_HTML.replace("__PARAMS__", json.dumps(params))
            .replace("__B64__", base64.b64encode(data).decode()))
    components.html(html, height=int(720 * h / (2 * w + gap)) + 80, scrolling=False)


def card(bgr, caption, legend=None):
    data = jpg(bgr)
    if data is None:
        st.error("Could not build the result image.")
        return None
    b64 = base64.b64encode(data).decode()
    chips = ""
    if legend:
        chips = '<div class="chips">' + "".join(
            f'<span class="chip"><span class="dot" style="background:{c}"></span>{t}</span>'
            for c, t in legend) + "</div>"
    st.markdown(f'<div class="card"><img src="data:image/jpeg;base64,{b64}">'
                f'<div class="cap">{caption}</div>{chips}</div>', unsafe_allow_html=True)
    return data, b64


def save_controls(data, b64, key):
    kwargs = dict(file_name="brand-v-contour.jpg", mime="image/jpeg", key=f"dl_{key}")
    try:
        st.download_button("Save image", data, on_click="ignore", **kwargs)
    except TypeError:
        st.download_button("Save image", data, **kwargs)
    components.html(SHARE_HTML.replace("__B64__", b64), height=90)


def metric_row(items):
    st.markdown('<div class="metrics">' + "".join(
        f'<div class="metric"><div class="v">{v}</div><div class="l">{l}</div></div>'
        for l, v in items) + "</div>", unsafe_allow_html=True)


def guide(text):
    st.markdown(f'<div class="guide">{text}</div>', unsafe_allow_html=True)


TIPS = ("Front-facing, head straight, soft even light, same distance every time. "
        "Remove glasses and pull hair away from the face.")
ACCEPT = ["jpg", "jpeg", "png"]

tab_scan, tab_cmp, tab_sym = st.tabs(["Contour Scan", "Before / After", "Symmetry"])

with tab_scan:
    up = st.file_uploader("Upload a front-facing photo", type=ACCEPT, key="scan")
    img, pts = load(up)
    if img is None:
        guide(TIPS)
    elif check("Photo", img, pts):
        ov, _, _ = mc.render_contour(img, pts, lines_n, color=(255, 255, 255))
        res = card(with_grid(ov, pts), "Face contour lines")
        if res:
            save_controls(*res, key="scan")
        with st.expander("View original photo"):
            card(img, "Original")

with tab_cmp:
    c1, c2 = st.columns(2)
    up_b = c1.file_uploader("Before", type=ACCEPT, key="before")
    up_a = c2.file_uploader("After", type=ACCEPT, key="after")
    b_img, b_pts = load(up_b)
    a_img, a_pts = load(up_a)
    if b_img is None or a_img is None:
        guide("Upload both photos, taken the same way. " + TIPS)
    elif check("Before", b_img, b_pts) and check("After", a_img, a_pts):
        left, right = mc.side_by_side(b_img, b_pts, a_img, a_pts, lines_n)
        if show_grid:
            interactive_pair(left, right, b_pts)
            st.caption("Drag the grid to move it, the round handle on its corner to "
                       "resize, the top handle to rotate. Red lines = before, "
                       "green lines = after.")
        else:
            pair = labelled_pair(left, right)
            res = card(pair, "Before and after, side by side",
                       legend=[("#E53935", "Before lines"), ("#00C800", "After lines")])
            if res:
                save_controls(*res, key="cmp_side")
        comp, rep = mc.compare_contours(b_img, b_pts, a_img, a_pts, lines_n)
        res = card(with_grid(comp, b_pts), "Contour change: red = before, green = after",
                   legend=[("#E53935", "Before"), ("#00C800", "After")])
        metric_row([("Jaw lift (% face height)", f"{rep['jaw_lift_pct_face_height']:.1f}%"),
                    ("Mean depth change", f"{rep['mean_abs_depth_change_pct']:.1f}%")])
        if res:
            save_controls(*res, key="cmp")

with tab_sym:
    up_s = st.file_uploader("Upload a front-facing photo", type=ACCEPT, key="sym")
    s_img, s_pts = load(up_s)
    if s_img is None:
        guide(TIPS)
    elif check("Photo", s_img, s_pts):
        res = card(with_grid(mc.symmetry_map(s_img, s_pts, lines_n), s_pts),
                   "Left side against mirrored right side",
                   legend=[("#E53935", "Left"), ("#00C800", "Mirrored right")])
        if res:
            save_controls(*res, key="sym")

st.markdown('<div class="note">Contours are estimated from the photo for visual '
            'comparison and are not a medical measurement. &nbsp;|&nbsp; build 8</div>', unsafe_allow_html=True)
