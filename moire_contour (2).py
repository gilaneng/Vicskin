"""
moire_contour.py - software "moire contour line" mode for the skin analyzer.

F-RAY (hardware) projects a grating onto the face; the grating interferes with a
reference grating and shows lines of equal depth. This module reproduces the
RESULT in software: estimate a 3D face surface from a photo, then draw
equal-depth contour fringes on it. Same pose + same distance => the fringes of
a before and after photo can be compared (red = before, green = after).

Needs: mediapipe==0.10.14, opencv-python-headless, numpy, scipy
"""
import cv2
import numpy as np
from scipy.interpolate import LinearNDInterpolator

# ---- landmark groups (MediaPipe FaceMesh 468 indices) -----------------------
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
             365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
             132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
# stable anchor points used for alignment (outer eye corners, nose bridge, chin)
ANCHORS = [33, 263, 133, 362, 168, 1, 152, 10]
LEFT_EYE_OUTER, RIGHT_EYE_OUTER = 33, 263
RED_BGR = (0, 0, 255)
GREEN_BGR = (0, 200, 0)   # deeper, more vivid green than pure lime
GREEN_THICK = 1.8         # green (after) lines drawn heavier so they stay visible


def get_landmarks(bgr):
    """Return (N,3) landmarks in pixel units (z scaled like x) or None."""
    import mediapipe as mp  # imported lazily so the math can be tested alone
    h, w = bgr.shape[:2]
    with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5) as fm:
        res = fm.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark
    return np.array([[p.x * w, p.y * h, p.z * w] for p in lm], dtype=np.float64)


def head_pose(pts):
    """Rough yaw/roll/pitch proxies (cephalostat substitute: tell the user to
    re-take when the head is turned). Returns degrees-ish values."""
    l, r = pts[LEFT_EYE_OUTER], pts[RIGHT_EYE_OUTER]
    roll = np.degrees(np.arctan2(r[1] - l[1], r[0] - l[0]))
    yaw = np.degrees(np.arctan2(r[2] - l[2], np.hypot(r[0] - l[0], r[1] - l[1])))
    nose, chin, brow = pts[1], pts[152], pts[10]
    pitch = np.degrees(np.arctan2(nose[2] - 0.5 * (chin[2] + brow[2]),
                                  np.linalg.norm(chin[:2] - brow[:2])))
    return {"yaw": float(yaw), "roll": float(roll), "pitch": float(pitch)}


def pose_ok(pts, tol=6.0):
    p = head_pose(pts)
    return abs(p["yaw"]) < tol and abs(p["roll"]) < tol, p


def similarity_align(src, dst):
    """Least-squares similarity transform (scale+rotation+translation) mapping
    src anchor points onto dst anchor points. Returns 2x3 matrix."""
    s = src[:, :2]
    d = dst[:, :2]
    sm, dm = s.mean(0), d.mean(0)
    s0, d0 = s - sm, d - dm
    u, sig, vt = np.linalg.svd(s0.T @ d0)
    rot = (u @ vt).T
    if np.linalg.det(rot) < 0:
        vt[-1] *= -1
        rot = (u @ vt).T
    scale = sig.sum() / (s0 ** 2).sum()
    a = scale * rot
    t = dm - a @ sm
    return np.hstack([a, t[:, None]]), scale


def depth_map(pts, shape):
    """Dense depth (z) over the image by interpolating landmark z on a
    Delaunay mesh. Returns (depth float32 HxW with NaN outside, mask uint8)."""
    h, w = shape[:2]
    mask = np.zeros((h, w), np.uint8)
    oval = pts[FACE_OVAL, :2].astype(np.int32)
    cv2.fillPoly(mask, [cv2.convexHull(oval)], 255)
    interp = LinearNDInterpolator(pts[:, :2], pts[:, 2], fill_value=np.nan)
    ys, xs = np.mgrid[0:h, 0:w]
    z = interp(xs, ys).astype(np.float32)
    # smooth lightly so fringes are clean lines, not mesh facets
    zf = np.where(np.isnan(z), np.nanmean(z), z)
    zf = cv2.GaussianBlur(zf, (0, 0), max(h, w) / 220.0)
    zf[mask == 0] = np.nan
    return zf, mask


def fringe_lines(depth, pitch_px, thickness=1.0):
    """Equal-depth contour lines. pitch_px = depth step between lines, in the
    same units as depth (pixel-scaled z). Returns float mask 0..1."""
    phase = np.nan_to_num(depth, nan=0.0) / float(pitch_px)
    frac = np.abs(phase - np.round(phase))          # 0 on a contour line
    width = 0.04 * thickness
    line = np.clip(1.0 - frac / width, 0, 1)
    line[np.isnan(depth)] = 0
    return line.astype(np.float32)


def render_contour(bgr, pts, lines_across_face=24, color=(0, 0, 255), alpha=0.85):
    """Photo + equal-depth moire-style lines. Returns (overlay, depth, lines)."""
    depth, mask = depth_map(pts, bgr.shape)
    span = np.nanmax(depth) - np.nanmin(depth)
    lines = fringe_lines(depth, max(span / lines_across_face, 1e-6))
    out = bgr.astype(np.float32).copy()
    col = np.array(color, np.float32)
    a = (lines * alpha)[..., None]
    out = out * (1 - a) + col * a
    return out.astype(np.uint8), depth, lines


def compare_contours(before_bgr, before_pts, after_bgr, after_pts,
                     lines_across_face=24):
    """Warp 'after' onto 'before' using anchors, then draw before fringes in
    RED and after fringes in GREEN on the before photo. Both depth maps use the
    SAME absolute depth step and are zeroed at the nose bridge, so a real
    change in volume shifts the lines instead of being normalised away.
    Returns (composite_bgr, report_dict)."""
    m, scale = similarity_align(after_pts[ANCHORS], before_pts[ANCHORS])
    h, w = before_bgr.shape[:2]
    # transform after landmarks into before's frame (z scaled by same factor)
    xy = (after_pts[:, :2] @ m[:, :2].T) + m[:, 2]
    a_pts = np.column_stack([xy, after_pts[:, 2] * scale])

    ref = 168  # nose bridge: depth reference
    b_pts = before_pts.copy()
    b_pts[:, 2] -= b_pts[ref, 2]
    a_pts[:, 2] -= a_pts[ref, 2]

    d_b, m_b = depth_map(b_pts, before_bgr.shape)
    d_a, m_a = depth_map(a_pts, before_bgr.shape)
    span = np.nanmax(d_b) - np.nanmin(d_b)
    step = max(span / lines_across_face, 1e-6)
    l_b = fringe_lines(d_b, step)
    l_a = fringe_lines(d_a, step, GREEN_THICK)

    comp = before_bgr.astype(np.float32) * 0.75
    for lines, col, alpha in ((l_b, RED_BGR, 0.9), (l_a, GREEN_BGR, 1.0)):
        a = (lines * alpha)[..., None]
        comp = comp * (1 - a) + np.array(col, np.float32) * a
    comp = comp.astype(np.uint8)

    # face-contour (jawline) shift in pixels, plus depth change
    jaw = [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361]
    jaw_shift = a_pts[jaw, :2] - before_pts[jaw, :2]
    face_h = np.linalg.norm(before_pts[10, :2] - before_pts[152, :2])
    both = (~np.isnan(d_b)) & (~np.isnan(d_a))
    report = {
        "jaw_lift_px": float(-jaw_shift[:, 1].mean()),       # + = jaw moved up
        "jaw_lift_pct_face_height": float(-jaw_shift[:, 1].mean() / face_h * 100),
        "mean_abs_depth_change_pct": float(
            np.mean(np.abs(d_a[both] - d_b[both])) / span * 100),
        "alignment_scale": float(scale),
    }
    return comp, report


def symmetry_map(bgr, pts, lines_across_face=24):
    """Mirror the fringes about the face midline; returns overlay with left
    fringes in red and mirrored right fringes in green (asymmetry check)."""
    depth, _ = depth_map(pts, bgr.shape)
    span = np.nanmax(depth) - np.nanmin(depth)
    lines = fringe_lines(depth, span / lines_across_face)
    mid_x = 0.5 * (pts[LEFT_EYE_OUTER, 0] + pts[RIGHT_EYE_OUTER, 0])
    h, w = lines.shape
    # mirror about the vertical line x = mid_x
    m = np.float32([[-1, 0, 2 * mid_x], [0, 1, 0]])
    mirrored = cv2.warpAffine(lines, m, (w, h))
    comp = bgr.astype(np.float32) * 0.75
    mirrored = np.clip(mirrored * GREEN_THICK, 0, 1)
    for ln, col, alpha in ((lines, RED_BGR, 0.9), (mirrored, GREEN_BGR, 1.0)):
        a = (ln * alpha)[..., None]
        comp = comp * (1 - a) + np.array(col, np.float32) * a
    return comp.astype(np.uint8)


def quality_check(bgr):
    """Blackout-box substitute: warn about blur / uneven or poor lighting."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(g, cv2.CV_64F).var()
    mean = g.mean()
    left, right = g[:, : g.shape[1] // 2].mean(), g[:, g.shape[1] // 2:].mean()
    issues = []
    if blur < 60:
        issues.append("Image is blurry - hold still / tap to focus.")
    if mean < 70 or mean > 200:
        issues.append("Lighting too dark or too bright.")
    if abs(left - right) / max(mean, 1) > 0.15:
        issues.append("Lighting is uneven left vs right.")
    return issues


def draw_grid(bgr, pts, rows=10, cols=8, color=(255, 255, 255), alpha=0.55,
              extend=0.10, midline=True):
    """Measurement grid locked to the face: runs forehead (landmark 10) to chin
    (152) and left to right face edge (234 / 454), and follows head tilt.
    The centre vertical line is drawn thicker as the midline. Pass the BEFORE
    landmarks when comparing, so both contour sets are read against one fixed
    grid."""
    top, chin = pts[10, :2], pts[152, :2]
    centre = (top + chin) / 2
    axis = chin - top
    half_h = np.linalg.norm(axis) / 2
    ev = axis / np.linalg.norm(axis)              # down the face
    eu = np.array([ev[1], -ev[0]])                # across the face (to the right)
    half_w = np.linalg.norm(pts[454, :2] - pts[234, :2]) / 2
    ew, eh = half_w * (1 + extend), half_h * (1 + extend)

    layer = bgr.copy()
    col = tuple(int(c) for c in color)

    def seg(a, b, thick=1):
        cv2.line(layer, tuple(np.int32(np.round(a))), tuple(np.int32(np.round(b))),
                 col, thick, cv2.LINE_AA)

    for k in range(rows + 1):                     # horizontal lines
        t = -half_h + 2 * half_h * k / rows
        seg(centre + eu * -ew + ev * t, centre + eu * ew + ev * t)
    for k in range(cols + 1):                     # vertical lines
        s = -half_w + 2 * half_w * k / cols
        thick = 2 if (midline and cols % 2 == 0 and k == cols // 2) else 1
        seg(centre + eu * s + ev * -eh, centre + eu * s + ev * eh, thick)
    if midline and cols % 2 == 1:                 # odd: add an explicit midline
        seg(centre + ev * -eh, centre + ev * eh, 2)
    return cv2.addWeighted(layer, alpha, bgr, 1 - alpha, 0)


def side_by_side(before_bgr, before_pts, after_bgr, after_pts, lines_across_face=24):
    """Before and after photos for a side-by-side view. The after photo is
    aligned into the before photo's frame (same anchors as compare_contours),
    so features line up across the two panels and one grid fits both.
    Before gets RED contour lines, after gets GREEN, drawn with the same
    absolute depth step so a real change shifts the lines. Returns (left, right)."""
    h, w = before_bgr.shape[:2]
    m, scale = similarity_align(after_pts[ANCHORS], before_pts[ANCHORS])
    after_al = cv2.warpAffine(after_bgr, m, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    xy = (after_pts[:, :2] @ m[:, :2].T) + m[:, 2]
    a_pts = np.column_stack([xy, after_pts[:, 2] * scale])
    b_pts = before_pts.copy()
    ref = 168
    b_pts[:, 2] -= b_pts[ref, 2]
    a_pts[:, 2] -= a_pts[ref, 2]

    d_b, _ = depth_map(b_pts, before_bgr.shape)
    d_a, _ = depth_map(a_pts, before_bgr.shape)
    step = max((np.nanmax(d_b) - np.nanmin(d_b)) / lines_across_face, 1e-6)

    def paint(img, lines, col, alpha=0.9):
        a = (lines * alpha)[..., None]
        out = img.astype(np.float32) * (1 - a) + np.array(col, np.float32) * a
        return out.astype(np.uint8)

    left = paint(before_bgr, fringe_lines(d_b, step), RED_BGR)
    right = paint(after_al, fringe_lines(d_a, step, GREEN_THICK), GREEN_BGR, 1.0)
    return left, right


def grid_geometry(pts):
    """Starting position of the face grid (same maths as draw_grid) as numbers,
    so the browser can draw it and let the user drag, resize and rotate it."""
    top, chin = pts[10, :2], pts[152, :2]
    axis = chin - top
    ev = axis / np.linalg.norm(axis)
    eu = np.array([ev[1], -ev[0]])
    return {"cx": float((top[0] + chin[0]) / 2), "cy": float((top[1] + chin[1]) / 2),
            "hw": float(np.linalg.norm(pts[454, :2] - pts[234, :2]) / 2),
            "hh": float(np.linalg.norm(axis) / 2),
            "a": float(np.arctan2(eu[1], eu[0]))}
