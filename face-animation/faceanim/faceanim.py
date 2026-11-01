#!/usr/bin/env python3
"""Face performance -> cartoon character animation.

Stages:
  track   : video -> per-frame blendshapes, landmarks, head pose (npz)
  render  : npz + video -> character video (modes: character, overlay, side)
  sweep   : render a contact sheet of synthetic expressions (visual test)
"""
import argparse
import json
import math
import os
import subprocess
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(os.path.dirname(HERE), "face_landmarker.task")


# --------------------------------------------------------------------------
# Video helpers
# --------------------------------------------------------------------------
def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path],
        capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def normalize_video(src, dst, fps=30, max_h=1280):
    """Re-encode to constant fps, upright (rotation applied), capped height, keep audio."""
    vf = f"fps={fps},scale=-2:'min({max_h},ih)'"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-vf", vf,
                    "-c:v", "libx264", "-crf", "16", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k", dst], check=True)


# --------------------------------------------------------------------------
# Tracking
# --------------------------------------------------------------------------
def track(video, out_npz, model=DEFAULT_MODEL):
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    import mediapipe as mp

    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model),
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.4,
        min_face_presence_confidence=0.4,
        min_tracking_confidence=0.4)
    lm = vision.FaceLandmarker.create_from_options(opts)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    names = None
    found, bss, lms, mats = [], [], [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = lm.detect_for_video(img, int(round(i * 1000.0 / fps)))
        if res.face_landmarks:
            if names is None:
                names = [c.category_name for c in res.face_blendshapes[0]]
            bss.append([c.score for c in res.face_blendshapes[0]])
            lms.append([[p.x * W, p.y * H, p.z * W] for p in res.face_landmarks[0]])
            mats.append(np.array(res.facial_transformation_matrixes[0]).reshape(4, 4)
                        if res.facial_transformation_matrixes else np.eye(4))
            found.append(True)
        else:
            bss.append(None); lms.append(None); mats.append(None); found.append(False)
        i += 1
        if i % 150 == 0:
            print(f"  tracked {i} frames", flush=True)
    cap.release()
    n = len(found)
    if names is None:
        raise SystemExit("No face found in the video.")
    nb, nl = len(names), 478
    B = np.zeros((n, nb), np.float32); L = np.zeros((n, nl, 3), np.float32); M = np.zeros((n, 4, 4), np.float32)
    for k in range(n):
        if found[k]:
            B[k] = bss[k]; L[k] = np.array(lms[k])[:nl]; M[k] = mats[k]
    np.savez_compressed(out_npz, blend=B, lms=L, mats=M, found=np.array(found), names=np.array(names),
                        fps=fps, W=W, H=H)
    print(f"tracked {n} frames, face found in {int(np.sum(found))} ({100*np.mean(found):.1f}%)")
    return out_npz


# --------------------------------------------------------------------------
# Signal processing: fill gaps, calibrate, smooth
# --------------------------------------------------------------------------
def fill_gaps(arr, found):
    """Linear interpolation over frames with no face (any trailing shape)."""
    arr = arr.copy()
    idx = np.where(found)[0]
    if len(idx) == 0:
        return arr
    flat = arr.reshape(len(arr), -1)
    allx = np.arange(len(arr))
    for j in range(flat.shape[1]):
        flat[:, j] = np.interp(allx, idx, flat[idx, j])
    return flat.reshape(arr.shape)


def fill_gaps_hold(arr, found, long_gap=10, fade=8):
    """Short gaps: linear. Long gaps: hold the value before, cross-fade at the middle to the value after."""
    out = fill_gaps(arr, found)
    n = len(found); i = 0
    while i < n:
        if not found[i]:
            j = i
            while j < n and not found[j]:
                j += 1
            if j - i >= long_gap:
                before = out[i - 1] if i > 0 else out[j] if j < n else out[i]
                after = out[j] if j < n else before
                mid = (i + j) / 2.0
                for k in range(i, j):
                    t = np.clip((k - (mid - fade / 2)) / fade, 0, 1)
                    t = t * t * (3 - 2 * t)
                    out[k] = (1 - t) * before + t * after
            i = j
        else:
            i += 1
    return out


def rolling_median(x, win):
    r = win // 2
    pad = np.pad(x, (r, r), mode="edge")
    return np.array([np.median(pad[k:k + win]) for k in range(len(x))], np.float32)


def clean_found(found, edge=3, min_run=12):
    """Distrust frames next to tracking dropouts and short isolated runs (e.g. hands over face)."""
    f = found.copy(); n = len(f)
    for i in np.where(~found)[0]:
        f[max(0, i - edge):min(n, i + edge + 1)] = False
    i = 0
    while i < n:
        if f[i]:
            j = i
            while j < n and f[j]:
                j += 1
            if j - i < min_run and (i > 0 or j < n):
                f[i:j] = False
            i = j
        else:
            i += 1
    return f if f.any() else found


def gsmooth(x, sigma):
    """Zero-phase gaussian smoothing along axis 0."""
    if sigma <= 0:
        return x
    r = int(math.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    pad = np.pad(x, [(r, r)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    out = np.zeros_like(x, dtype=np.float32)
    for i, w in enumerate(k):
        out += w * pad[i:i + len(x)]
    return out


def lr_mapping(B, L, names, found):
    """Decide whether MediaPipe '...Left' blendshapes are screen-left. Returns True if so."""
    try:
        bl = B[found, names.index("eyeBlinkLeft")]
        br = B[found, names.index("eyeBlinkRight")]
    except ValueError:
        return True
    P = L[found]
    def ear(up, lo, c1, c2):
        return np.linalg.norm(P[:, up, :2] - P[:, lo, :2], axis=1) / (np.linalg.norm(P[:, c1, :2] - P[:, c2, :2], axis=1) + 1e-6)
    ear_screen_left = ear(159, 145, 33, 133)   # subject's right eye appears on screen left
    if np.std(ear_screen_left) < 1e-4 or np.std(bl - br) < 1e-3:
        return False  # MediaPipe default: '...Left' is the subject's left = screen right
    d = ear_screen_left - ear(386, 374, 362, 263)
    c = np.corrcoef(d, bl - br)[0, 1]
    # if screen-left eye closing (d down) goes with eyeBlinkLeft up, Left == screen-left
    return bool(c < 0)


GAINS = {
    "jawOpen": 1.9, "mouthClose": 1.5, "mouthSmile": 1.6, "mouthFrown": 2.2, "mouthStretch": 2.0,
    "mouthPucker": 1.5, "mouthFunnel": 1.7, "mouthUpperUp": 1.6, "mouthLowerDown": 1.6,
    "mouthLeft": 2.0, "mouthRight": 2.0, "mouthDimple": 1.5, "mouthPress": 1.4,
    "browInnerUp": 1.7, "browDown": 1.8, "browOuterUp": 1.8,
    "eyeWide": 2.0, "eyeSquint": 1.3, "cheekSquint": 1.5, "cheekPuff": 2.2, "noseSneer": 2.0,
}


def gain_for(name):
    for k, g in GAINS.items():
        if name.startswith(k):
            return g
    return 1.0


def build_expressions(data):
    """Return per-frame dicts of screen-space expression values + head pose."""
    names = [str(s) for s in data["names"]]
    found = clean_found(data["found"].astype(bool))
    B = fill_gaps_hold(data["blend"], found)
    L = fill_gaps(data["lms"], found)
    Lh = fill_gaps_hold(data["lms"], found)
    n = len(B)
    left_is_screen_left = lr_mapping(B, L, names, found)
    idx = {nm: i for i, nm in enumerate(names)}

    # calibrate each channel
    C = np.zeros_like(B)
    for nm, i in idx.items():
        v = B[:, i]
        fv = v[found] if found.any() else v
        if nm.startswith("eyeBlink"):
            base = np.percentile(fv, 50)
            top = max(np.percentile(fv, 98.5) - base, 0.22)
            x = np.clip((v - base) / top, 0, 1)
            x = np.clip((x - 0.12) / 0.7, 0, 1)
            C[:, i] = x * x * (3 - 2 * x)
        elif nm.startswith("eyeLook"):
            C[:, i] = v - np.median(fv)
        else:
            base = np.percentile(fv, 8)
            C[:, i] = np.clip((v - base) / max(1 - base, 1e-3) * gain_for(nm), 0, 1)

    # smoothing (blinks fast, the rest a bit softer)
    for nm, i in idx.items():
        sig = 0.6 if nm.startswith("eyeBlink") else (1.0 if nm.startswith(("jaw", "mouth")) else 1.4)
        C[:, i] = gsmooth(C[:, i], sig)

    def g(nm):
        return C[:, idx[nm]] if nm in idx else np.zeros(n, np.float32)

    def side(base):
        a, b = g(base + "Left"), g(base + "Right")
        return (a, b) if left_is_screen_left else (b, a)  # (screen-left, screen-right)

    E = {}
    for base, key in [("eyeBlink", "blink"), ("eyeWide", "wide"), ("eyeSquint", "squint"),
                      ("browDown", "browDown"), ("browOuterUp", "browOuter"), ("cheekSquint", "cheekSq"),
                      ("mouthSmile", "smile"), ("mouthFrown", "frown"), ("mouthStretch", "stretch"),
                      ("mouthDimple", "dimple"), ("mouthUpperUp", "upperUp"), ("mouthLowerDown", "lowerDown"),
                      ("mouthPress", "press"), ("noseSneer", "sneer")]:
        l, r = side(base)
        E[key + "L"], E[key + "R"] = l, r
    E["browInner"] = g("browInnerUp")
    E["jaw"] = g("jawOpen"); E["close"] = g("mouthClose")
    E["pucker"] = g("mouthPucker"); E["funnel"] = g("mouthFunnel")
    E["puff"] = g("cheekPuff")
    ml, mr = g("mouthLeft"), g("mouthRight")
    # mouthLeft = subject's left; on screen that's right unless mapping says otherwise
    E["mouthShift"] = (ml - mr) if not left_is_screen_left else (mr - ml)
    # gaze, screen space
    li, lo = side("eyeLookIn"); oi, oo = side("eyeLookOut")  # (screenL,screenR) per channel
    # screen-left eye: looking "in" = toward +x ; screen-right eye: looking "out" = toward +x
    gx = 0.5 * ((li - oi) + (oo - lo))
    up = 0.5 * (g("eyeLookUpLeft") + g("eyeLookUpRight"))
    dn = 0.5 * (g("eyeLookDownLeft") + g("eyeLookDownRight"))
    E["gazeX"] = np.clip(gsmooth(gx, 1.2) * 1.6, -1, 1)
    E["gazeY"] = np.clip(gsmooth(dn - up, 1.2) * 1.6, -1, 1)

    # head pose from landmarks (image space)
    P = L[:, :, :2]
    le, re = P[:, 33], P[:, 263]
    roll = np.degrees(np.arctan2(re[:, 1] - le[:, 1], re[:, 0] - le[:, 0]))
    roll = np.unwrap(np.radians(roll)); roll = np.degrees(roll)
    fl, fr = P[:, 234], P[:, 454]
    top, chin = P[:, 10], P[:, 152]
    nose = P[:, 1]
    eye_mid = 0.5 * (le + re)
    mouth_mid = 0.5 * (P[:, 61] + P[:, 291])
    face_h = np.linalg.norm(chin - top, axis=1)
    # rotate into roll-free frame to measure yaw/pitch
    th = -np.radians(roll)
    def unroll(p, c):
        d = p - c
        return np.stack([d[:, 0] * np.cos(th) - d[:, 1] * np.sin(th),
                         d[:, 0] * np.sin(th) + d[:, 1] * np.cos(th)], 1)
    c0 = 0.5 * (fl + fr)
    nl = unroll(nose, c0); fll = unroll(fl, c0); frl = unroll(fr, c0)
    half_w = 0.5 * (frl[:, 0] - fll[:, 0]) + 1e-6
    yaw = np.clip(nl[:, 0] / half_w, -1, 1)
    eml = unroll(eye_mid, c0); mml = unroll(mouth_mid, c0)
    t = (nl[:, 1] - eml[:, 1]) / (mml[:, 1] - eml[:, 1] + 1e-6)
    pitch = np.clip((t - np.median(t[found] if found.any() else t)) * 3.0, -1, 1)
    yaw = yaw - np.median(yaw[found] if found.any() else yaw) * 0.5
    anchor = np.stack([0.5 * (fl[:, 0] + fr[:, 0]), eye_mid[:, 1]], 1)  # face centre at eye level
    eye_chin = np.linalg.norm(chin - eye_mid, axis=1)

    pose = {
        "roll": gsmooth(roll, 1.6), "yaw": gsmooth(yaw, 1.6), "pitch": gsmooth(pitch, 1.6),
        "anchor": gsmooth(anchor, 1.4), "face_h": gsmooth(face_h, 2.0), "eye_chin": gsmooth(eye_chin, 2.0),
    }
    # ---- geometry-based signals: more sensitive than blendshapes for eyes, brows and lip gap
    P = Lh[:, :, :2]
    fh = np.linalg.norm(P[:, 152] - P[:, 10], axis=1) + 1e-6
    def ear(up, lo, c1, c2):
        return np.linalg.norm(P[:, up] - P[:, lo], axis=1) / (np.linalg.norm(P[:, c1] - P[:, c2], axis=1) + 1e-6)
    def pct(v, q):
        return float(np.percentile(v[found] if found.any() else v, q))
    for s_, (up_, lo_, c1, c2) in (("L", (159, 145, 33, 133)), ("R", (386, 374, 362, 263))):
        o = gsmooth(ear(up_, lo_, c1, c2), 0.7)
        rest = pct(o, 50); closed = min(pct(o, 1), rest * 0.35); wide_top = max(pct(o, 99.5), rest * 1.15)
        b = np.clip((rest - o) / max(rest - closed, 1e-3), 0, 1)
        b = np.clip((b - 0.08) / 0.8, 0, 1); b = b * b * (3 - 2 * b)
        w = np.clip((o - rest) / max(wide_top - rest, 1e-3), 0, 1)
        w = np.clip((w - 0.15) / 0.75, 0, 1)
        E["blink" + s_] = b
        E["wide" + s_] = np.maximum(E["wide" + s_], w)
    raise_ = {}
    for s_, (bi, bo, c1, c2) in (("L", (107, 70, 33, 133)), ("R", (336, 300, 362, 263))):
        ec = 0.5 * (P[:, c1] + P[:, c2])
        hi = gsmooth(np.linalg.norm(P[:, bi] - ec, axis=1) / fh, 1.2)
        ho = gsmooth(np.linalg.norm(P[:, bo] - ec, axis=1) / fh, 1.2)
        win = int(round(float(data["fps"]) * 5)) | 1
        hi = hi - rolling_median(hi, win) + np.median(hi)
        ho = ho - rolling_median(ho, win) + np.median(ho)
        def updown(h):
            r = pct(h, 50); u = max(pct(h, 99) - r, 0.006); d = max(r - pct(h, 1), 0.006)
            return np.clip((h - r) / u, 0, 1), np.clip((r - h) / d, 0, 1)
        iu, idn = updown(hi); ou, _ = updown(ho)
        raise_[s_] = (iu, ou, idn)
    wide_avg = 0.5 * (E["wideL"] + E["wideR"])
    E["browInner"] = np.maximum.reduce([E["browInner"], 0.45 * (raise_["L"][0] + raise_["R"][0]), 0.6 * wide_avg])
    for s_ in "LR":
        E["browOuter" + s_] = np.maximum.reduce([E["browOuter" + s_], 0.9 * raise_[s_][1], 0.7 * E["wide" + s_]])
        E["browDown" + s_] = np.maximum(E["browDown" + s_], 0.8 * raise_[s_][2])
    gap = gsmooth(np.linalg.norm(P[:, 13] - P[:, 14], axis=1) / fh, 0.8)
    E["lipGap"] = np.clip((gap - pct(gap, 5) - 0.004) / 0.085, 0, 1)
    pose["chin"] = gsmooth(chin, 1.4)
    pose["face_box"] = gsmooth(np.stack([fl[:, 0], top[:, 1], fr[:, 0], chin[:, 1]], 1), 1.4)

    E = {k: np.asarray(v, np.float32) for k, v in E.items()}
    return E, pose, found, left_is_screen_left


# --------------------------------------------------------------------------
# Drawing canvas (premultiplied RGBA, float32)
# --------------------------------------------------------------------------
SHIFT = 4
SC = 1 << SHIFT


def hexrgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255.0


class Canvas:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.rgb = np.zeros((h, w, 3), np.float32)
        self.a = np.zeros((h, w), np.float32)

    def _bbox(self, pts_list, pad):
        allp = np.concatenate([np.asarray(p, np.float32).reshape(-1, 2) for p in pts_list])
        x0 = int(max(0, math.floor(allp[:, 0].min() - pad))); y0 = int(max(0, math.floor(allp[:, 1].min() - pad)))
        x1 = int(min(self.w, math.ceil(allp[:, 0].max() + pad))); y1 = int(min(self.h, math.ceil(allp[:, 1].max() + pad)))
        return x0, y0, x1, y1

    def _blend(self, box, m, color, opacity, clip):
        x0, y0, x1, y1 = box
        if x1 <= x0 or y1 <= y0:
            return
        a = m.astype(np.float32) * (opacity / 255.0)
        if clip is not None:
            a *= clip[y0:y1, x0:x1]
        sl = (slice(y0, y1), slice(x0, x1))
        a3 = a[..., None]
        self.rgb[sl] = self.rgb[sl] * (1 - a3) + color * a3
        self.a[sl] = self.a[sl] * (1 - a) + a

    @staticmethod
    def _ipts(p, x0, y0):
        return (np.round((np.asarray(p, np.float32).reshape(-1, 2) - [x0, y0]) * SC)).astype(np.int32)

    def fill(self, polys, color, opacity=1.0, clip=None):
        if isinstance(polys, np.ndarray):
            polys = [polys]
        polys = [p for p in polys if p is not None and len(p) >= 3]
        if not polys:
            return
        box = self._bbox(polys, 2)
        x0, y0, x1, y1 = box
        if x1 <= x0 or y1 <= y0:
            return
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(m, [self._ipts(p, x0, y0) for p in polys], 255, cv2.LINE_AA, SHIFT)
        self._blend(box, m, hexrgb(color) if isinstance(color, str) else color, opacity, clip)

    def stroke(self, pts, color, thick, closed=False, opacity=1.0, clip=None):
        pts = np.asarray(pts, np.float32)
        if len(pts) < 2:
            return
        box = self._bbox([pts], thick + 2)
        x0, y0, x1, y1 = box
        if x1 <= x0 or y1 <= y0:
            return
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.polylines(m, [self._ipts(pts, x0, y0)], closed, 255, max(1, int(round(thick))), cv2.LINE_AA, SHIFT)
        self._blend(box, m, hexrgb(color) if isinstance(color, str) else color, opacity, clip)

    def mask(self, polys):
        if isinstance(polys, np.ndarray):
            polys = [polys]
        m = np.zeros((self.h, self.w), np.uint8)
        cv2.fillPoly(m, [self._ipts(p, 0, 0) for p in polys if p is not None and len(p) >= 3], 255, cv2.LINE_AA, SHIFT)
        return m.astype(np.float32) / 255.0

    def fill_mask(self, m, color, opacity=1.0):
        a = m * opacity
        self.rgb = self.rgb * (1 - a[..., None]) + hexrgb(color) * a[..., None]
        self.a = self.a * (1 - a) + a


def ellipse_pts(cx, cy, rx, ry, n=64, rot=0.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x, y = rx * np.cos(t), ry * np.sin(t)
    if rot:
        c, s = math.cos(rot), math.sin(rot)
        x, y = x * c - y * s, x * s + y * c
    return np.stack([cx + x, cy + y], 1)


def cubic(p0, p1, p2, p3, n=28):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = (np.asarray(p, np.float32) for p in (p0, p1, p2, p3))
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def cross2(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def tapered(path, w0, w1):
    """Polygon of a stroke along path whose width goes w0 -> w1, with round ends."""
    path = np.asarray(path, np.float32)
    d = np.gradient(path, axis=0)
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-6
    w = np.linspace(w0, w1, len(path))[:, None] / 2
    left, right = path + nrm * w, path - nrm * w
    # round caps
    a0 = math.atan2(nrm[0, 1], nrm[0, 0]); a1 = math.atan2(nrm[-1, 1], nrm[-1, 0])
    cap0 = [path[0] + w[0, 0] * np.array([math.cos(a0 + t), math.sin(a0 + t)]) for t in np.linspace(0, np.pi, 10)]
    cap1 = [path[-1] + w[-1, 0] * np.array([math.cos(a1 + np.pi + t), math.sin(a1 + np.pi + t)]) for t in np.linspace(0, np.pi, 10)]
    return np.concatenate([left, np.array(cap1), right[::-1], np.array(cap0)])


# --------------------------------------------------------------------------
# Character
# --------------------------------------------------------------------------
PALETTES = {
    "buddy": dict(skin="#f4c9a3", skin_dark="#d99f78", hair="#2f2b4f", hair_hi="#4a4577", iris="#7a5230",
                  line="#2a1c19", mouth="#5b1f2e", tongue="#e8707f", blush="#ff8a9b", shirt="#3f7de8",
                  shirt_dark="#2d5fb8", bg1="#ffe8cf", bg2="#f7b993"),
    "mint": dict(skin="#a8e6c1", skin_dark="#78c49c", hair="#1f4d4a", hair_hi="#2f6e69", iris="#1d2b5c",
                 line="#173029", mouth="#3d1630", tongue="#f07b9a", blush="#ff9fb4", shirt="#f2a33a",
                 shirt_dark="#c7811f", bg1="#e6f7ff", bg2="#a9d8f5"),
    "berry": dict(skin="#c9b6ff", skin_dark="#9f88e6", hair="#ff6f91", hair_hi="#ff9ab2", iris="#2b2250",
                  line="#251a40", mouth="#3a1235", tongue="#ff8fa8", blush="#ff7fa8", shirt="#2bc4a8",
                  shirt_dark="#1f9a84", bg1="#fff4d6", bg2="#ffd18a"),
}

OV_SCALE = 1.12      # overlay: character eye-to-chin vs real eye-to-chin
CHAR_SCALE = 0.86    # character-only mode: head a bit smaller than the overlay head
CS = 1024            # character canvas size
CX, CY = 512, 560    # head centre
EYE_Y = CY - 10      # eye line (anchor row)
HEAD_A, HEAD_B = 330, 370


def head_outline(puff=0.0, jaw=0.0, n=160, grow=0.0, dy=0.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    ex = 2 / 2.5
    x = np.sign(c) * np.abs(c) ** ex
    y = np.sign(s) * np.abs(s) ** ex
    lower = y > 0
    a = np.where(lower, HEAD_A + 34 * puff * np.sin(np.clip(y, 0, 1) * np.pi * 0.9), HEAD_A) + grow
    b = np.where(lower, HEAD_B + 24 * jaw, HEAD_B) + grow
    taper = 1 - 0.13 * np.clip(y, 0, 1) ** 2
    return np.stack([CX + a * x * taper, CY + dy + b * y], 1)


def draw_character(e, pose, pal, cv=None):
    """e: dict of scalar expression values for one frame. Returns Canvas (premultiplied RGBA)."""
    if cv is None:
        cv = Canvas(CS, CS)
    P = {k: hexrgb(v) for k, v in pal.items()}
    yaw = float(np.clip(pose.get("yaw", 0.0), -1, 1))
    pitch = float(np.clip(pose.get("pitch", 0.0), -1, 1))
    fx, fy = yaw * 70.0, pitch * 42.0
    E = lambda k: float(e.get(k, 0.0))
    smile = 0.5 * (E("smileL") + E("smileR"))

    # ears (behind head)
    for sgn in (-1, 1):
        ex_ = CX + sgn * (HEAD_A - 8) - yaw * 38
        ear = ellipse_pts(ex_, CY + 20 + fy * 0.3, 58, 78, 48)
        cv.fill(ear, P["skin"]); cv.stroke(ear, P["line"], 8, True)
        cv.fill(ellipse_pts(ex_ + sgn * 6, CY + 24 + fy * 0.3, 28, 44, 32), P["skin_dark"], 0.8)

    # head
    head = head_outline(E("puff"), E("jaw"))
    hm = cv.mask(head)
    cv.fill(head, P["skin"])
    # soft shading crescent (light from upper left)
    shifted = head_outline(E("puff"), E("jaw")) + np.array([-26, -22])
    shade = hm * (1 - cv.mask(shifted))
    cv.fill_mask(shade, pal["skin_dark"], 0.45)

    # blush
    bl_op = 0.22 + 0.45 * max(0.5 * (E("cheekSqL") + E("cheekSqR")), smile)
    for sgn in (-1, 1):
        cv.fill(ellipse_pts(CX + sgn * 205 + fx * 0.8, CY + 125 + fy, 58 + 10 * E("puff"), 32, 40), P["blush"], bl_op)

    # eyes
    for sgn, s in ((-1, "L"), (1, "R")):
        far = max(0.0, yaw * sgn)            # eye on the side the head turns toward foreshortens
        exx = CX + sgn * (140 - 18 * abs(yaw)) + fx
        eyy = EYE_Y + fy
        rx = 64 * (1 + 0.06 * E("wide" + s)) * (1 - 0.28 * far)
        ry = 82 * (1 + 0.20 * E("wide" + s) - 0.10 * E("squint" + s))
        blink = E("blink" + s)
        sq = min(0.55, 0.45 * E("squint" + s) + 0.25 * E("cheekSq" + s))
        k_up = -1 + 2 * 0.97 * blink
        k_lo = 1 - 2 * sq
        xs = np.linspace(-1, 1, 40)
        arc = np.sqrt(np.clip(1 - xs ** 2, 0, 1))
        X = exx + xs * rx
        y_up = eyy + ry * arc * k_up
        y_lo = eyy + ry * arc * k_lo
        open_h = float(np.max(y_lo - y_up))
        if open_h < 7:
            # closed eye: happy arc if smiling, else a soft lid line
            lid = np.stack([X, eyy + ry * 0.22 + ry * 0.20 * arc], 1)
            if smile > 0.35:
                lid = np.stack([X, eyy + 6 - ry * 0.35 * arc], 1)
            cv.stroke(lid, P["line"], 10)
            continue
        poly = np.concatenate([np.stack([X, y_up], 1), np.stack([X, np.maximum(y_lo, y_up)], 1)[::-1]])
        cv.fill(poly, "#ffffff")
        clip = cv.mask(poly)
        gx, gy = E("gazeX"), E("gazeY")
        ix = exx + gx * rx * 0.42 + yaw * rx * 0.18
        iy = eyy + gy * ry * 0.35 + pitch * 6
        ir = 42 * (1 - 0.15 * far)
        cv.fill(ellipse_pts(ix, iy, ir * (1 - 0.3 * far), ir, 48), P["iris"], 1, clip)
        cv.fill(ellipse_pts(ix, iy + ir * 0.35, ir * 0.85 * (1 - 0.3 * far), ir * 0.55, 40), P["line"], 0.18, clip)
        cv.fill(ellipse_pts(ix, iy, ir * 0.48, ir * 0.48, 36), "#140d0c", 1, clip)
        cv.fill(ellipse_pts(ix - ir * 0.36, iy - ir * 0.38, ir * 0.27, ir * 0.27, 24), "#ffffff", 0.95, clip)
        cv.fill(ellipse_pts(ix + ir * 0.30, iy + ir * 0.30, ir * 0.11, ir * 0.11, 16), "#ffffff", 0.85, clip)
        # lid shadow
        cv.fill(np.concatenate([np.stack([X, y_up], 1), np.stack([X, y_up + 14], 1)[::-1]]), P["line"], 0.12, clip)
        cv.stroke(poly, P["line"], 5, True)
        cv.stroke(np.stack([X, y_up], 1)[3:-3], P["line"], 11)
        # outer lash flick
        ox = X[-1] if sgn > 0 else X[0]
        oy = y_up[-1] if sgn > 0 else y_up[0]
        cv.stroke(np.array([[ox, oy], [ox + sgn * 16, oy - 12 - 6 * (1 - blink)]]), P["line"], 8)

    # nose
    nx, ny = CX + fx * 1.35, CY + 98 + fy * 1.25
    cv.fill(ellipse_pts(nx + 6, ny + 6, 26, 14, 32), P["skin_dark"], 0.55)
    cv.stroke(cubic([nx - 20, ny - 4], [nx - 14, ny + 14], [nx + 14, ny + 14], [nx + 20, ny - 4], 20), P["line"], 7)
    sn = 0.5 * (E("sneerL") + E("sneerR"))
    if sn > 0.05:
        for sgn in (-1, 1):
            cv.stroke(np.array([[nx + sgn * 12, ny - 52], [nx + sgn * 22, ny - 38]]), P["line"], 5, opacity=min(1, sn * 1.5))

    # mouth
    draw_mouth(cv, e, fx, fy, P)

    # hair: inside an enlarged head, above a hairline with bangs
    big = head_outline(0, 0, grow=16, dy=-10)
    hx = fx * 0.55
    hairline = np.concatenate([
        cubic([CX - 380, CY - 30], [CX - 330, CY - 160], [CX - 250, CY - 200], [CX - 170 + hx, CY - 225], 20),
        cubic([CX - 170 + hx, CY - 225], [CX - 60 + hx, CY - 175], [CX + 40 + hx, CY - 240], [CX + 120 + hx, CY - 262], 20),
        cubic([CX + 120 + hx, CY - 262], [CX + 230 + hx, CY - 220], [CX + 320, CY - 195], [CX + 380, CY - 30], 20),
    ])
    cap = np.concatenate([hairline, np.array([[CX + 420, 0], [CX - 420, 0]])])
    hair_m = cv.mask(big) * cv.mask(cap)
    cv.stroke(head, P["line"], 9, True)
    # outline of hair = dilate trick
    k = np.ones((11, 11), np.uint8)
    outline = cv2.dilate((hair_m * 255).astype(np.uint8), k).astype(np.float32) / 255.0
    cv.fill_mask(outline, pal["line"], 1.0)
    cv.fill_mask(hair_m, pal["hair"], 1.0)
    # hair highlight
    hl = cubic([CX - 230, CY - 305], [CX - 150, CY - 350], [CX - 40, CY - 355], [CX + 40, CY - 342], 24)
    cv.stroke(hl, P["hair_hi"], 18, opacity=0.9, clip=hair_m)
    # tuft
    tuft = np.concatenate([cubic([CX - 10 + hx * 0.3, CY - 365], [CX - 30, CY - 450], [CX + 40, CY - 480], [CX + 70, CY - 470], 16),
                           cubic([CX + 70, CY - 470], [CX + 30, CY - 440], [CX + 20, CY - 400], [CX + 30 + hx * 0.3, CY - 360], 16)])
    cv.stroke(tuft, P["line"], 10, True)
    cv.fill(tuft, P["hair"])
    # eyebrows
    for sgn, s in ((-1, "L"), (1, "R")):
        exx = CX + sgn * (140 - 18 * abs(yaw)) + fx
        base = EYE_Y + fy - 82 - 46
        down = E("browDown" + s)
        inner = np.array([exx - sgn * 58 - sgn * 10 * down, base - 44 * E("browInner") + 30 * down])
        outer = np.array([exx + sgn * 74, base + 10 - 36 * E("browOuter" + s) + 10 * down - 10 * E("browInner")])
        mid1 = inner + (outer - inner) * 0.33 + np.array([0, -16 - 6 * E("browOuter" + s)])
        mid2 = inner + (outer - inner) * 0.70 + np.array([0, -18 - 4 * E("browOuter" + s)])
        path = cubic(inner, mid1, mid2, outer, 30)
        bp = tapered(path, 30, 15)
        cv.stroke(bp, P["skin"], 9, True)
        cv.fill(bp, P["hair"])

    return cv


def draw_mouth(cv, e, fx, fy, P):
    E = lambda k: float(e.get(k, 0.0))
    smL, smR = E("smileL"), E("smileR")
    sm = 0.5 * (smL + smR)
    st = 0.5 * (E("stretchL") + E("stretchR"))
    pucker, funnel = E("pucker"), E("funnel")
    mx = CX + fx * 1.12 + 38 * E("mouthShift")
    my = CY + 205 + fy
    w = 92 * (1 + 0.32 * sm + 0.25 * st - 0.42 * pucker - 0.30 * funnel)
    cL = np.array([mx - w * (1 + 0.1 * (E("stretchL") - E("stretchR"))),
                   my - 44 * smL + 34 * E("frownL") - 8 * E("dimpleL")])
    cR = np.array([mx + w * (1 + 0.1 * (E("stretchR") - E("stretchL"))),
                   my - 44 * smR + 34 * E("frownR") - 8 * E("dimpleR")])
    up = 0.5 * (E("upperUpL") + E("upperUpR")); lo = 0.5 * (E("lowerDownL") + E("lowerDownR"))
    opn = np.clip(E("jaw") - 0.75 * E("close") + 0.35 * (up + lo) - 0.3 * 0.5 * (E("pressL") + E("pressR")), 0, 1)
    opn = max(float(opn), E("lipGap"))
    h = 175 * opn
    ro = float(np.clip(1.2 * funnel + 0.6 * pucker, 0, 0.9))
    if h < 9:
        if pucker > 0.35 and sm < 0.3:
            lips = ellipse_pts(mx, my + 2, 24 + 10 * (1 - pucker), 16, 32)
            cv.fill(lips, P["mouth"], 0.85); cv.stroke(lips, P["line"], 7, True)
        else:
            mid_y = my + 6 + 10 * sm - 18 * (E("frownL") + E("frownR")) * 0.5
            line = cubic(cL, [cL[0] + w * 0.55, mid_y + 6 * sm], [cR[0] - w * 0.55, mid_y + 6 * sm], cR, 30)
            cv.stroke(line, P["line"], 9)
            for c, sg, sml in ((cL, -1, smL), (cR, 1, smR)):
                if sml > 0.25:  # dimple tick
                    cv.stroke(np.array([c + [sg * 2, -8], c + [sg * 9, 6]]), P["line"], 6, opacity=min(1, (sml - 0.25) * 3))
        return
    yu = my - 4 - 16 * up - 0.08 * h
    yl = my + 0.86 * h + 8 * lo
    upper = cubic(cL, [cL[0] + w * 0.45, yu + 4 * sm], [cR[0] - w * 0.45, yu + 4 * sm], cR, 30)
    lower = cubic(cR, [cR[0] - w * 0.25, yl + 26 * sm], [cL[0] + w * 0.25, yl + 26 * sm], cL, 30)
    poly = np.concatenate([upper, lower])
    if ro > 0.05:
        oval = ellipse_pts(mx, my + 0.42 * h, max(w * 0.5, 26), max(0.55 * h, 20), len(poly))
        # align start points (leftmost) so the blend does not twist
        start = int(np.argmin(oval[:, 0])); oval = np.roll(oval, -start, axis=0)
        if cross2(poly[1] - poly[0], poly[2] - poly[1]) * cross2(oval[1] - oval[0], oval[2] - oval[1]) < 0:
            oval = oval[::-1]; oval = np.roll(oval, -int(np.argmin(oval[:, 0])), axis=0)
        poly = (1 - ro) * poly + ro * oval
    cv.fill(poly, P["mouth"])
    clip = cv.mask(poly)
    top = poly[:30]
    if h > 16 and ro < 0.6:
        tb = min(24.0, 0.32 * h)
        teeth = np.concatenate([top, (top + [0, tb])[::-1]])
        cv.fill(teeth, "#ffffff", 1, clip)
    tongue_y = my + 0.86 * h + 8 * lo
    cv.fill(ellipse_pts(mx, tongue_y - 0.05 * h, w * 0.62, max(0.36 * h, 12), 40), P["tongue"], 1, clip)
    cv.stroke(poly, P["line"], 8, True)


# --------------------------------------------------------------------------
# Compositing
# --------------------------------------------------------------------------
def place(frame, cv, anchor_xy, scale, roll_deg, src_anchor=(CX, EYE_Y)):
    """Warp character canvas onto frame (uint8 BGR) in place."""
    H, W = frame.shape[:2]
    M = cv2.getRotationMatrix2D((float(src_anchor[0]), float(src_anchor[1])), -roll_deg, scale)
    M[0, 2] += anchor_xy[0] - src_anchor[0]
    M[1, 2] += anchor_xy[1] - src_anchor[1]
    rgba = np.dstack([cv.rgb, cv.a])
    warped = cv2.warpAffine(rgba, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    a = warped[..., 3:4]
    fr = frame.astype(np.float32) / 255.0
    out = fr * (1 - a) + warped[..., [2, 1, 0]]  # premultiplied RGB -> BGR
    frame[:] = np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)


def gradient_bg(W, H, c1, c2):
    t = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    img = (1 - t) * hexrgb(c1) + t * hexrgb(c2)
    img = np.repeat(img, W, axis=1)
    return (img[..., ::-1] * 255).astype(np.uint8)


def draw_body(frame, anchor, scale, pal, sway=0.0):
    H, W = frame.shape[:2]
    cvb = Canvas(W, H)
    neck_top = anchor[1] + 300 * scale
    nx = anchor[0]
    neck = np.array([[nx - 70 * scale, neck_top - 60 * scale], [nx + 70 * scale, neck_top - 60 * scale],
                     [nx + 80 * scale, neck_top + 150 * scale], [nx - 80 * scale, neck_top + 150 * scale]])
    sh_y = neck_top + 120 * scale
    bx = nx * 0.7 + W * 0.15
    body = np.concatenate([
        cubic([bx - 120 * scale, sh_y], [bx - 380 * scale, sh_y + 10 * scale], [bx - 470 * scale, sh_y + 120 * scale], [bx - 500 * scale, H + 40], 24),
        np.array([[bx + 500 * scale, H + 40]]),
        cubic([bx + 500 * scale, H + 40], [bx + 470 * scale, sh_y + 120 * scale], [bx + 380 * scale, sh_y + 10 * scale], [bx + 120 * scale, sh_y], 24),
    ])
    cvb.fill(neck, pal["skin"]); cvb.stroke(neck, pal["line"], max(2, 8 * scale), True)
    cvb.fill(ellipse_pts(nx, neck_top - 20 * scale, 72 * scale, 30 * scale, 32), pal["skin_dark"], 0.6)
    cvb.fill(body, pal["shirt"]); cvb.stroke(body, pal["line"], max(2, 9 * scale), True)
    collar = cubic([bx - 120 * scale, sh_y], [bx - 60 * scale, sh_y + 90 * scale], [bx + 60 * scale, sh_y + 90 * scale], [bx + 120 * scale, sh_y], 24)
    cvb.fill(np.concatenate([collar, np.array([[bx + 120 * scale, sh_y]])]), pal["shirt_dark"], 0.9)
    cvb.stroke(collar, pal["line"], max(2, 7 * scale))
    fr = frame.astype(np.float32) / 255.0
    out = fr * (1 - cvb.a[..., None]) + cvb.rgb[..., ::-1]
    frame[:] = np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)


def make_segmenter():
    path = os.path.join(os.path.dirname(HERE), "selfie_multiclass_256x256.tflite")
    if not os.path.exists(path):
        return None
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    opts = vision.ImageSegmenterOptions(base_options=mp_python.BaseOptions(model_asset_path=path),
                                        running_mode=vision.RunningMode.VIDEO,
                                        output_confidence_masks=True, output_category_mask=False)
    return vision.ImageSegmenter.create_from_options(opts)


def hand_mask(seg, frame_bgr, i, fps, p, cut=True):
    """Soft mask of hands/arms (body-skin) in front of the face region, or None."""
    import mediapipe as mp
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = seg.segment_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(round(i * 1000 / fps)))
    skin = np.array(res.confidence_masks[2].numpy_view(), np.float32).reshape(frame_bgr.shape[:2])
    H, W = skin.shape
    x0, y0, x1, y1 = [float(v) for v in p["face_box"]]
    fw, fh = max(x1 - x0, 1), max(y1 - y0, 1)
    bin_ = (skin > 0.5).astype(np.uint8)
    if cut:
        bin_[int(min(H, max(0, p["chin"][1] + 0.12 * fh))):] = 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bin_, connectivity=8)
    ix0, iy0 = int(max(0, x0 + 0.2 * fw)), int(max(0, y0 + 0.15 * fh))
    ix1, iy1 = int(min(W, x1 - 0.2 * fw)), int(min(H, y1 - 0.1 * fh))
    keep = np.zeros((H, W), np.float32)
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < 0.06 * fw * fh:
            continue
        comp = lab == k
        if comp[iy0:iy1, ix0:ix1].any():
            keep[comp] = 1.0
    if not keep.any():
        return None
    m = cv2.GaussianBlur(keep * np.clip(skin * 1.4, 0, 1), (0, 0), 2.0)
    return np.clip(m, 0, 1)


def draw_cartoon_hands(canvas, frame_bgr, hm, M, pal, line_px):
    """Warp the hand mask from the source frame into the output and draw it as a flat cartoon hand."""
    ch, cw = canvas.shape[:2]
    sm = cv2.GaussianBlur(hm, (0, 0), 3.0)
    m = cv2.warpAffine(sm, M, (cw, ch), flags=cv2.INTER_LINEAR)
    binm = (m > 0.5).astype(np.uint8)
    if binm.sum() < 50:
        return
    fill = cv2.GaussianBlur(binm.astype(np.float32), (0, 0), 0.9)
    k = max(3, int(line_px) | 1)
    outline = cv2.GaussianBlur(cv2.dilate(binm, np.ones((k, k), np.uint8)).astype(np.float32), (0, 0), 0.9)
    # inner lines (finger gaps, knuckles) from strong edges only, keep longer strokes
    gray = cv2.GaussianBlur(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), (0, 0), 2.2)
    edges = cv2.Canny(gray, 35, 90)
    inner = cv2.erode((hm > 0.5).astype(np.uint8), np.ones((15, 15), np.uint8))
    edges = edges * inner
    n, lab, stats, _ = cv2.connectedComponentsWithStats((edges > 0).astype(np.uint8), connectivity=8)
    keep = np.zeros(edges.shape, np.float32)
    for j in range(1, n):
        if max(stats[j, cv2.CC_STAT_WIDTH], stats[j, cv2.CC_STAT_HEIGHT]) >= 28:
            keep[lab == j] = 1.0
    lines = cv2.warpAffine(cv2.dilate(keep, np.ones((2, 2), np.uint8)), M, (cw, ch), flags=cv2.INTER_LINEAR)
    lines = np.clip(cv2.GaussianBlur(lines, (0, 0), 0.7) * 2.0, 0, 1) * fill
    shade = np.clip((1 - cv2.GaussianBlur(fill, (0, 0), line_px * 2.5)) * fill * 1.6, 0, 1)
    img = canvas.astype(np.float32) / 255.0
    def over(img, a, col):
        a3 = a[..., None]
        return img * (1 - a3) + hexrgb(col)[::-1] * a3
    img = over(img, outline, pal["line"])
    img = over(img, fill, pal["skin"])
    img = over(img, shade * 0.8, pal["skin_dark"])
    img = over(img, lines * 0.6, pal["line"])
    canvas[:] = np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)


def eye_coverage(seg, video, pose, found, fps):
    """Per-frame fraction of the eye regions hidden by hands (only for untracked frames)."""
    cap = cv2.VideoCapture(video)
    cov = np.zeros(len(found), np.float32)
    i = 0
    while i < len(found):
        ok, frame = cap.read()
        if not ok:
            break
        if not found[i]:
            p = {k: (v[i] if np.ndim(v) else v) for k, v in pose.items()}
            hm = hand_mask(seg, frame, i, fps, p)
            if hm is not None:
                x0, y0, x1, y1 = [float(v) for v in p["face_box"]]
                fw = x1 - x0; ey = float(p["anchor"][1])
                vals = []
                for cx in (x0 + 0.3 * fw, x0 + 0.7 * fw):
                    bx0, bx1 = int(max(0, cx - 0.13 * fw)), int(cx + 0.13 * fw)
                    by0, by1 = int(max(0, ey - 0.08 * fw)), int(ey + 0.08 * fw)
                    patch = hm[by0:by1, bx0:bx1]
                    vals.append(float(patch.mean()) if patch.size else 0.0)
                cov[i] = max(vals)
        i += 1
    cap.release()
    return cov


def frame_expr(E, i):
    return {k: float(v[i]) for k, v in E.items()}


def render(video, npz, out, mode="character", palette="buddy", out_size=None, head_scale=1.0, preview_frames=None):
    data = np.load(npz, allow_pickle=False)
    E, pose, found, lsl = build_expressions(data)
    pal = PALETTES[palette]
    cap = cv2.VideoCapture(video)
    fps = float(data["fps"]); W, H = int(data["W"]), int(data["H"])
    n = len(found)
    med_anchor = np.median(pose["anchor"][found], axis=0)
    med_ec = float(np.median(pose["eye_chin"][found]))
    char_eye_chin = (CY + HEAD_B) - EYE_Y  # canvas units from eye line to chin

    if mode == "character":
        OW, OH = out_size or (W, H)
    elif mode == "side":
        OW, OH = W * 2, H
    else:
        OW, OH = W, H
    OW -= OW % 2; OH -= OH % 2
    bg = gradient_bg(OW, OH, pal["bg1"], pal["bg2"]) if mode in ("character", "side") else None

    writer = None
    if preview_frames is None:
        cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OW}x{OH}", "-r", f"{fps}",
               "-i", "-", "-i", video, "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", "18", "-preset", "medium",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out]
        writer = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    frames_out = []
    seg = make_segmenter()
    if seg is not None and (~found).any():
        cov = eye_coverage(seg, video, pose, found, fps)
        shut = gsmooth((cov > 0.12).astype(np.float32), 2.5)
        for s_ in "LR":
            E["blink" + s_] = np.maximum(E["blink" + s_], np.clip(shut * 1.3, 0, 1))
        seg = make_segmenter()  # fresh instance: VIDEO mode needs timestamps from zero again
    want = set(preview_frames or [])
    i = 0
    while i < n:
        ok, frame = cap.read()
        if not ok:
            break
        if preview_frames is not None and i not in want:
            i += 1
            continue
        e = frame_expr(E, i)
        p = {k: (v[i] if np.ndim(v) else v) for k, v in pose.items()}
        cv = draw_character(e, p, pal)
        if mode == "overlay":
            orig = frame.copy()
            scale = OV_SCALE * head_scale * float(p["eye_chin"]) / char_eye_chin
            place(frame, cv, p["anchor"], scale, float(p["roll"]))
            if seg is not None:
                hm = hand_mask(seg, orig, i, fps, p)
                if hm is not None:
                    a3 = hm[..., None]
                    frame[:] = np.clip(frame * (1 - a3) + orig * a3 + 0.5, 0, 255).astype(np.uint8)
            outf = frame
        else:
            canvas = bg.copy() if mode == "character" else bg[:, :W].copy()
            cw, ch = canvas.shape[1], canvas.shape[0]
            # same framing as the source video, so hands map 1:1 and never hit an edge
            g = ch / H
            ox = (cw - W * g) / 2
            scale = CHAR_SCALE * head_scale * OV_SCALE * float(p["eye_chin"]) / char_eye_chin * g
            ax, ay = ox + p["anchor"][0] * g, p["anchor"][1] * g
            draw_body(canvas, (ax, ay), scale, pal)
            place(canvas, cv, (ax, ay), scale, float(p["roll"]))
            if seg is not None:
                hm = hand_mask(seg, frame, i, fps, p, cut=False)
                if hm is not None:
                    Mh = np.array([[g, 0, ox], [0, g, 0]], np.float32)
                    draw_cartoon_hands(canvas, frame, hm, Mh, pal, max(3, 9 * scale))
            outf = canvas if mode == "character" else np.hstack([frame, canvas])
        if writer is not None:
            writer.stdin.write(np.ascontiguousarray(outf[:OH, :OW]).tobytes())
        else:
            frames_out.append((i, outf.copy()))
        i += 1
        if i % 150 == 0:
            print(f"  rendered {i}/{n}", flush=True)
    cap.release()
    if writer is not None:
        writer.stdin.close(); writer.wait()
        print("wrote", out)
    return frames_out


# --------------------------------------------------------------------------
# Expression sweep (visual test)
# --------------------------------------------------------------------------
SWEEP = [
    ("neutral", {}, {}),
    ("talk open", {"jaw": 0.55, "lowerDownL": 0.3, "lowerDownR": 0.3}, {}),
    ("big smile", {"smileL": 0.9, "smileR": 0.9, "cheekSqL": 0.6, "cheekSqR": 0.6}, {}),
    ("laugh", {"smileL": 0.9, "smileR": 0.9, "jaw": 0.6, "cheekSqL": 0.8, "cheekSqR": 0.8, "squintL": 0.5, "squintR": 0.5}, {}),
    ("surprise", {"jaw": 0.7, "funnel": 0.5, "wideL": 1, "wideR": 1, "browInner": 1, "browOuterL": 1, "browOuterR": 1}, {}),
    ("angry", {"browDownL": 1, "browDownR": 1, "frownL": 0.6, "frownR": 0.6, "squintL": 0.4, "squintR": 0.4, "sneerL": 0.6, "sneerR": 0.6}, {}),
    ("sad", {"browInner": 1, "frownL": 0.9, "frownR": 0.9, "blinkL": 0.25, "blinkR": 0.25}, {}),
    ("blink", {"blinkL": 1, "blinkR": 1}, {}),
    ("wink + smirk", {"blinkR": 1, "smileL": 0.7, "browOuterL": 0.6, "mouthShift": -0.4}, {}),
    ("pucker", {"pucker": 1, "puff": 0.6}, {}),
    ("oh", {"jaw": 0.45, "funnel": 1, "pucker": 0.4}, {}),
    ("look left, turn", {"gazeX": -0.9, "smileL": 0.3, "smileR": 0.3}, {"yaw": -0.6}),
    ("look up, turn", {"gazeY": -0.8, "browInner": 0.4}, {"yaw": 0.6, "pitch": -0.5}),
    ("disgust", {"sneerL": 1, "sneerR": 1, "upperUpL": 0.6, "upperUpR": 0.6, "browDownL": 0.6, "browDownR": 0.6, "jaw": 0.12}, {}),
    ("smile talk", {"smileL": 0.6, "smileR": 0.6, "jaw": 0.35}, {}),
    ("tilt", {"smileL": 0.4, "smileR": 0.4}, {"roll": 15}),
]


def sweep(out_png, palette="buddy", cell=320):
    pal = PALETTES[palette]
    cols = 4
    rows = math.ceil(len(SWEEP) / cols)
    sheet = np.full((rows * (cell + 34), cols * cell, 3), 245, np.uint8)
    for k, (label, e, p) in enumerate(SWEEP):
        cv = draw_character(e, p, pal)
        tile = gradient_bg(cell, cell, pal["bg1"], pal["bg2"])
        s = cell / CS * 1.05
        place(tile, cv, (cell / 2, cell * 0.50), s, p.get("roll", 0.0))
        r, c = divmod(k, cols)
        y0 = r * (cell + 34)
        sheet[y0:y0 + cell, c * cell:(c + 1) * cell] = tile
        cv2.putText(sheet, label, (c * cell + 10, y0 + cell + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.imwrite(out_png, sheet)
    print("wrote", out_png)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("track"); a.add_argument("video"); a.add_argument("out"); a.add_argument("--model", default=DEFAULT_MODEL)
    b = sub.add_parser("render"); b.add_argument("video"); b.add_argument("npz"); b.add_argument("out")
    b.add_argument("--mode", default="character", choices=["character", "overlay", "side"])
    b.add_argument("--palette", default="buddy", choices=list(PALETTES))
    b.add_argument("--head-scale", type=float, default=1.0)
    b.add_argument("--preview", default=None, help="comma separated frame indices -> PNGs instead of video")
    c = sub.add_parser("sweep"); c.add_argument("out"); c.add_argument("--palette", default="buddy", choices=list(PALETTES))
    d = sub.add_parser("normalize"); d.add_argument("src"); d.add_argument("dst"); d.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    if args.cmd == "track":
        track(args.video, args.out, args.model)
    elif args.cmd == "render":
        if args.preview:
            idx = [int(x) for x in args.preview.split(",")]
            frames = render(args.video, args.npz, args.out, args.mode, args.palette, head_scale=args.head_scale, preview_frames=idx)
            base, _ = os.path.splitext(args.out)
            for i, f in frames:
                cv2.imwrite(f"{base}_{i:05d}.png", f)
                print("wrote", f"{base}_{i:05d}.png")
        else:
            render(args.video, args.npz, args.out, args.mode, args.palette, head_scale=args.head_scale)
    elif args.cmd == "sweep":
        sweep(args.out, args.palette)
    elif args.cmd == "normalize":
        normalize_video(args.src, args.dst, args.fps)


if __name__ == "__main__":
    main()
