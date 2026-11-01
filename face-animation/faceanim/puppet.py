#!/usr/bin/env python3
"""Photo puppet: animate a still portrait with the tracked performance (relative landmark motion transfer
+ piecewise-affine mesh warp + synthesized mouth interior).

usage: puppet.py <portrait.jpg> <track.npz> <driver_video> <out.mp4|out_prefix> [--frames 0,10] [--size 720x1280]
"""
import argparse
import os
import subprocess

import cv2
import numpy as np
from scipy.spatial import Delaunay
from matplotlib.path import Path

import faceanim as fa

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TRIS = np.load(os.path.join(ROOT, "assets", "face_tris.npy"))

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
             176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
INNER_LIPS = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LOWER_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
EYE_R = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
EYE_L = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
IRIS_R = [468, 469, 470, 471, 472]
IRIS_L = [473, 474, 475, 476, 477]
LID_PAIRS = [(246, 7), (161, 163), (160, 144), (159, 145), (158, 153), (157, 154), (173, 155),
             (466, 249), (388, 390), (387, 373), (386, 374), (385, 380), (384, 381), (398, 382)]
LIPS = sorted(set([61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409,
                   78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415]))
BROWS = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
STABLE = [10, 151, 9, 8, 168, 6, 197, 195, 127, 356, 33, 263, 133, 362, 109, 338, 67, 297]


# ---------------------------------------------------------------- geometry helpers
def umeyama(a, b):
    """Similarity (3x3) mapping points a -> b (least squares)."""
    ma, mb = a.mean(0), b.mean(0)
    A, B = a - ma, b - mb
    U, S, Vt = np.linalg.svd(B.T @ A / len(a))
    D = np.eye(2)
    if np.linalg.det(U @ Vt) < 0:
        D[1, 1] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (A ** 2).sum(1).mean()
    T = np.eye(3); T[:2, :2] = s * R; T[:2, 2] = mb - s * R @ ma
    return T


def umeyama3(a, b):
    """Similarity mapping 3D points a -> b: returns s, R (3x3), t."""
    ma, mb = a.mean(0), b.mean(0)
    A, B = a - ma, b - mb
    U, S, Vt = np.linalg.svd(B.T @ A / len(a))
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (A ** 2).sum(1).mean()
    return s, R, mb - s * R @ ma


def apply(T, p):
    return p @ T[:2, :2].T + T[:2, 2]


def detect(img_bgr, model=fa.DEFAULT_MODEL):
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    import mediapipe as mp
    opts = vision.FaceLandmarkerOptions(base_options=mp_python.BaseOptions(model_asset_path=model), num_faces=1,
                                        running_mode=vision.RunningMode.IMAGE)
    lm = vision.FaceLandmarker.create_from_options(opts)
    h, w = img_bgr.shape[:2]
    res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)))
    if not res.face_landmarks:
        raise SystemExit("no face found in portrait")
    return np.array([[p.x * w, p.y * h, p.z * w] for p in res.face_landmarks[0]], np.float32)


# ---------------------------------------------------------------- mesh
def build_mesh(Ls, W, H, head_box=None):
    """Control points (source) and triangles. Returns pts, tris_outer, tris_face, ring info.
    head_box (x0, y0, x1, y1): region that moves rigidly with the head (face + all of the hair)."""
    oval = Ls[FACE_OVAL]
    c = oval.mean(0)
    hw = (oval[:, 0].max() - oval[:, 0].min()) / 2
    hh = (oval[:, 1].max() - oval[:, 1].min()) / 2
    t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    if head_box is not None:
        x0, y0, x1, y1 = head_box
        bc = np.array([(x0 + x1) / 2, (y0 + y1) / 2]); ax = (x1 - x0) / 2 * 1.1; ay = (y1 - y0) / 2 * 1.12
        ring1 = np.stack([bc[0] + ax * np.cos(t), bc[1] + ay * np.sin(t)], 1)
        ring2 = np.stack([bc[0] + ax * 1.7 * np.cos(t), bc[1] + 0.1 * ay + ay * 1.6 * np.sin(t)], 1)
    else:
        ring1 = np.stack([c[0] + 1.32 * hw * np.cos(t), c[1] - 0.18 * hh + 1.45 * hh * np.sin(t)], 1)
        ring2 = np.stack([c[0] + 2.3 * hw * np.cos(t), c[1] - 0.1 * hh + 2.2 * hh * np.sin(t)], 1)
    bx = np.linspace(0, W - 1, 7); by = np.linspace(0, H - 1, 11)
    border = np.concatenate([np.stack([bx, np.zeros_like(bx)], 1), np.stack([bx, np.full_like(bx, H - 1)], 1),
                             np.stack([np.zeros_like(by[1:-1]), by[1:-1]], 1), np.stack([np.full_like(by[1:-1], W - 1), by[1:-1]], 1)])
    pts = np.concatenate([Ls, ring1, ring2, border]).astype(np.float32)
    n_l = len(Ls); i_r1 = n_l; i_r2 = n_l + len(ring1); i_b = i_r2 + len(ring2)
    # outer triangulation (drawn first, face mesh overwrites the face area)
    outer_idx = np.array(FACE_OVAL + list(range(i_r1, len(pts))))
    d = Delaunay(pts[outer_idx])
    tris_outer = outer_idx[d.simplices]
    # face: canonical mesh, eyes re-triangulated with iris points so gaze moves the iris
    cen = Ls[TRIS].mean(1)
    keep = np.ones(len(TRIS), bool)
    eye_tris = []
    for contour, iris in ((EYE_R, IRIS_R), (EYE_L, IRIS_L)):
        poly = Path(Ls[contour])
        keep &= ~poly.contains_points(cen)
        idx = np.array(contour + iris)
        de = Delaunay(Ls[idx])
        tt = idx[de.simplices]
        inside = poly.contains_points(Ls[tt].mean(1))
        eye_tris.append(tt[inside])
    tris_face = np.concatenate([TRIS[keep]] + eye_tris)
    return pts, tris_outer, tris_face, (i_r1, i_r2, i_b)


def warp(src, src_pts, dst_pts, tris_outer, tris_face):
    H, W = src.shape[:2]
    tris = np.concatenate([tris_outer, tris_face])
    idmap = np.zeros((H, W), np.int32)
    q = np.round(dst_pts * 16).astype(np.int32)
    for k, tr in enumerate(tris):
        cv2.fillConvexPoly(idmap, q[tr], k + 1, cv2.LINE_8, 4)
    # affine per triangle: dst -> src
    D = np.concatenate([dst_pts[tris], np.ones((len(tris), 3, 1), np.float32)], 2)  # (n,3,3)
    S = src_pts[tris]                                                                # (n,3,2)
    # [x y 1] @ A = src ; collapsed triangles (e.g. a fully closed eye) get a pseudo-inverse
    det = np.linalg.det(D)
    ok = np.abs(det) > 1e-4
    A = np.zeros((len(tris), 3, 2), np.float64)
    A[ok] = np.linalg.solve(D[ok].astype(np.float64), S[ok].astype(np.float64))
    if (~ok).any():
        A[~ok] = np.linalg.pinv(D[~ok].astype(np.float64)) @ S[~ok].astype(np.float64)
    A = np.concatenate([np.zeros((1, 3, 2), A.dtype), A])  # id 0 -> identity handled below
    A[0] = np.array([[1, 0], [0, 1], [0, 0]], A.dtype)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    Ai = A[idmap]
    mx = xx * Ai[..., 0, 0] + yy * Ai[..., 1, 0] + Ai[..., 2, 0]
    my = xx * Ai[..., 0, 1] + yy * Ai[..., 1, 1] + Ai[..., 2, 1]
    return cv2.remap(src, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


# ---------------------------------------------------------------- mouth interior
def draw_mouth(img, P, mouth_w):
    """Synthesize the mouth interior inside the target inner-lip polygon (in place)."""
    poly = P[INNER_LIPS]
    area = cv2.contourArea(poly.astype(np.float32))
    open_ratio = area / (mouth_w ** 2 + 1e-6)
    if open_ratio < 0.004:
        return
    H, W = img.shape[:2]
    x0, y0 = np.floor(poly.min(0)).astype(int) - 4
    x1, y1 = np.ceil(poly.max(0)).astype(int) + 4
    x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, W), min(y1, H)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return
    off = np.array([x0, y0], np.float32)
    ss = 4  # supersample
    w, h = (x1 - x0) * ss, (y1 - y0) * ss
    pp = (poly - off) * ss
    m = np.zeros((h, w), np.uint8); cv2.fillPoly(m, [np.round(pp).astype(np.int32)], 255, cv2.LINE_AA)
    mask = cv2.resize(m, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    mask = cv2.GaussianBlur(mask, (0, 0), 0.6)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    up = (P[UPPER_INNER] - off) * ss; lo = (P[LOWER_INNER] - off) * ss
    yu = np.interp(xs[0], up[:, 0], up[:, 1]) if up[0, 0] < up[-1, 0] else np.interp(xs[0], up[::-1, 0], up[::-1, 1])
    yl = np.interp(xs[0], lo[:, 0], lo[:, 1]) if lo[0, 0] < lo[-1, 0] else np.interp(xs[0], lo[::-1, 0], lo[::-1, 1])
    gap = np.maximum(yl - yu, 1)
    tdepth = (ys - yu[None]) / gap[None]  # 0 at upper lip, 1 at lower lip
    cx = (xs - pp[:, 0].min()) / max(pp[:, 0].max() - pp[:, 0].min(), 1)
    edge = np.clip(1 - np.abs(cx - 0.5) * 2, 0, 1) ** 0.6  # darker toward the corners
    # cavity
    col = np.zeros((h, w, 3), np.float32)
    cav_dark = np.array([12, 9, 20], np.float32); cav_light = np.array([36, 30, 66], np.float32)
    col[:] = cav_dark + (cav_light - cav_dark) * (edge * np.clip(tdepth, 0, 1))[..., None] * 0.8
    # tongue (lower middle)
    tongue = np.clip(1 - (((cx - 0.5) / 0.34) ** 2 + ((tdepth - 1.05) / 0.55) ** 2), 0, 1)
    col = col * (1 - tongue[..., None] * 0.9) + np.array([92, 88, 168], np.float32) * (0.55 + 0.45 * edge)[..., None] * tongue[..., None] * 0.9
    # upper teeth: band under the upper lip, shown once the mouth is open enough
    opened = np.clip((open_ratio - 0.015) / 0.05, 0, 1)
    tooth_h = np.minimum(gap * 0.33, mouth_w * ss * 0.085)
    teeth = np.clip(1 - (ys - yu[None]) / tooth_h[None], 0, 1)
    teeth = (teeth > 0.02).astype(np.float32) * np.clip(edge * 1.5, 0, 1) ** 1.3 * opened
    teeth = cv2.GaussianBlur(teeth, (0, 0), ss * 0.7)
    # warm off-white, shadowed under the lip and toward the corners
    vt = np.clip((ys - yu[None]) / np.maximum(tooth_h[None], 1), 0, 1)
    tcol = np.array([198, 207, 218], np.float32) * ((0.62 + 0.36 * edge) * (0.78 + 0.22 * np.sin(np.pi * np.clip(vt * 0.9 + 0.1, 0, 1))))[..., None]
    # individual teeth: darker separations
    period = mouth_w * ss * 0.11
    sep = 0.5 + 0.5 * np.cos((xs - (pp[:, 0].min() + pp[:, 0].max()) / 2) / period * 2 * np.pi)
    tcol = tcol * (0.78 + 0.22 * sep[..., None] ** 0.35)
    col = col * (1 - teeth[..., None]) + tcol * teeth[..., None]
    # lower teeth hint for wide openings
    lower = np.clip(1 - (yl[None] - ys) / (tooth_h[None] * 0.6), 0, 1)
    lower = (lower > 0.02).astype(np.float32) * edge * np.clip((open_ratio - 0.08) / 0.06, 0, 1) * 0.6
    lower = cv2.GaussianBlur(lower, (0, 0), ss * 0.6)
    col = col * (1 - lower[..., None]) + np.array([190, 198, 206], np.float32) * lower[..., None]
    # lip contact shadow
    near = np.minimum(np.abs(ys - yu[None]), np.abs(yl[None] - ys)) / (mouth_w * ss * 0.05)
    col *= (0.42 + 0.58 * np.clip(near, 0, 1))[..., None]
    col = cv2.resize(col, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    roi = img[y0:y1, x0:x1].astype(np.float32)
    a = mask[..., None]
    img[y0:y1, x0:x1] = np.clip(roi * (1 - a) + col * a, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- eyelids
def lid_open(P, pairs):
    return float(np.mean([np.linalg.norm(P[u] - P[l]) for u, l in pairs]))


def draw_eyelids(img, P, src_open):
    """Cover a closing eye with eyelid skin (inpainted from the surrounding skin) and a soft lash line."""
    H, W = img.shape[:2]
    for contour, pairs, k in ((EYE_R, LID_PAIRS[:7], 0), (EYE_L, LID_PAIRS[7:], 1)):
        op = lid_open(P, pairs)
        c = np.clip(1 - op / max(src_open[k], 1e-3), 0, 1)
        a = np.clip((c - 0.45) / 0.35, 0, 1); a = a * a * (3 - 2 * a)
        if a <= 0.01:
            continue
        poly = P[contour].astype(np.float32)
        ew = float(np.linalg.norm(P[contour[0]] - P[contour[8]]))
        pad = int(ew * 0.6)
        x0, y0 = np.floor(poly.min(0)).astype(int) - pad; x1, y1 = np.ceil(poly.max(0)).astype(int) + pad
        x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, W), min(y1, H)
        roi = img[y0:y1, x0:x1].copy()
        off = np.array([x0, y0], np.float32)
        # cover the closed eye plus the band above it where the photo's lash line and eyeball top got stretched
        upper_t = np.array([P[u] for u, _ in pairs], np.float32)
        lower_t = np.array([P[l] for _, l in pairs], np.float32)
        h_up = 0.6 * src_open[k] * a + 1.5
        h_dn = 0.1 * src_open[k] + 1.0
        ca, cb = P[contour[0]].astype(np.float32), P[contour[8]].astype(np.float32)
        out_dir = (ca - cb) / (np.linalg.norm(ca - cb) + 1e-6)
        ca2, cb2 = ca + out_dir * 2, cb - out_dir * 2
        top = upper_t - [0, h_up]; bot = lower_t + [0, h_dn]
        order_t = np.argsort(top[:, 0]); order_b = np.argsort(bot[:, 0])
        left, right = (ca2, cb2) if ca2[0] < cb2[0] else (cb2, ca2)
        grown = np.concatenate([[left], top[order_t], [right], bot[order_b][::-1]])
        m = np.zeros(roi.shape[:2], np.uint8)
        cv2.fillPoly(m, [np.round(grown - off).astype(np.int32)], 255)
        m = cv2.dilate(m, np.ones((3, 3), np.uint8))
        lid = cv2.inpaint(roi, m, max(3, int(ew * 0.12)), cv2.INPAINT_TELEA)
        lid = cv2.GaussianBlur(lid, (0, 0), 0.8)
        # skin grain so the patch is not plastic-smooth
        noise = np.random.default_rng(k).normal(0, 2.2, lid.shape[:2]).astype(np.float32)
        lid = np.clip(lid.astype(np.float32) + noise[..., None], 0, 255)
        mf = cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), max(1.0, ew * 0.035)) * a
        out = roi.astype(np.float32) * (1 - mf[..., None]) + lid * mf[..., None]
        # lash line: along the meeting line of the lids, darker in the middle, fading at the corners
        lower = np.array([P[l] for _, l in pairs], np.float32)
        upper = np.array([P[u] for u, _ in pairs], np.float32)
        line = 0.4 * upper + 0.6 * lower
        path = np.concatenate([[P[contour[0]]], line, [P[contour[8]]]]) - off
        path = path[np.argsort(path[:, 0])]
        lm = np.zeros(roi.shape[:2], np.uint8)
        cv2.polylines(lm, [np.round(path * 4).astype(np.int32)], False, 255, max(1, int(round(ew * 0.035))), cv2.LINE_AA, 2)
        lmf = cv2.GaussianBlur(lm.astype(np.float32) / 255, (0, 0), 0.9)
        xs = np.arange(roi.shape[1], dtype=np.float32)[None, :]
        xc = (path[0, 0] + path[-1, 0]) / 2; hw_ = max((path[-1, 0] - path[0, 0]) / 2, 1)
        lmf *= np.clip(1 - np.abs(xs - xc) / hw_, 0, 1) ** 0.5 * a * 0.6
        dark = np.array([30, 28, 32], np.float32)
        out = out * (1 - lmf[..., None]) + dark * lmf[..., None]
        img[y0:y1, x0:x1] = np.clip(out, 0, 255).astype(np.uint8)


def build_closed_eyes(src, Ls):
    """Portrait with both eyes closed, made like a real blink: the upper-lid skin and its lash line
    (taken from just above each eye in the same image) slide down until the lashes meet the lower lid."""
    out = src.copy()
    masks = []
    H, W = src.shape[:2]
    for contour, pairs in ((EYE_R, LID_PAIRS[:7]), (EYE_L, LID_PAIRS[7:])):
        up = np.array([Ls[u] for u, _ in pairs], np.float32)
        lo = np.array([Ls[l] for _, l in pairs], np.float32)
        ca, cb = Ls[contour[0]].astype(np.float32), Ls[contour[8]].astype(np.float32)
        left, right = (ca, cb) if ca[0] < cb[0] else (cb, ca)
        h = float(max(np.linalg.norm(Ls[u] - Ls[l]) for u, l in pairs))
        ux = np.concatenate([[left[0]], up[:, 0], [right[0]]]); uy = np.concatenate([[left[1]], up[:, 1], [right[1]]])
        lx = np.concatenate([[left[0]], lo[:, 0], [right[0]]]); ly = np.concatenate([[left[1]], lo[:, 1], [right[1]]])
        ou = np.argsort(ux); ol = np.argsort(lx)
        x0 = int(np.floor(left[0] - 0.15 * h)); x1 = int(np.ceil(right[0] + 0.15 * h))
        xs = np.arange(max(x0, 0), min(x1, W), dtype=np.float32)
        y_up = np.interp(xs, ux[ou], uy[ou]) + 0.02 * h          # lid margin (lash roots)
        y_lo = np.interp(xs, lx[ol], ly[ol]) + 0.12 * h          # a little below the lower lid: no eyeball sliver
        lt = 0.24 * h                                             # lash strip thickness, moved without stretching
        y_top = y_up - 0.42 * h                                   # just below the lid crease: the crease stays put
        # columns fade out beyond the eye corners
        wcol = np.clip(np.minimum(xs - left[0], right[0] - xs) / (0.35 * h) + 0.25, 0, 1)
        yt0 = int(max(0, np.floor(y_top.min()) - 2)); yb1 = int(min(H, np.ceil(y_lo.max()) + 3))
        yy = np.arange(yt0, yb1, dtype=np.float32)[:, None]
        # target rows [y_top, y_lo] sample source rows [y_top, y_up]: the lid stretches down, lashes land on the lower lid
        # lid skin [y_top, y_up - lt] stretches over [y_top, y_lo - lt]; lash strip [y_up - lt, y_up] moves down intact
        skin_end_t = (y_lo - lt)[None]; skin_end_s = (y_up - lt)[None]
        frac = (yy - y_top[None]) / np.maximum(skin_end_t - y_top[None], 1)
        sy_skin = y_top[None] + np.clip(frac, 0, 1) * (skin_end_s - y_top[None])
        sy_lash = yy - (y_lo - y_up)[None]
        sy = np.where(yy < skin_end_t, sy_skin, np.clip(sy_lash, skin_end_s, (y_up + 0.04 * h)[None]))
        mx = np.repeat(xs[None], len(yy), 0).astype(np.float32)
        my = np.where((yy >= y_top[None]) & (yy <= y_lo[None] + 1.5), sy, yy).astype(np.float32)
        patch = cv2.remap(src, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)
        # the stretched skin is smoothed sideways (no vertical streaks) and given fine grain back
        is_skin = ((yy >= y_top[None]) & (yy < skin_end_t)).astype(np.float32)
        sm = cv2.GaussianBlur(patch, (0, 0), sigmaX=max(1.0, 0.16 * h), sigmaY=0.6)
        grain = np.random.default_rng(len(masks) + 7).normal(0, 2.2, patch.shape[:2]).astype(np.float32)
        sm = sm + cv2.GaussianBlur(grain, (0, 0), 0.6)[..., None]
        # lid tone: pull toward the skin above the crease, plus a soft highlight where the eyeball rounds the lid
        ref_rows = (int(max(0, np.median(y_up) - 1.0 * h)), int(max(1, np.median(y_up) - 0.7 * h)))
        xa_ = int(xs[0]); xb_ = xa_ + len(xs)
        skin_ref = np.median(src[ref_rows[0]:ref_rows[1], xa_:xb_].reshape(-1, 3), axis=0).astype(np.float32)
        sm = sm * 0.72 + skin_ref[None, None, :] * 0.28
        cxe = 0.5 * (left[0] + right[0]); hw_e = 0.5 * (right[0] - left[0])
        ycen = 0.5 * (y_top + skin_end_t[0])
        hl = np.exp(-(((xs[None] - cxe) / (0.65 * hw_e)) ** 2) - (((yy - ycen[None]) / (0.45 * h)) ** 2))
        sm = sm * (1 + 0.07 * hl[..., None])
        ks = cv2.GaussianBlur(is_skin, (0, 0), 1.2)[..., None]
        patch = patch * (1 - ks) + sm * ks
        inside = ((yy >= y_top[None]) & (yy <= y_lo[None] + 1.5)).astype(np.float32)
        wm = inside * wcol[None]
        # soft top edge (the lid crease stays put) and soft bottom edge
        topfade = np.clip((yy - y_top[None]) / (0.12 * h), 0, 1)
        wm = cv2.GaussianBlur(wm * topfade, (0, 0), max(1.0, 0.06 * h))
        xa = int(xs[0]); xb = xa + len(xs)
        roi = out[yt0:yb1, xa:xb].astype(np.float32)
        out[yt0:yb1, xa:xb] = np.clip(roi * (1 - wm[..., None]) + patch * wm[..., None], 0, 255).astype(np.uint8)
        m = np.zeros((H, W), np.float32); m[yt0:yb1, xa:xb] = wm
        masks.append(m)
    return out, masks


def draw_smile_lines(img, P, smile, mw):
    """Nasolabial folds and lifted cheeks, scaled by how much the mouth corners rise."""
    if smile <= 0.05:
        return
    s = float(np.clip(smile, 0, 1.4))
    H, W = img.shape[:2]
    layer = np.zeros((H, W), np.float32)
    hl = np.zeros((H, W), np.float32)
    for nose, mid, corner, cheek, sgn in ((129, 206, 61, 50, -1), (358, 426, 291, 280, 1)):
        a = P[nose]; c = P[corner] + np.array([sgn * 0.13 * mw, -0.02 * mw])
        b = 0.5 * (a + c) + np.array([sgn * 0.08 * mw, 0])
        t = np.linspace(0, 1, 24)[:, None]
        curve = (1 - t) ** 2 * a + 2 * (1 - t) * t * b + t ** 2 * c
        cv2.polylines(layer, [np.round(curve * 4).astype(np.int32)], False, 1.0, max(2, int(0.07 * mw)), cv2.LINE_AA, 2)
        cv2.ellipse(hl, tuple(np.round(P[cheek]).astype(int)), (int(0.28 * mw), int(0.18 * mw)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    layer = cv2.GaussianBlur(layer, (0, 0), max(1.5, 0.045 * mw))
    hl = cv2.GaussianBlur(hl, (0, 0), max(2.0, 0.12 * mw))
    f = img.astype(np.float32)
    f *= (1 - 0.26 * s * layer)[..., None]
    f = f + (255 - f) * (0.07 * s * hl)[..., None]
    img[:] = np.clip(f, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- driver signals
def driver_landmarks(npz, video):
    data = np.load(npz)
    found = fa.clean_found(data["found"].astype(bool))
    L = fa.fill_gaps_hold(data["lms"], found).astype(np.float32)
    L = fa.gsmooth(L, 1.0)
    # neutral reference: least expressive tracked frame
    E, pose, _, _ = fa.build_expressions(data)
    score = (E["blinkL"] + E["blinkR"] + 2 * E["lipGap"] + E["smileL"] + E["smileR"] + E["wideL"] + E["wideR"]
             + E["browInner"] + 0.5 * (E["browOuterL"] + E["browOuterR"]))
    score = np.where(found, score, 1e9)
    ref = int(np.argmin(fa.gsmooth(score, 3.0)))
    # eyes shut while hands cover the face during a long dropout
    shut = np.zeros(len(found), np.float32)
    seg = fa.make_segmenter()
    if seg is not None and (~found).any():
        cov = fa.eye_coverage(seg, video, pose, found, float(data["fps"]))
        raw = (cov > 0.12).astype(np.float32)
        i, n = 0, len(found)
        while i < n:
            if not found[i]:
                j = i
                while j < n and not found[j]:
                    j += 1
                if j - i >= 15 and np.mean(cov[i:j] > 0.03) >= 0.3:
                    raw[i:j] = 1
                i = j
            else:
                i += 1
        shut = fa.gsmooth(raw, 2.5)
    # weight that ramps up inside long tracking dropouts (used to ease toward a calm pose)
    n = len(found)
    dist = np.full(n, 1e9)
    idx = np.where(found)[0]
    if len(idx):
        for k in range(n):
            dist[k] = np.min(np.abs(idx - k))
    gapw = fa.gsmooth(np.clip((dist - 2) / 10.0, 0, 1).astype(np.float32), 2.0)
    return L, ref, shut, float(data["fps"]), gapw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("portrait"); ap.add_argument("npz"); ap.add_argument("video"); ap.add_argument("out")
    ap.add_argument("--frames", default=None)
    ap.add_argument("--size", default="720x1280")
    ap.add_argument("--expr", type=float, default=1.25, help="expression strength")
    ap.add_argument("--lips", type=float, default=1.25, help="extra gain on lip motion")
    ap.add_argument("--corners", type=float, default=1.45, help="extra gain on mouth corners (smile)")
    ap.add_argument("--move", type=float, default=0.8, help="head translation strength")
    ap.add_argument("--bg", default=None, help="background image (used when the portrait has transparency)")
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--end", type=int, default=None)
    args = ap.parse_args()
    OW, OH = map(int, args.size.split("x"))

    raw = cv2.imread(args.portrait, cv2.IMREAD_UNCHANGED)
    has_alpha = raw.ndim == 3 and raw.shape[2] == 4
    if has_alpha and raw.shape[1] == OW and raw.shape[0] == OH:
        fitted = raw
    else:
        # fit portrait to the output frame (cover)
        s = max(OW / raw.shape[1], OH / raw.shape[0])
        fitted = cv2.resize(raw, (int(round(raw.shape[1] * s)), int(round(raw.shape[0] * s))), interpolation=cv2.INTER_AREA)
        ox, oy = (fitted.shape[1] - OW) // 2, (fitted.shape[0] - OH) // 2
        fitted = fitted[oy:oy + OH, ox:ox + OW].copy()
    src = fitted[..., :3].copy()
    src_a = fitted[..., 3].astype(np.float32) / 255.0 if has_alpha else None
    bg = None
    if src_a is not None:
        bg = cv2.resize(cv2.imread(args.bg), (OW, OH)) if args.bg else np.full((OH, OW, 3), 40, np.uint8)
        det_img = np.clip(src * src_a[..., None] + bg * (1 - src_a[..., None]), 0, 255).astype(np.uint8)
    else:
        det_img = src
    Ls3 = detect(det_img)
    Ls = Ls3[:, :2].copy()
    head_box = None
    if src_a is not None:
        chin_y = Ls[152, 1]
        rows = np.where((src_a > 0.5).any(1))[0]
        ys = np.arange(OH)
        mask_head = (src_a > 0.5) & (ys[:, None] < chin_y)
        cols = np.where(mask_head.any(0))[0]
        head_box = (float(cols.min()), float(rows.min()), float(cols.max()), float(chin_y))
        print("head box (face + hair):", [round(v) for v in head_box])
    pts, tris_outer, tris_face, (i_r1, i_r2, i_b) = build_mesh(Ls, OW, OH, head_box)

    Ld, ref, shut, fps, gapw = driver_landmarks(args.npz, args.video)
    print("neutral reference frame:", ref)
    R3 = Ld[ref]
    R = R3[:, :2]
    S0 = umeyama(R[STABLE], Ls[STABLE])           # driver ref -> source
    S0i = np.linalg.inv(S0)
    mouth_w_src = np.linalg.norm(Ls[61] - Ls[291])
    src_open = (lid_open(Ls, LID_PAIRS[:7]), lid_open(Ls, LID_PAIRS[7:]))
    gains = np.full(len(Ls), args.expr, np.float32)
    gains[LIPS] = args.expr * args.lips
    gains[BROWS] = args.expr * 1.15
    gains[[61, 291, 146, 91, 185, 40, 375, 321, 409, 270, 76, 306, 62, 292]] = args.expr * args.corners
    gains[FACE_OVAL] = 0.5

    c3s = Ls3[STABLE].mean(0)
    s0 = np.sqrt(abs(np.linalg.det(S0[:2, :2])))

    def target(t):
        P3 = Ld[t]
        sc, Rm, tr = umeyama3(R3[STABLE], P3[STABLE])   # 3D head motion (ref -> current)
        Pn3 = ((P3 - tr) @ Rm) / sc                      # frontalised: pose removed in 3D
        Pn = Pn3[:, :2]
        rv = cv2.Rodrigues(Rm.astype(np.float64))[0].ravel()
        gw = float(gapw[t])
        if gw > 0.001:                              # inside a long dropout: ease toward a calm, level pose
            Pn = Pn + gw * 0.85 * (R - Pn)
            rv = rv * (1 - 0.8 * gw)
        rv_d = rv * np.array([0.45, 0.55, 1.0])     # soften pitch / yaw, keep roll
        Rd = cv2.Rodrigues(rv_d)[0].astype(np.float32)
        # 2D part of the head motion (roll, scale, translation of the head centre) for rings and placement
        cen_ref = R3[STABLE].mean(0); cen_cur = P3[STABLE].mean(0)
        move = (cen_cur - cen_ref)[:2] * args.move * (1 - 0.7 * gw)
        scl = 1 + (sc - 1) * (1 - 0.7 * gw)
        roll = rv_d[2]
        M = np.eye(3)
        c2, s2 = np.cos(roll), np.sin(roll)
        M[:2, :2] = scl * np.array([[c2, -s2], [s2, c2]])
        M[:2, 2] = cen_ref[:2] + move - M[:2, :2] @ cen_ref[:2]
        delta = (Pn - R) * gains[:, None]
        Lexp = Ls + delta @ S0[:2, :2].T           # expression deltas mapped onto the portrait
        # eyelids: transfer openness as a ratio, so a closed driver eye fully closes the portrait's eye
        closes = []
        rs = [float(np.clip(lid_open(Pn, pr) / max(lid_open(R, pr), 1e-3), 0, 1.5)) for pr in (LID_PAIRS[:7], LID_PAIRS[7:])]
        rm = 0.5 * (rs[0] + rs[1])
        if min(rs) < 0.7 and abs(rs[0] - rs[1]) < 0.45:   # a blink, not a wink: close both eyes together
            k = 0.25
            rs = [rm + k * (rs[0] - rm), rm + k * (rs[1] - rm)]
        for (pairs, contour, iris), r in zip(((LID_PAIRS[:7], EYE_R, IRIS_R), (LID_PAIRS[7:], EYE_L, IRIS_L)), rs):
            r = r * (1 - shut[t])
            closes.append(float(np.clip((0.33 - r) / 0.09, 0, 1)))
            r = max(r, 0.28)
            for u, l in pairs:
                Lexp[u] = Lexp[l] + (Ls[u] - Ls[l]) * r
            # keep the iris inside the lids: squash it toward the lower lid as the eye closes
            if r < 1.0:
                lo = np.array([Lexp[l] for _, l in pairs]); lo_s = np.array([Ls[l] for _, l in pairs])
                for i in iris:
                    xi = Lexp[i, 0]
                    order = np.argsort(lo[:, 0])
                    yl = np.interp(xi, lo[order, 0], lo[order, 1])
                    yls = np.interp(Ls[i, 0], lo_s[np.argsort(lo_s[:, 0]), 0], lo_s[np.argsort(lo_s[:, 0]), 1])
                    Lexp[i, 1] = yl + (Ls[i, 1] - yls) * r
        G = S0 @ M @ S0i                           # 2D head motion in portrait space
        # portrait landmarks rotated in 3D (softened yaw/pitch/roll), scaled and moved like the head
        L3 = np.concatenate([Lexp, Ls3[:, 2:3]], 1)
        L3r = (L3 - c3s) @ Rd.T * scl + c3s
        cen_s = apply(G, c3s[None, :2])[0]
        Lrot2 = L3r[:, :2] - c3s[:2] + cen_s
        mw_r = np.linalg.norm(R[61] - R[291])
        smile = (-(delta[61, 1] + delta[291, 1]) / 2 + 0.5 * (delta[13, 1] + delta[14, 1])) / mw_r
        smile += 0.5 * max(0.0, (np.linalg.norm(Pn[61] - Pn[291]) / mw_r) - 1.03)
        tgt = pts.copy()
        tgt[:len(Ls)] = Lrot2
        tgt[i_r1:i_r2] = apply(G, pts[i_r1:i_r2])
        tgt[i_r2:i_b] = pts[i_r2:i_b] + 0.35 * (apply(G, pts[i_r2:i_b]) - pts[i_r2:i_b])
        return tgt, smile / 0.12, closes

    src_closed, eye_masks = build_closed_eyes(src, Ls)
    cv2.imwrite(os.path.splitext(args.out)[0] + "_closed_eyes_check.png", src_closed)

    def render(t):
        tgt, smile, closes = target(t)
        frame_src = src
        if max(closes) > 0.01:
            wmask = sum(np.clip(c * c * (3 - 2 * c), 0, 1) * m for c, m in zip(closes, eye_masks))
            frame_src = np.clip(src * (1 - wmask[..., None]) + src_closed * wmask[..., None], 0, 255).astype(np.uint8)
        if os.environ.get("PUPPET_DEBUG"):
            cv2.imwrite(f"{args.out}_dbgsrc_{t:05d}.png", frame_src)
            print("debug closes", t, closes)
        mw = np.linalg.norm(tgt[61] - tgt[291])
        if src_a is None:
            out = warp(frame_src, pts, tgt, tris_outer, tris_face)
            draw_mouth(out, tgt, mw)
            return out
        prem = np.dstack([frame_src.astype(np.float32) * src_a[..., None], src_a * 255.0])
        w4 = warp(prem, pts, tgt, tris_outer, tris_face)
        a = np.clip(w4[..., 3] / 255.0, 0, 1)
        rgb = np.clip(w4[..., :3], 0, 255).astype(np.uint8)
        draw_mouth(rgb, tgt, mw)
        f = rgb.astype(np.float32) * np.array([0.97, 1.0, 1.03], np.float32)   # slight warm grade to sit in the scene
        # light wrap: background light spills softly onto the subject's edges
        edge = np.clip(a - cv2.erode(a, np.ones((9, 9), np.uint8)), 0, 1)
        edge = cv2.GaussianBlur(edge, (0, 0), 3.0) * a
        out = bg.astype(np.float32) * (1 - a[..., None]) + f
        out = out + edge[..., None] * cv2.GaussianBlur(bg.astype(np.float32), (0, 0), 12) * 0.22
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)

    n = len(Ld)
    if args.frames:
        for t in [int(x) for x in args.frames.split(",")]:
            cv2.imwrite(f"{args.out}_{t:05d}.png", render(t))
            print("wrote", f"{args.out}_{t:05d}.png")
        return
    t0, t1 = args.start, min(n, args.end if args.end is not None else n)
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OW}x{OH}", "-r", f"{fps}",
           "-i", "-", "-ss", f"{t0 / fps:.4f}", "-t", f"{(t1 - t0) / fps:.4f}", "-i", args.video, "-map", "0:v", "-map", "1:a?",
           "-c:v", "libx264", "-crf", "16", "-preset", "medium",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", args.out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for t in range(t0, t1):
        p.stdin.write(render(t).tobytes())
        if t % 100 == 0:
            print(f"  frame {t}/{n}", flush=True)
    p.stdin.close(); p.wait()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
