#!/usr/bin/env python3
"""Cut a person out of an image whose 'transparent' checkerboard was baked into the pixels.
Hair uses a difference matte against the reconstructed checker; the rest uses segmentation.
usage: matte_checker.py in.png out_rgba.png"""
import sys

import cv2
import numpy as np


def segment(img):
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    import mediapipe as mp
    import os
    model = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "selfie_multiclass_256x256.tflite")
    opts = vision.ImageSegmenterOptions(base_options=mp_python.BaseOptions(model_asset_path=model),
                                        output_confidence_masks=True, output_category_mask=True)
    seg = vision.ImageSegmenter.create_from_options(opts)
    res = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    conf = [np.array(m.numpy_view(), np.float32).reshape(img.shape[:2]) for m in res.confidence_masks]
    return conf  # 0 bg, 1 hair, 2 body skin, 3 face skin, 4 clothes, 5 others


def checker_background(img, bgmask):
    """Reconstruct the checker colour at every pixel from the confident background pixels."""
    H, W = img.shape[:2]
    gray = img.mean(2)
    # estimate cell size and phase from a background row and column
    def period_phase(profile):
        e = np.where(np.abs(np.diff(profile)) > 8)[0].astype(np.float32) + 0.5
        e = e[np.r_[True, np.diff(e) > 3]]
        d = np.diff(e); c0 = float(np.median(d[(d > 8) & (d < 40)]))
        k = np.round((e - e[0]) / c0)
        A = np.stack([k, np.ones_like(k)], 1)
        c, ph = np.linalg.lstsq(A, e, rcond=None)[0]
        return float(c), float(ph % c)
    # the top-left corner is pure background in these portraits
    cx, phx = period_phase(gray[8, :260]); cy, phy = period_phase(gray[:400, 8])
    print(f'  period x {cx:.3f} phase {phx:.2f} | period y {cy:.3f} phase {phy:.2f}')
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    par = ((np.floor((xx - phx) / cx) + np.floor((yy - phy) / cy)) % 2).astype(np.int32)
    c = 0.5 * (cx + cy)
    out = np.zeros_like(img, np.float32)
    for k in (0, 1):
        sel = (par == k) & (bgmask > 0.5)
        col = np.median(img[sel], axis=0)
        out[par == k] = col
    # verify parity assignment matches observed background
    err = np.abs(img.astype(np.float32) - out).mean(2)[bgmask > 0.5].mean()
    return out, c, err


def main(src, dst):
    img = cv2.imread(src)
    H, W = img.shape[:2]
    conf = segment(img)
    p_bg = cv2.GaussianBlur(conf[0], (0, 0), 1.0)
    hair = conf[1]
    bgsure = (p_bg > 0.9).astype(np.float32)
    I = img.astype(np.float32)
    lum = I.mean(2)
    # local average background colour (both checker greys blended), used for edge decontamination
    wsum = cv2.GaussianBlur(bgsure, (0, 0), 25)
    B = cv2.GaussianBlur(I * bgsure[..., None], (0, 0), 25) / np.maximum(wsum[..., None], 1e-3)
    B = np.where(wsum[..., None] > 1e-2, B, 238.0)
    # luminance key: dark hair against light checker (both greys are >= ~222)
    a_diff = np.clip((219.0 - lum) / (219.0 - 70.0), 0, 1)
    print("luma key ready")
    # segmentation alpha, sharpened
    a_seg = np.clip((1 - p_bg - 0.35) / 0.3, 0, 1)
    # hair zone: hair class, grown to cover flyaway strands
    hz = cv2.dilate((hair > 0.3).astype(np.uint8), np.ones((31, 31), np.uint8)).astype(np.float32)
    hz = cv2.GaussianBlur(hz, (0, 0), 6)
    core = cv2.erode((a_seg > 0.99).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(np.float32)
    non_hair_core = core * (1 - (hair > 0.5))
    nonhair = np.clip(conf[2] + conf[3] + conf[4], 0, 1)
    a_nonhair = np.clip((nonhair - 0.3) / 0.4, 0, 1)
    # the luma key is only trusted next to visible checker: deep inside dense hair, light highlights stay solid
    # (otherwise a green or grey background shows through the shine). Checker seen through gaps is bright and
    # near-neutral (luma >= ~205, slightly tinted by hair), while shine inside dark hair stays well below 150.
    sat = I.max(2) - I.min(2)
    ck = ((lum >= 196) & (sat <= 24)).astype(np.uint8)
    dist_ck = cv2.distanceTransform(1 - ck, cv2.DIST_L2, 5)
    near = 1 - np.clip((dist_ck - 3) / 12, 0, 1)
    a_hair = a_diff * near + (1 - near)
    alpha = a_seg * (1 - hz) + np.maximum(a_hair, a_nonhair * a_seg) * hz
    alpha = np.maximum(alpha, non_hair_core)                     # face/clothes interior always solid
    alpha = np.where(bgsure > 0, np.minimum(alpha, a_diff), alpha)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.6)
    alpha = np.clip(alpha, 0, 1)
    # colour decontamination: remove the checker contribution from semi-transparent edges
    a3 = np.maximum(alpha[..., None], 1e-3)
    F = (I - (1 - a3) * B) / a3
    F = np.where(alpha[..., None] > 0.02, np.clip(F, 0, 255), I)
    # in mostly transparent hair edges, pull colour toward local hair colour to avoid grey fringes
    dark_hair = cv2.GaussianBlur(np.where(alpha[..., None] > 0.9, I, 0), (0, 0), 6) / np.maximum(cv2.GaussianBlur((alpha > 0.9).astype(np.float32), (0, 0), 6)[..., None], 1e-3)
    edge = ((alpha > 0.02) & (alpha < 0.6))[..., None] * hz[..., None]
    F = F * (1 - edge * 0.7) + dark_hair * edge * 0.7
    out = np.dstack([np.clip(F, 0, 255).astype(np.uint8), (alpha * 255).astype(np.uint8)])
    cv2.imwrite(dst, out)
    # preview on two backgrounds
    for name, col in (("dark", (40, 30, 30)), ("light", (180, 210, 240))):
        bgc = np.full_like(I, col)
        comp = F * alpha[..., None] + bgc * (1 - alpha[..., None])
        cv2.imwrite(dst.replace(".png", f"_prev_{name}.jpg"), np.clip(comp, 0, 255).astype(np.uint8))
    print("wrote", dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
