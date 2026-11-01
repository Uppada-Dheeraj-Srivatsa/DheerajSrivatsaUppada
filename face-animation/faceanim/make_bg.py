#!/usr/bin/env python3
"""Warm festive bokeh background (out-of-focus diya lights), e.g. 1080x1350.
usage: make_bg.py out.png [WxH]"""
import sys

import cv2
import numpy as np


def main(out, size="1080x1350"):
    W, H = map(int, size.split("x"))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    t = yy / H
    top = np.array([28, 18, 52], np.float32)      # BGR deep maroon
    mid = np.array([30, 52, 120], np.float32)     # warm brown-orange
    bot = np.array([40, 92, 170], np.float32)     # amber
    col = np.where(t[..., None] < 0.55, top + (mid - top) * (t[..., None] / 0.55), mid + (bot - mid) * ((t[..., None] - 0.55) / 0.45))
    img = col.copy()
    rng = np.random.default_rng(21)
    # bokeh discs: soft, warm, larger and dimmer toward the back
    layer = np.zeros((H, W, 3), np.float32)
    for _ in range(46):
        x, y = rng.uniform(0, W), rng.uniform(0, H * 0.85)
        r = rng.uniform(18, 75)
        hue = rng.choice([0, 1, 2], p=[0.55, 0.3, 0.15])
        c = [np.array([90, 190, 255]), np.array([60, 140, 255]), np.array([170, 220, 255])][hue].astype(np.float32)
        a = rng.uniform(0.10, 0.32)
        d = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        disc = np.clip((r - d) / 2.5, 0, 1) * (0.75 + 0.25 * np.clip(d / r, 0, 1))  # slightly brighter rim
        layer += disc[..., None] * c * a
    layer = cv2.GaussianBlur(layer, (0, 0), 2.0)
    img = img + layer
    # warm glow behind where the head will be
    g = np.exp(-(((xx - W * 0.5) / (W * 0.42)) ** 2 + ((yy - H * 0.33) / (H * 0.30)) ** 2))
    img += g[..., None] * np.array([40, 95, 150], np.float32) * 0.55
    # vignette
    v = np.sqrt(((xx - W / 2) / (W * 0.75)) ** 2 + ((yy - H * 0.45) / (H * 0.75)) ** 2)
    img *= (1 - 0.45 * np.clip(v - 0.35, 0, 1))[..., None]
    # fine film grain
    img += rng.normal(0, 2.0, (H, W, 1)).astype(np.float32)
    cv2.imwrite(out, np.clip(img, 0, 255).astype(np.uint8))
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
