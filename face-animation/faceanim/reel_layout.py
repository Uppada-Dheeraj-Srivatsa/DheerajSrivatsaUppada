#!/usr/bin/env python3
"""Vertical reel layout (1080x1920): a post/carousel card on top, the animated presenter below.

Writes OUT_DIR/layout_bg.png (backdrop + card) and OUT_DIR/char_canvas.png (the presenter cut-out placed for
puppet.py --size 1080x1920 --bg OUT_DIR/layout_bg.png).

usage: reel_layout.py card.png character_rgba_1080x1350.png OUT_DIR [--card-scale 0.80] [--char-scale 0.63]
"""
import argparse
import os

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("card"); ap.add_argument("character"); ap.add_argument("out_dir")
    ap.add_argument("--card-scale", type=float, default=0.80)
    ap.add_argument("--char-scale", type=float, default=0.63)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    W, H = 1080, 1920
    card = cv2.imread(args.card)
    card = cv2.resize(card, (1080, int(round(card.shape[0] * 1080 / card.shape[1]))), interpolation=cv2.INTER_AREA)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # backdrop from the card's own palette (top-left and bottom-left corners)
    top = card[20:120, 20:200].reshape(-1, 3).mean(0)
    bot = card[-100:-10, 20:200].reshape(-1, 3).mean(0)
    t = (yy / H)[..., None]
    bg = top * (1 - t) + bot * t
    g = np.exp(-(((xx - W / 2) / 420) ** 2 + ((yy - 1430) / 330) ** 2))[..., None]
    bg = bg + g * np.array([30, 52, 92], np.float32) * 0.55          # warm glow behind the presenter
    v = np.sqrt(((xx - W / 2) / (W * 0.8)) ** 2 + ((yy - H * 0.5) / (H * 0.7)) ** 2)
    bg *= (1 - 0.35 * np.clip(v - 0.4, 0, 1))[..., None]
    bg += np.random.default_rng(5).normal(0, 1.5, (H, W, 1)).astype(np.float32)
    # card with rounded corners, soft drop shadow and a faint edge
    cw, ch = int(round(card.shape[1] * args.card_scale)), int(round(card.shape[0] * args.card_scale))
    cs = cv2.resize(card, (cw, ch), interpolation=cv2.INTER_AREA).astype(np.float32)
    x0, y0, r = (W - cw) // 2, 40, 26
    mask = np.zeros((ch, cw), np.uint8)
    cv2.rectangle(mask, (r, 0), (cw - r, ch), 255, -1); cv2.rectangle(mask, (0, r), (cw, ch - r), 255, -1)
    for cx, cy in ((r, r), (cw - r, r), (r, ch - r), (cw - r, ch - r)):
        cv2.circle(mask, (cx, cy), r, 255, -1, cv2.LINE_AA)
    m = cv2.GaussianBlur(mask, (0, 0), 0.8).astype(np.float32) / 255
    sh = np.zeros((H, W), np.float32)
    sh[y0 + 14:y0 + ch + 14, x0:x0 + cw] = m[:max(0, min(ch, H - y0 - 14))]
    bg *= (1 - 0.55 * cv2.GaussianBlur(sh, (0, 0), 18))[..., None]
    roi = bg[y0:y0 + ch, x0:x0 + cw]
    bg[y0:y0 + ch, x0:x0 + cw] = roi * (1 - m[..., None]) + cs * m[..., None]
    edge = cv2.morphologyEx((mask > 127).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    bg[y0:y0 + ch, x0:x0 + cw] += edge[..., None].astype(np.float32) * 28
    cv2.imwrite(os.path.join(args.out_dir, "layout_bg.png"), np.clip(bg, 0, 255).astype(np.uint8))
    # presenter: scaled cut-out, centred, shoulders on the bottom edge
    c = cv2.imread(args.character, cv2.IMREAD_UNCHANGED)
    pw, ph = int(round(c.shape[1] * args.char_scale)), int(round(c.shape[0] * args.char_scale))
    cr = cv2.resize(c, (pw, ph), interpolation=cv2.INTER_AREA)
    can = np.zeros((H, W, 4), np.uint8)
    px, py = (W - pw) // 2, H - ph
    can[py:py + ph, px:px + pw] = cr
    cv2.imwrite(os.path.join(args.out_dir, "char_canvas.png"), can)
    print(f"card at y {y0}..{y0 + ch}; presenter at y {py}..{H}")


if __name__ == "__main__":
    main()
