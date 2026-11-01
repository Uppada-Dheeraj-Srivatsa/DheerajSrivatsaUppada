#!/usr/bin/env python3
"""Composite the rendered cartoon head (RGBA frames) onto the real video, Filmymoji-style:
real body and hands, cartoon head tracked to the real head, with hand occlusion and a soft contact shadow.

usage: moji_composite.py <track.npz> <video> <frames_dir> <out.mp4> [--scale 1.55] [--preview 0,10]
"""
import argparse
import json
import os
import subprocess

import cv2
import numpy as np

import faceanim as fa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz"); ap.add_argument("video"); ap.add_argument("frames"); ap.add_argument("out")
    ap.add_argument("--scale", type=float, default=1.5, help="cartoon head height / real forehead-to-chin")
    ap.add_argument("--preview", default=None)
    a = ap.parse_args()

    data = np.load(a.npz)
    found = fa.clean_found(data["found"].astype(bool))
    L = fa.gsmooth(fa.fill_gaps(data["lms"], found)[:, :, :2].astype(np.float32), 1.6)
    _, pose, _, _ = fa.build_expressions(data)
    fps = float(data["fps"]); W, H = int(data["W"]), int(data["H"])
    top, chin, lft, rgt = L[:, 10], L[:, 152], L[:, 234], L[:, 454]
    face_h = fa.gsmooth(np.linalg.norm(chin - top, axis=1), 2.0)
    cx = 0.5 * (lft[:, 0] + rgt[:, 0])
    cy = 0.5 * (top[:, 1] + chin[:, 1])

    info = json.load(open(os.path.join(a.frames, "info.json")))
    rc = np.array(info["centre"], np.float32)
    r_h = info["chin"][1] - info["top"][1]          # rendered head height (hair top to chin) in px

    seg = fa.make_segmenter()
    cap = cv2.VideoCapture(a.video)
    n = len(found)
    want = set(int(x) for x in a.preview.split(",")) if a.preview else None
    writer = None
    if want is None:
        writer = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{fps}",
                                   "-i", "-", "-i", a.video, "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", "17",
                                   "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest",
                                   "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        if want is not None and i not in want:
            continue
        p = {k: (v[i] if np.ndim(v) else v) for k, v in pose.items()}
        orig = frame.copy()
        head = cv2.imread(os.path.join(a.frames, f"f_{i:05d}.png"), cv2.IMREAD_UNCHANGED)
        s = a.scale * float(face_h[i]) / r_h
        tx = float(cx[i]) - s * rc[0]
        ty = float(cy[i]) - 0.19 * float(face_h[i]) - s * rc[1]
        M = np.array([[s, 0, tx], [0, s, ty]], np.float32)
        warped = cv2.warpAffine(head, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        alpha = warped[..., 3:4].astype(np.float32) / 255.0
        rgb = warped[..., :3].astype(np.float32)
        f = frame.astype(np.float32)
        # soft contact shadow under the head (onto neck / shoulders)
        sh = cv2.GaussianBlur(alpha[..., 0], (0, 0), max(3.0, 0.06 * float(face_h[i])))
        shift = int(0.08 * float(face_h[i]))
        sh = np.roll(sh, shift, axis=0); sh[:shift] = 0
        f *= (1 - 0.35 * sh * (1 - alpha[..., 0]))[..., None]
        f = f * (1 - alpha) + rgb * alpha
        # real hands stay in front of the cartoon head
        if seg is not None:
            hm = fa.hand_mask(seg, orig, i, fps, p)
            if hm is not None:
                f = f * (1 - hm[..., None]) + orig.astype(np.float32) * hm[..., None]
        out = np.clip(f + 0.5, 0, 255).astype(np.uint8)
        if writer is not None:
            writer.stdin.write(out.tobytes())
            if i % 100 == 0:
                print(f"  composited {i}/{n}", flush=True)
        else:
            cv2.imwrite(f"{os.path.splitext(a.out)[0]}_{i:05d}.png", out)
            print("wrote", i)
    cap.release()
    if writer is not None:
        writer.stdin.close(); writer.wait()
        print("wrote", a.out)


if __name__ == "__main__":
    main()
