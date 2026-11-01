#!/usr/bin/env python3
"""Build a talking performance from a voice recording, using one tracked video only as a library of face shapes.

The mouth opens with the voice (loudness envelope, leading the sound slightly), and the open-mouth shape is the
average of the person's own widest mouth shapes in the tracked video. Blinks come at natural, varied intervals,
eyebrows lift a little at the start of some phrases, and the head moves gently with small nods on stressed syllables.
Writes an .npz in the same format as `faceanim.py track`, so every renderer (puppet.py, arkit_export.py) can use it.

usage: audio_driver.py track.npz voice_audio out.npz [--seed 3]
"""
import argparse
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import faceanim as fa  # noqa: E402
from puppet import STABLE, umeyama3, EYE_R, EYE_L, IRIS_R, IRIS_L, BROWS, LID_PAIRS  # noqa: E402


def voice_envelope(path, fps, n_hint=None):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    sr = 16000
    n = int(np.ceil(len(x) / sr * fps))
    hop = sr / fps
    win = int(0.05 * sr)
    db = np.empty(n, np.float32)
    for i in range(n):
        c = int((i + 0.5) * hop)
        seg = x[max(0, c - win // 2): c + win // 2]
        db[i] = 10 * np.log10(np.mean(seg * seg) + 1e-10) if len(seg) else -100
    floor = np.percentile(db, 10)
    top = np.percentile(db, 96)
    a = np.clip((db - (floor + 7)) / max(top - (floor + 7), 1), 0, 1) ** 0.85
    # asymmetric follower: mouths open fast and close a little slower
    env = np.zeros_like(a)
    for i in range(n):
        prev = env[i - 1] if i else 0
        k = 0.75 if a[i] > prev else 0.5
        env[i] = prev + k * (a[i] - prev)
    env = fa.gsmooth(env, 0.6)
    env = np.concatenate([env[1:], env[-1:]])   # lips lead the sound by about one frame
    speaking = fa.gsmooth((a > 0.08).astype(np.float32), 3.0) > 0.5
    return env.astype(np.float32), speaking, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz"); ap.add_argument("voice"); ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--open", type=float, default=0.75, help="mouth opening at full voice, relative to the widest tracked")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    d = np.load(args.npz)
    fps = float(d["fps"])
    found = fa.clean_found(d["found"].astype(bool))
    L = fa.fill_gaps_hold(d["lms"], found).astype(np.float64)
    B = d["blend"].astype(np.float64)
    names = [str(s) for s in d["names"]]
    bi = {nm: i for i, nm in enumerate(names)}

    # neutral reference: calm, mouth closed, eyes open
    # (same choice as puppet.py makes for a tracked video, so wide eyes or raised brows never become "neutral")
    lg = np.linalg.norm(L[:, 13] - L[:, 14], axis=1) / np.linalg.norm(L[:, 61] - L[:, 291], axis=1)
    E, _, _, _ = fa.build_expressions(d)
    calm = (E["blinkL"] + E["blinkR"] + 2 * E["lipGap"] + E["smileL"] + E["smileR"] + E["wideL"] + E["wideR"]
            + E["browInner"] + 0.5 * (E["browOuterL"] + E["browOuterR"]))
    calm = np.where(found, fa.gsmooth(calm, 3.0), 1e9)
    ref = int(np.argmin(calm))
    R3 = L[ref]

    def frontal(P3):
        sc, Rm, tr = umeyama3(R3[STABLE], P3[STABLE])
        return ((P3 - tr) @ Rm) / sc

    Pn = np.stack([frontal(L[t]) if found[t] else R3 for t in range(len(L))])
    D = Pn - R3[None]
    mw = np.linalg.norm(R3[61] - R3[291])

    # region weights on the reference face
    y = R3[:, 1]
    y_nose, y_lip = R3[2, 1], R3[13, 1]
    w_low = np.clip((y - y_nose) / max(y_lip - y_nose, 1), 0, 1)                      # lower face
    eye_c = [R3[EYE_R].mean(0), R3[EYE_L].mean(0)]
    ew = np.linalg.norm(R3[33] - R3[133])
    w_eye = np.clip(1 - (np.minimum(np.linalg.norm(R3 - eye_c[0], axis=1), np.linalg.norm(R3 - eye_c[1], axis=1))
                         - 0.75 * ew) / (0.5 * ew), 0, 1)
    w_brow = np.clip((R3[EYE_R + EYE_L, 1].min() - y) / (0.6 * ew), 0, 1)                # above the eyes

    def pick(score, k=1, sep=6):
        order = np.argsort(-np.where(found, score, -1e9))
        out = []
        for t in order:
            if all(abs(t - s) >= sep for s in out):
                out.append(int(t))
            if len(out) == k:
                break
        return out

    # open-mouth shape per unit of lip gap: average of the widest real mouth shapes
    gap = lg * 1.0
    opens = pick(gap, k=6)
    unit = np.mean([D[t] / max(np.linalg.norm(Pn[t, 13] - Pn[t, 14]) - np.linalg.norm(R3[13] - R3[14]), 1e-3)
                    for t in opens], 0) * w_low[:, None]
    gmax = max(np.linalg.norm(Pn[t, 13] - Pn[t, 14]) - np.linalg.norm(R3[13] - R3[14]) for t in opens)
    blink_f = pick(B[:, bi["eyeBlinkLeft"]] + B[:, bi["eyeBlinkRight"]] - 2 * (B[:, bi["mouthSmileLeft"]] + B[:, bi["mouthSmileRight"]]))[0]
    brow_f = pick(B[:, bi["browInnerUp"]] - 2 * (B[:, bi["eyeBlinkLeft"]] + B[:, bi["eyeBlinkRight"]]))[0]
    d_blink = D[blink_f] * w_eye[:, None]
    d_brow = D[brow_f] * w_brow[:, None]
    print(f"reference frame {ref}; open shapes {opens}; blink {blink_f}; brow {brow_f}; widest gap {gmax:.2f}")

    env, speaking, n = voice_envelope(args.voice, fps)

    # blinks: every 2.2 to 5 s, a bit more likely just after a phrase ends
    blink = np.zeros(n)
    shape = np.array([0.35, 0.8, 1.0, 1.0, 0.75, 0.45, 0.2, 0.07])
    t = int(fps * rng.uniform(0.8, 2.0))
    while t < n - len(shape):
        blink[t:t + len(shape)] = np.maximum(blink[t:t + len(shape)], shape)
        nxt = t + int(fps * rng.uniform(2.6, 5.2))
        ends = [i for i in range(t + int(2.2 * fps), min(nxt, n - 1)) if speaking[i - 1] and not speaking[i]]
        t = (ends[0] + 2) if (ends and rng.random() < 0.5) else nxt
    # eyebrow lifts at the start of some phrases
    brow = np.zeros(n)
    # a phrase start = speech after a real pause (>= 0.3 s); lifts at most every ~2.5 s
    starts, last = [], -10 ** 9
    for i in range(int(0.3 * fps), n):
        if speaking[i] and not speaking[i - int(0.3 * fps):i].any() and i - last > 2.5 * fps:
            starts.append(i); last = i
    for s in starts:
        if rng.random() < 0.45:
            amp = rng.uniform(0.25, 0.5)
            hold = int(fps * rng.uniform(0.3, 0.8))
            env_b = np.concatenate([np.linspace(0, 1, 5), np.ones(hold), np.linspace(1, 0, 10)]) * amp
            e = min(n, s + len(env_b))
            brow[s:e] = np.maximum(brow[s:e], env_b[:e - s])
    # head: slow drift (smoothed noise) + small nods that follow the voice's stress
    def drift(sig_s, amp):
        z = fa.gsmooth(rng.normal(0, 1, n).astype(np.float32), sig_s * fps)
        return z / (z.std() + 1e-6) * amp
    stress = np.clip(env - fa.gsmooth(env, 8.0), 0, None)
    pitch = drift(1.2, 0.022) + 0.09 * fa.gsmooth(stress, 2.0)
    yaw = drift(1.8, 0.03)
    roll = drift(2.2, 0.014)
    tx = drift(2.0, 0.012 * mw * 3); ty = drift(2.0, 0.008 * mw * 3)

    c = R3[STABLE].mean(0)
    lms = np.zeros((n, 478, 3), np.float32)
    blend = np.zeros((n, len(names)), np.float32)
    b_ref = B[ref]
    jaw_i = bi["jawOpen"]
    for t in range(n):
        g = env[t] * args.open * gmax
        P = R3 + g * unit + brow[t] * d_brow
        if blink[t] > 0:                       # symmetric blink: each upper-lid point slides onto its lower-lid partner
            k = 0.94 * blink[t]
            for u, l in LID_PAIRS:
                P[u] = P[u] + k * (P[l] - P[u]) * np.array([1.0, 1.0, 0.0])
        rv = np.array([pitch[t], yaw[t], roll[t]], np.float64)
        Rm = cv2.Rodrigues(rv)[0]
        lms[t] = (P - c) @ Rm.T + c + np.array([tx[t], ty[t], 0])
        blend[t] = b_ref + (B[opens[0]] - b_ref) * (g / max(gmax, 1e-6)) + (B[blink_f] - b_ref) * blink[t] \
            + (B[brow_f] - b_ref) * brow[t]
        blend[t, jaw_i] = b_ref[jaw_i] + (B[opens[0], jaw_i] - b_ref[jaw_i]) * (g / max(gmax, 1e-6))
    mats = np.repeat(d["mats"][ref:ref + 1], n, 0)
    np.savez_compressed(args.out, lms=lms, blend=np.clip(blend, 0, 1), mats=mats, found=np.ones(n, bool),
                        names=d["names"], fps=d["fps"], W=d["W"], H=d["H"])
    print(f"wrote {args.out}: {n} frames ({n / fps:.1f} s), {int((np.diff((blink > 0.5).astype(int)) > 0).sum())} blinks, "
          f"{int((np.diff((brow > 0.05).astype(int)) > 0).sum())} brow lifts, speaking {speaking.mean() * 100:.0f}% of the time")


if __name__ == "__main__":
    main()
