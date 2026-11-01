#!/usr/bin/env python3
"""Convert calibrated rig data (arkit.json) into VTuber-style VRM controls:
stylized, quantized mouth shapes held on twos, anime emotion presets, blinks, gaze and head pose."""
import json
import sys

import numpy as np

import faceanim as fa


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def main(src_json, out_json, hold=2):
    d = json.load(open(src_json))
    names = d["names"]
    F = np.array(d["frames"], np.float32)
    g = lambda k: F[:, names.index(k)]
    n = len(F)

    smile = 0.5 * (g("mouthSmile_L") + g("mouthSmile_R"))
    frown = 0.5 * (g("mouthFrown_L") + g("mouthFrown_R"))
    wide = 0.5 * (g("eyeWide_L") + g("eyeWide_R"))
    brow_dn = 0.5 * (g("browDown_L") + g("browDown_R"))
    brow_in = g("browInnerUp")
    brow_out = 0.5 * (g("browOuterUp_L") + g("browOuterUp_R"))
    jaw = g("jawOpen")
    rnd = np.maximum(g("mouthFunnel"), 0.8 * g("mouthPucker"))
    stretch = 0.5 * (g("mouthStretch_L") + g("mouthStretch_R"))

    # ---- emotions (anime presets), smoothed so they ease in and out
    happy = smoothstep(0.25, 0.75, smile)
    surprised = smoothstep(0.25, 0.8, 0.6 * wide + 0.4 * np.maximum(brow_in, brow_out)) * (1 - happy)
    angry = smoothstep(0.35, 0.8, brow_dn - 0.5 * smile) * (1 - surprised)
    sad = smoothstep(0.35, 0.8, 0.6 * brow_in + 0.6 * frown - 0.5 * smile) * (1 - surprised) * (1 - angry)
    emo = {k: fa.gsmooth(v.astype(np.float32), 2.5) for k, v in
           dict(happy=happy, surprised=surprised, angry=angry, sad=sad).items()}

    # ---- mouth: a few discrete shapes, updated every `hold` frames like hand-drawn animation
    openness = np.clip(jaw / 0.42, 0, 1)
    vis = np.zeros((n, 5), np.float32)  # aa ih ou ee oh
    state = 0
    shape = "aa"
    for t in range(n):
        if t % hold == 0:
            o = openness[t]
            # hysteresis between levels
            levels = [0.12, 0.38, 0.68]
            lvl = sum(o > (L - 0.04 if i < state else L + 0.04) for i, L in enumerate(levels))
            state = lvl
            if state > 0:
                if rnd[t] > 0.35:
                    shape = "oh" if state >= 2 else "ou"
                elif smile[t] + stretch[t] > 0.45:
                    shape = "ih" if state >= 2 else "ee"
                else:
                    shape = "aa"
        amount = [0.0, 0.35, 0.65, 1.0][state]
        j = ["aa", "ih", "ou", "ee", "oh"].index(shape)
        vis[t, j] = amount
    # big smiles: keep the mouth in the preset smile, small flaps only
    vis *= (1 - 0.5 * emo["happy"])[:, None]

    blinkL = g("eyeBlink_L"); blinkR = g("eyeBlink_R")
    # anime blinks are snappy: push partial closures toward open or closed
    snap = lambda b: smoothstep(0.25, 0.7, b)
    blinkL, blinkR = snap(blinkL), snap(blinkR)
    blinkL = blinkL * (1 - emo["happy"]); blinkR = blinkR * (1 - emo["happy"])

    # gaze (degrees): subject's look values -> avatar yaw/pitch
    look_out = g("eyeLookOut_L") - g("eyeLookIn_L")      # left eye looking out = toward subject's left
    look_in = g("eyeLookIn_R") - g("eyeLookOut_R")       # right eye looking in = toward subject's left
    yaw = fa.gsmooth(((look_out + look_in) * 0.5 * 25).astype(np.float32), 1.5)
    pitch = fa.gsmooth(((0.5 * (g("eyeLookUp_L") + g("eyeLookUp_R")) - 0.5 * (g("eyeLookDown_L") + g("eyeLookDown_R"))) * 18).astype(np.float32), 1.5)

    out = {"fps": d["fps"], "n": n, "rot": d["rot"], "pos": d["pos"],
           "expr": {
               "aa": vis[:, 0], "ih": vis[:, 1], "ou": vis[:, 2], "ee": vis[:, 3], "oh": vis[:, 4],
               "blinkLeft": blinkL, "blinkRight": blinkR,
               "happy": emo["happy"] * 0.85, "surprised": emo["surprised"] * 0.9,
               "angry": emo["angry"] * 0.7, "sad": emo["sad"] * 0.7,
           },
           "gaze": np.stack([yaw, pitch], 1)}
    out["expr"] = {k: np.round(np.asarray(v, np.float32), 3).tolist() for k, v in out["expr"].items()}
    out["gaze"] = np.round(out["gaze"], 2).tolist()
    json.dump(out, open(out_json, "w"))
    for k, v in out["expr"].items():
        a = np.array(v); print(f"{k:11s} max {a.max():.2f} frames>0.3: {(a > 0.3).sum()}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
