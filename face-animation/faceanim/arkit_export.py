#!/usr/bin/env python3
"""Export calibrated ARKit-style blendshapes + head pose per frame for the 3D renderer."""
import json
import sys

import numpy as np

import faceanim as fa

RIG = ['browInnerUp', 'browDown_L', 'browDown_R', 'browOuterUp_L', 'browOuterUp_R', 'eyeLookUp_L', 'eyeLookUp_R',
       'eyeLookDown_L', 'eyeLookDown_R', 'eyeLookIn_L', 'eyeLookIn_R', 'eyeLookOut_L', 'eyeLookOut_R', 'eyeBlink_L',
       'eyeBlink_R', 'eyeSquint_L', 'eyeSquint_R', 'eyeWide_L', 'eyeWide_R', 'cheekPuff', 'cheekSquint_L',
       'cheekSquint_R', 'noseSneer_L', 'noseSneer_R', 'jawOpen', 'jawForward', 'jawLeft', 'jawRight', 'mouthFunnel',
       'mouthPucker', 'mouthLeft', 'mouthRight', 'mouthRollUpper', 'mouthRollLower', 'mouthShrugUpper',
       'mouthShrugLower', 'mouthClose', 'mouthSmile_L', 'mouthSmile_R', 'mouthFrown_L', 'mouthFrown_R',
       'mouthDimple_L', 'mouthDimple_R', 'mouthUpperUp_L', 'mouthUpperUp_R', 'mouthLowerDown_L', 'mouthLowerDown_R',
       'mouthPress_L', 'mouthPress_R', 'mouthStretch_L', 'mouthStretch_R', 'tongueOut']

# moderate gains: a realistic head looks wrong when pushed as hard as a cartoon
GAINS = {"jawOpen": 1.4, "mouthSmile": 1.25, "mouthFrown": 1.6, "mouthStretch": 1.5, "mouthPucker": 1.2,
         "mouthFunnel": 1.4, "mouthUpperUp": 1.3, "mouthLowerDown": 1.3, "mouthLeft": 1.5, "mouthRight": 1.5,
         "jawLeft": 1.5, "jawRight": 1.5, "mouthDimple": 1.3, "mouthPress": 1.2, "mouthShrug": 1.2,
         "browInnerUp": 1.6, "browDown": 1.4, "browOuterUp": 1.5, "eyeWide": 1.5, "eyeSquint": 1.1,
         "cheekSquint": 1.3, "cheekPuff": 1.6, "noseSneer": 1.5}


def gain(name):
    for k, g in GAINS.items():
        if name.startswith(k):
            return g
    return 1.0


def mp_to_rig(name):
    if name.endswith("Left"):
        return name[:-4] + "_L"
    if name.endswith("Right"):
        return name[:-5] + "_R"
    return name


def export(npz, video, out_json):
    data = np.load(npz)
    names = [str(s) for s in data["names"]]
    found = fa.clean_found(data["found"].astype(bool))
    fps = float(data["fps"])
    n = len(found)
    B = fa.fill_gaps_hold(data["blend"], found)
    left_is_screen_left = fa.lr_mapping(fa.fill_gaps(data["blend"], found), fa.fill_gaps(data["lms"], found), names, found)

    rig = {k: np.zeros(n, np.float32) for k in RIG}
    for i, nm in enumerate(names):
        r = mp_to_rig(nm)
        if r not in rig:
            continue
        if left_is_screen_left and (r.endswith("_L") or r.endswith("_R")):
            r = r[:-1] + ("R" if r.endswith("L") else "L")  # make _L mean the subject's left
        v = B[:, i]; fv = v[found]
        if nm.startswith("eyeLook"):
            x = np.clip((v - np.median(fv)) * 1.3, 0, 1)
        else:
            base = np.percentile(fv, 8)
            x = np.clip((v - base) / max(1 - base, 1e-3) * gain(nm), 0, 1)
        sig = 0.7 if nm.startswith("eyeBlink") else (1.0 if nm.startswith(("jaw", "mouth")) else 1.4)
        rig[r] = fa.gsmooth(x, sig)

    # geometry-based signals from the 2D pipeline (screen-left eye = subject's right = _R)
    E, pose, _, _ = fa.build_expressions(data)
    seg = fa.make_segmenter()
    if seg is not None and (~found).any():
        cov = fa.eye_coverage(seg, video, pose, found, fps)
        shut_raw = (cov > 0.12).astype(np.float32)
        # a long dropout with hands over the face most of the time: keep the eyes shut throughout
        i = 0
        while i < n:
            if not found[i]:
                j = i
                while j < n and not found[j]:
                    j += 1
                if j - i >= 15 and np.mean(cov[i:j] > 0.03) >= 0.3:
                    shut_raw[i:j] = 1.0
                i = j
            else:
                i += 1
        shut = fa.gsmooth(shut_raw, 2.5)
        for s in "LR":
            E["blink" + s] = np.maximum(E["blink" + s], np.clip(shut * 1.3, 0, 1))
    for scr, subj in (("L", "R"), ("R", "L")):
        rig["eyeBlink_" + subj] = np.clip(E["blink" + scr] * 0.98, 0, 1)
        rig["eyeWide_" + subj] = np.maximum(rig["eyeWide_" + subj], 0.8 * E["wide" + scr])
        rig["browOuterUp_" + subj] = np.maximum(rig["browOuterUp_" + subj], 0.75 * E["browOuter" + scr])
        rig["browDown_" + subj] = np.maximum(rig["browDown_" + subj], 0.7 * E["browDown" + scr])
        # squint with the smile, as the face really does
        rig["eyeSquint_" + subj] = np.maximum(rig["eyeSquint_" + subj], 0.5 * rig["mouthSmile_" + subj])
    rig["browInnerUp"] = np.maximum(rig["browInnerUp"], 0.75 * E["browInner"])
    rig["jawOpen"] = np.maximum(rig["jawOpen"], 0.55 * E["lipGap"])
    rig["mouthClose"] = rig["mouthClose"] * (1 - E["lipGap"])

    # head rotation from MediaPipe's transformation matrix (already matches three.js XYZ euler)
    M = fa.fill_gaps(data["mats"], found)
    R = M[:, :3, :3]
    ry = np.arcsin(np.clip(R[:, 0, 2], -1, 1))
    rx = np.arctan2(-R[:, 1, 2], R[:, 2, 2])
    rz = np.arctan2(-R[:, 0, 1], R[:, 0, 0])
    med = lambda a: float(np.median(a[found]))
    rx = fa.gsmooth(rx - 0.8 * med(rx), 1.8)
    ry = fa.gsmooth(ry - 0.6 * med(ry), 1.8)
    rz = fa.gsmooth(rz - med(rz), 1.8)
    t = M[:, :3, 3]
    tm = np.median(t[found], axis=0)
    pos = fa.gsmooth((t - tm) / 15.0, 2.0)  # in head-heights (~15 cm head)

    frames = np.stack([rig[k] for k in RIG], 1)
    out = {"fps": fps, "n": n, "names": RIG,
           "frames": np.round(frames, 4).tolist(),
           "rot": np.round(np.stack([rx, ry, rz], 1), 5).tolist(),
           "pos": np.round(pos, 5).tolist()}
    with open(out_json, "w") as f:
        json.dump(out, f)
    print("wrote", out_json, "frames", n, "left_is_screen_left", left_is_screen_left)


if __name__ == "__main__":
    export(sys.argv[1], sys.argv[2], sys.argv[3])
