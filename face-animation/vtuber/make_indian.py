#!/usr/bin/env python3
"""Build 'Indian guy' textures for Seed-san: warm brown skin, short beard + mustache painted into the face
texture using the real face geometry, dark brown irises."""
import json
import os
import struct
import sys

import cv2
import numpy as np

VRM = "models/seed-san.vrm"
OUT = "indian"
os.makedirs(OUT, exist_ok=True)
SKIN = np.array([184, 128, 92], np.float32)   # target skin tone (RGB), warm medium brown
BEARD = np.array([24, 17, 14], np.float32)    # beard colour (RGB)

b = open(VRM, "rb").read()
jl = struct.unpack("<I", b[12:16])[0]
J = json.loads(b[20:20 + jl])
off = 20 + jl
BIN = b[off + 8: off + 8 + struct.unpack("<I", b[off:off + 4])[0]]

CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def _view(bvi, ctype, n, c, extra_off=0):
    bv = J["bufferViews"][bvi]
    dt = np.dtype(CT[ctype])
    start = bv.get("byteOffset", 0) + extra_off
    stride = bv.get("byteStride", dt.itemsize * c)
    raw = np.frombuffer(BIN, np.uint8, count=stride * (n - 1) + dt.itemsize * c, offset=start)
    return np.lib.stride_tricks.as_strided(raw, (n, c * dt.itemsize), (stride, 1)).copy().view(dt).reshape(n, c)


def acc(i):
    a = J["accessors"][i]
    if "bufferView" not in a:  # sparse accessor: zeros + sparse values
        n, c = a["count"], NC[a["type"]]
        arr = np.zeros((n, c), np.float32)
        sp = a.get("sparse")
        if sp:
            ii = _view(sp["indices"]["bufferView"], sp["indices"]["componentType"], sp["count"], 1, sp["indices"].get("byteOffset", 0)).ravel()
            vv = _view(sp["values"]["bufferView"], a["componentType"], sp["count"], c, sp["values"].get("byteOffset", 0))
            arr[ii.astype(np.int64)] = vv
        return arr
    bv = J["bufferViews"][a["bufferView"]]
    dt = np.dtype(CT[a["componentType"]])
    n, c = a["count"], NC[a["type"]]
    start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = bv.get("byteStride", dt.itemsize * c)
    raw = np.frombuffer(BIN, np.uint8, count=stride * (n - 1) + dt.itemsize * c, offset=start)
    arr = np.lib.stride_tricks.as_strided(raw, (n, c * dt.itemsize), (stride, 1)).copy().view(dt).reshape(n, c)
    if a.get("normalized"):
        arr = arr.astype(np.float32) / np.iinfo(dt).max
    return arr.astype(np.float32) if dt != np.float32 else arr


def tex_image(ti):
    im = J["images"][J["textures"][ti]["source"]]
    bv = J["bufferViews"][im["bufferView"]]
    data = BIN[bv.get("byteOffset", 0): bv.get("byteOffset", 0) + bv["byteLength"]]
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)


mats = J["materials"]
mi_body = [i for i, m in enumerate(mats) if m["name"] == "body_bake"][0]
mi_eye = [i for i, m in enumerate(mats) if m["name"] == "eye"][0]
head_mesh_i = [i for i, m in enumerate(J["meshes"]) if m.get("name") == "head"][0]
head_node_i = [i for i, n in enumerate(J["nodes"]) if n.get("mesh") == head_mesh_i][0]
prim = [p for p in J["meshes"][head_mesh_i]["primitives"] if p["material"] == mi_body][0]
P = acc(prim["attributes"]["POSITION"])
N = acc(prim["attributes"]["NORMAL"])
UV = acc(prim["attributes"]["TEXCOORD_0"])
IDX = acc(prim["indices"]).astype(np.int64).reshape(-1, 3)

# mouth vertices from the 'aa' expression's morph target
vrm = J["extensions"]["VRMC_vrm"]
aa_binds = vrm["expressions"]["preset"]["aa"]["morphTargetBinds"]
aa_idx = [bd["index"] for bd in aa_binds if bd["node"] == head_node_i][0]
D = acc(prim["targets"][aa_idx]["POSITION"])
disp = np.linalg.norm(D, axis=1)
mouth = disp > disp.max() * 0.35
mc = P[mouth].mean(0)
mouth_hw = (P[mouth, 0].max() - P[mouth, 0].min()) / 2
front = (N[:, 2] > 0.35)
centre = front & (np.abs(P[:, 0]) < 0.006)
nose = P[centre][np.argmax(P[centre, 2])]
below = centre & (P[:, 1] < mc[1])
chin = P[below][np.argmin(P[below, 1])]
face_hw = np.percentile(np.abs(P[front & (np.abs(P[:, 1] - mc[1]) < 0.03), 0]), 98)
d = nose[1] - mc[1]          # nose-to-mouth height
print("mouth centre", mc.round(4), "mouth half width", round(float(mouth_hw), 4))
print("nose", nose.round(4), "chin", chin.round(4), "face half width", round(float(face_hw), 4), "d", round(float(d), 4))


def ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


x, y, z = P[:, 0] - mc[0], P[:, 1] - mc[1], P[:, 2]
ax = np.abs(x)
soft = 0.0035
# beard line: at the mouth corners it sits a bit above the mouth, rising toward the sideburns
line = -0.18 * d + (ax / face_hw) ** 2.0 * (0.85 * d)
beard = ss(line + soft, line - soft, y)
# lower limit: just under the jaw / chin, a little onto the neck at the front
beard *= ss(chin[1] - mc[1] - 0.016, chin[1] - mc[1] - 0.006, y)
# only the front and sides of the face (not the back of the head)
beard *= ss(-0.35, 0.05, N[:, 2]) * ss(nose[2] - 0.11, nose[2] - 0.08, z)
# mustache band above the upper lip
# mustache: a clean band above the upper lip whose lower edge follows the lip and tapers at the ends
taper = (ax / (mouth_hw * 1.3)) ** 2
must = ss(mouth_hw * 1.5, mouth_hw * 1.25, ax) * ss(0.25 * d + 0.02 * d * taper, 0.33 * d + 0.02 * d * taper, y) \
    * ss(0.62 * d - 0.16 * d * taper, 0.52 * d - 0.16 * d * taper, y)
must *= ss(-0.2, 0.2, N[:, 2])
# keep the lips and a little clearance under the lower lip
lips = ((x / (mouth_hw * 1.08)) ** 2 + ((y + 0.02 * d) / (0.2 * d)) ** 2)
clear = ss(1.0, 1.6, lips)
alpha = np.clip(np.maximum(beard, must * 1.0) * clear, 0, 1)
# a slightly lighter, more stubbly beard high on the cheeks than on the chin / mustache
stub = np.clip(beard * clear, 0, 1)                      # stubble tone layer
mus = np.clip(must * clear, 0, 1)                        # mustache layer (defined, darker)
chin_core = ss(-0.35 * d, -0.7 * d, y) * ss(mouth_hw * 1.6, mouth_hw * 0.9, ax)  # fuller under the lower lip
alpha = stub
# never paint the mouth interior (reddish islands of the cavity mesh)
_t = tex_image(mats[mi_body]["pbrMetallicRoughness"]["baseColorTexture"]["index"])
_uvp = (UV * [_t.shape[1], _t.shape[0]]).astype(int).clip(0, [_t.shape[1] - 1, _t.shape[0] - 1])
_hsv = cv2.cvtColor(_t[..., :3], cv2.COLOR_BGR2HSV)[_uvp[:, 1], _uvp[:, 0]].astype(np.float32)
for arr_ in (alpha, mus, chin_core):
    arr_[(_hsv[:, 1] > 90)] = 0
print("beard vertices", int((alpha > 0.5).sum()), "of", len(alpha))

# ---- skin texture
tex = tex_image(mats[mi_body]["pbrMetallicRoughness"]["baseColorTexture"]["index"])
H, W = tex.shape[:2]
rgb = cv2.cvtColor(tex[..., :3], cv2.COLOR_BGR2RGB).astype(np.float32)
hsv = cv2.cvtColor(tex[..., :3], cv2.COLOR_BGR2HSV).astype(np.float32)
h, s, v = hsv[..., 0] * 2, hsv[..., 1] / 255, hsv[..., 2] / 255
skin_mask = ((h > 4) & (h < 40) & (s > 0.06) & (s < 0.42) & (v > 0.55)).astype(np.float32)
skin_mask = cv2.GaussianBlur(skin_mask, (0, 0), 1.0)
# reference skin colour = median of face island pixels
uvpix = (UV * [W, H]).astype(int).clip(0, [W - 1, H - 1])
ref = np.median(rgb[uvpix[front, 1], uvpix[front, 0]], axis=0)
print("reference skin", ref.round(1))
ratio = rgb / np.maximum(ref, 1)
new_skin = np.clip(SKIN * ratio ** 1.15, 0, 255)
out = rgb * (1 - skin_mask[..., None]) + new_skin * skin_mask[..., None]

# ---- rasterise the beard alpha into UV space (barycentric interpolation per triangle)
def raster(vals):
  mask = np.zeros((H, W), np.float32)
  tri_uv = UV[IDX] * [W, H]
  tri_a = vals[IDX]
  for k in np.where(tri_a.max(1) > 0.01)[0]:
      t = tri_uv[k]; a3 = tri_a[k]
      x0, y0 = np.floor(t.min(0)).astype(int); x1, y1 = np.ceil(t.max(0)).astype(int)
      x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, W - 1), min(y1, H - 1)
      if x1 < x0 or y1 < y0:
          continue
      yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1].astype(np.float32) + 0.5
      T = np.array([[t[0, 0] - t[2, 0], t[1, 0] - t[2, 0]], [t[0, 1] - t[2, 1], t[1, 1] - t[2, 1]]])
      det = np.linalg.det(T)
      if abs(det) < 1e-9:
          continue
      Ti = np.linalg.inv(T)
      dx, dy = xx - t[2, 0], yy - t[2, 1]
      l0 = Ti[0, 0] * dx + Ti[0, 1] * dy
      l1 = Ti[1, 0] * dx + Ti[1, 1] * dy
      l2 = 1 - l0 - l1
      inside = (l0 >= -0.02) & (l1 >= -0.02) & (l2 >= -0.02)
      val = l0 * a3[0] + l1 * a3[1] + l2 * a3[2]
      sub = mask[y0:y1 + 1, x0:x1 + 1]
      sub[inside] = np.maximum(sub[inside], val[inside])
  return cv2.GaussianBlur(mask, (0, 0), 1.2)



# hand-drawn feel: fine grain + soft hair-direction streaks
rng = np.random.default_rng(3)
grain = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 0.8)
streak = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), sigmaX=0.6, sigmaY=2.2)
texture = 0.82 + 0.25 * (grain - 0.5) + 0.25 * (streak - 0.5)
m_stub = raster(alpha); m_must = raster(mus); m_chin = raster(chin_core * stub)
# cel-shaded beard: crisp shapes, flat tones (like a drawn anime character), tiny grain only
cel = lambda m, lo=0.42, hi=0.56: np.clip((m - lo) / (hi - lo), 0, 1) ** 1.0
stub_shape = cel(m_stub)
dark_shape = np.maximum(cel(m_must, 0.38, 0.52), cel(m_chin, 0.40, 0.54))
fine = 1 + 0.06 * (grain - 0.5)
out = out * (1 - (0.42 * stub_shape)[..., None] * fine[..., None])          # stubble = darker skin tone
out = out * (1 - (0.93 * dark_shape)[..., None]) + (BEARD * 1.3) * (0.93 * dark_shape)[..., None]
mask = np.maximum(m_stub * 0.6, np.maximum(m_must, m_chin))
cv2.imwrite(f"{OUT}/body_bake_indian.png", cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
cv2.imwrite(f"{OUT}/beard_mask.png", (mask * 255).astype(np.uint8))

# ---- eyes: amber/gold iris -> dark brown
eye = tex_image(mats[mi_eye]["pbrMetallicRoughness"]["baseColorTexture"]["index"])
ergb = eye[..., :3].astype(np.float32)
ehsv = cv2.cvtColor(eye[..., :3], cv2.COLOR_BGR2HSV).astype(np.float32)
eh, es, ev = ehsv[..., 0] * 2, ehsv[..., 1] / 255, ehsv[..., 2] / 255
iris = ss(0.25, 0.4, es) * ss(18, 28, eh) * ss(70, 58, eh) * ss(0.25, 0.4, ev)
lum = (0.3 * ergb[..., 2] + 0.59 * ergb[..., 1] + 0.11 * ergb[..., 0]) / 255
brown = np.stack([30 + 70 * lum, 50 + 95 * lum, 80 + 120 * lum], -1)  # BGR, warm dark brown scaled by brightness
ergb = ergb * (1 - iris[..., None]) + brown * iris[..., None]
eye_out = eye.copy(); eye_out[..., :3] = np.clip(ergb, 0, 255).astype(np.uint8)
cv2.imwrite(f"{OUT}/eye_indian.png", eye_out)

# preview
prev = np.clip(out, 0, 255).astype(np.uint8)
cv2.imwrite(f"{OUT}/preview_face.jpg", cv2.cvtColor(prev, cv2.COLOR_RGB2BGR)[100:800, 0:700])
print("done")
