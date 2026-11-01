# Face animation toolkit

Turn a talking-head video into an animated character that copies your facial expressions
(smile, surprise, blinks, talking, head movement) while the face itself is a different character.

Everything starts from one tracking pass with MediaPipe Face Landmarker (478 landmarks + 52 ARKit-style
blendshapes per frame), then a choice of renderers:

| Style | What you get | Renderer |
|---|---|---|
| 2D cartoon | Flat cartoon head; character-only, over your real video, or side by side | `faceanim/faceanim.py render` |
| 3D head | Porcelain bust or gold statue with studio lighting | `three3d/` (three.js) |
| Photo puppet | A still portrait photo that performs your video | `faceanim/puppet.py` |
| VTuber | Anime VRM avatar with cartoon mouth shapes, anime emotions, hair physics | `vtuber/render.js` |
| Moji head | Original glossy 3D cartoon head placed on your real body (Filmymoji-style) | `vtuber/moji_render.js` + `faceanim/moji_composite.py` |

## Setup

Needs Python 3.10+, Node 18+, ffmpeg, and Playwright with Chromium (for the three.js renderers).

```bash
cd face-animation
./setup.sh                      # Python packages, MediaPipe models, 3D/VRM models, npm installs
npm i -g playwright && npx playwright install chromium   # if Playwright is not installed yet
export NODE_PATH=$(npm root -g) # lets the render scripts find a global Playwright
```

On a headless Linux box MediaPipe also needs `libegl1 libgles2` (`apt-get install libegl1 libgles2`).

## 1. Track the video (always first)

```bash
mkdir -p work out
python3 faceanim/faceanim.py normalize my_video.mp4 work/norm.mp4      # 30 fps, upright, keeps audio
python3 faceanim/faceanim.py track work/norm.mp4 work/track.npz
```

## 2. Pick a style

### 2D cartoon
```bash
python3 faceanim/faceanim.py render work/norm.mp4 work/track.npz out/character.mp4 --mode character   # or overlay / side
python3 faceanim/faceanim.py sweep out/expressions.png                                               # expression test sheet
```

### 3D head (porcelain / gold)
```bash
python3 faceanim/arkit_export.py work/track.npz work/norm.mp4 work/arkit.json
(cd three3d && python3 -m http.server 8765 --bind 127.0.0.1 &)
node three3d/render.js work/arkit.json porcelain out/3d_porcelain.mp4 work/norm.mp4 --post   # or: gold
```

### Photo puppet
Use a frontal, neutral, mouth-closed portrait of an AI-generated person or someone who agreed to it.
```bash
python3 faceanim/puppet.py portrait.png work/track.npz work/norm.mp4 out/realistic.mp4 --size 1080x1080
```

### VTuber avatar
```bash
python3 faceanim/arkit_export.py work/track.npz work/norm.mp4 work/arkit.json   # if not done yet
python3 faceanim/vtuber_export.py work/arkit.json work/vrm.json
(cd vtuber && python3 -m http.server 8766 --bind 127.0.0.1 &)
node vtuber/render.js work/vrm.json seed-san out/vtuber.mp4 work/norm.mp4 --bg dusk          # or: twist-sample, --bg peach
# optional variant with brown skin, beard and no backpack:
(cd vtuber && python3 make_indian.py) && node vtuber/render.js work/vrm.json seed-san out/vtuber_variant.mp4 work/norm.mp4 --variant indian
```
Any VRM 1.0 avatar works: put it in `vtuber/models/NAME.vrm` and pass `NAME` (VRoid Studio exports VRM for free).

### Moji head on your real body (Filmymoji-style)
```bash
(cd vtuber && python3 -m http.server 8766 --bind 127.0.0.1 &)
node vtuber/moji_render.js work/arkit.json work/moji_frames          # transparent PNG head per frame
python3 faceanim/moji_composite.py work/track.npz work/norm.mp4 work/moji_frames out/moji.mp4
```
The head is built procedurally in `vtuber/moji.js` (skin tone, hair, mustache, eyes are all editable there).

## How it works (short)

- **Tracking:** MediaPipe landmarks + blendshapes, cleaned (dropouts next to occlusions are distrusted, long gaps are
  held instead of morphed), calibrated per video, and smoothed with zero-phase filters.
- **Geometry signals:** eye openness, brow height and lip gap are measured from the landmarks, which is more
  sensitive than the raw blendshapes for wide eyes, blinks and talking.
- **Hands over the face:** a selfie segmentation model finds hands in front of the face; overlay styles keep the
  real hands in front, and eyes are kept closed during long occlusions.
- **Photo puppet:** 3D-frontalised landmark motion transferred onto the portrait's MediaPipe mesh, piecewise-affine
  warp, synthesized mouth interior, and a real-skin closed-eye texture for blinks.

## Credits and licenses

- Seed-san avatar: VirtualCast, Inc., [VRM Public License 1.0](https://vrm.dev/licenses/1.0/). **Credit required** in
  anything you publish ("Seed-san model by VirtualCast, Inc."; for the modified variant "Based on Seed-san by VirtualCast, Inc.").
- VRM1_Constraint_Twist_Sample avatar: pixiv Inc., VRM Public License 1.0.
- facecap.glb: from the three.js examples.
- MediaPipe models and canonical face mesh: Google, Apache 2.0.
