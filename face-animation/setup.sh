#!/usr/bin/env bash
# Downloads the models this project needs and installs dependencies.
# Run from the face-animation/ folder:  ./setup.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "== Python packages"
pip install mediapipe opencv-python-headless numpy scipy matplotlib

echo "== MediaPipe models"
curl -sSL -o face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
curl -sSL -o selfie_multiclass_256x256.tflite \
  https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite

echo "== Face mesh triangles (MediaPipe canonical face model) for the photo puppet"
mkdir -p assets
curl -sSL -o assets/canonical_face_model.obj \
  https://raw.githubusercontent.com/google/mediapipe/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj
python3 - <<'EOF'
import numpy as np
tris = [[int(t.split('/')[0]) - 1 for t in l.split()[1:4]] for l in open('assets/canonical_face_model.obj') if l.startswith('f ')]
np.save('assets/face_tris.npy', np.array(tris))
print('face_tris.npy:', len(tris), 'triangles')
EOF

echo "== 3D head (three.js facecap model with 52 ARKit blendshapes)"
curl -sSL -o three3d/facecap.glb \
  https://raw.githubusercontent.com/mrdoob/three.js/r160/examples/models/gltf/facecap.glb
(cd three3d && npm install --no-audit --no-fund)

echo "== VTuber avatars (official VRM samples) + VRM renderer"
mkdir -p vtuber/models
curl -sSL -o vtuber/models/seed-san.vrm \
  https://raw.githubusercontent.com/vrm-c/vrm-specification/master/samples/Seed-san/vrm/Seed-san.vrm
curl -sSL -o vtuber/models/twist-sample.vrm \
  https://raw.githubusercontent.com/pixiv/three-vrm/dev/packages/three-vrm/examples/models/VRM1_Constraint_Twist_Sample.vrm
(cd vtuber && npm install --no-audit --no-fund)

echo "== Done. The render scripts also need ffmpeg and Playwright with Chromium (see README)."
