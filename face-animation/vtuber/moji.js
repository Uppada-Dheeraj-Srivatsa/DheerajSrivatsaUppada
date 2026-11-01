// Original glossy 3D cartoon head ("moji" style) built procedurally, driven by tracked expressions.
import * as THREE from 'three';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

const q = new URLSearchParams(location.search);
const S = +(q.get('s') || 768);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(1);
renderer.setSize(S, S);
renderer.setClearColor(0x000000, 0);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
renderer.outputColorSpace = THREE.SRGBColorSpace;
document.body.appendChild(renderer.domElement);
if (q.get('bg')) renderer.setClearColor(new THREE.Color(q.get('bg')), 1);

const scene = new THREE.Scene();
const pm = new THREE.PMREMGenerator(renderer);
scene.environment = pm.fromScene(new RoomEnvironment(renderer), 0.04).texture;
scene.environmentIntensity = 0.55;

const camera = new THREE.PerspectiveCamera(20, 1, 0.1, 50);
camera.position.set(0, 0.04, 4.6);
camera.lookAt(0, 0.04, 0);

const key = new THREE.DirectionalLight(0xfff1e2, 2.4); key.position.set(-1.6, 2.2, 3.2); scene.add(key);
const fill = new THREE.DirectionalLight(0xc9d8ff, 0.7); fill.position.set(2.5, 0.4, 2.0); scene.add(fill);
const rim = new THREE.DirectionalLight(0xffe2c8, 1.6); rim.position.set(0.5, 1.5, -3.0); scene.add(rim);
scene.add(new THREE.HemisphereLight(0xfff4ea, 0x6a4a3a, 0.35));

// ---------------------------------------------------------------- palette
const SKIN = new THREE.Color('#9c6440');
const SKIN_DARK = new THREE.Color('#74452b');
const HAIR = new THREE.Color('#15100e');

// ---------------------------------------------------------------- skin texture (stubble + blush) on the head's UVs
function skinTexture() {
  const W = 1024, H = 512;
  const cv = document.createElement('canvas'); cv.width = W; cv.height = H;
  const g = cv.getContext('2d');
  g.fillStyle = '#' + SKIN.getHexString(); g.fillRect(0, 0, W, H);
  // +z (face front) is at u = 0.25 on three's sphere UVs
  const fx = 0.25 * W;
  // cheeks blush
  for (const s of [-1, 1]) {
    const rg = g.createRadialGradient(fx + s * 70, H * 0.56, 4, fx + s * 70, H * 0.56, 60);
    rg.addColorStop(0, 'rgba(190,80,70,0.20)'); rg.addColorStop(1, 'rgba(190,80,70,0)');
    g.fillStyle = rg; g.fillRect(0, 0, W, H);
  }
  // stubble: lower face and jaw, soft top edge
  const stub = document.createElement('canvas'); stub.width = W; stub.height = H;
  const sg = stub.getContext('2d');
  const grd = sg.createLinearGradient(0, H * 0.58, 0, H * 0.70);
  grd.addColorStop(0, 'rgba(40,24,18,0)'); grd.addColorStop(1, 'rgba(40,24,18,0.42)');
  sg.fillStyle = grd;
  sg.beginPath(); sg.ellipse(fx, H * 0.80, 175, 120, 0, 0, Math.PI * 2); sg.fill();
  // keep lips area lighter
  sg.globalCompositeOperation = 'destination-out';
  const lip = sg.createRadialGradient(fx, H * 0.665, 2, fx, H * 0.665, 40);
  lip.addColorStop(0, 'rgba(0,0,0,0.9)'); lip.addColorStop(1, 'rgba(0,0,0,0)');
  sg.fillStyle = lip; sg.fillRect(0, 0, W, H);
  sg.globalCompositeOperation = 'source-over';
  // grain
  const id = sg.getImageData(0, 0, W, H); const d = id.data;
  let seed = 11; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  for (let i = 0; i < d.length; i += 4) d[i + 3] = d[i + 3] * (0.75 + 0.5 * rnd());
  sg.putImageData(id, 0, 0);
  g.drawImage(stub, 0, 0);
  const t = new THREE.CanvasTexture(cv); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 4; return t;
}

const skinMat = new THREE.MeshPhysicalMaterial({ map: skinTexture(), roughness: 0.48, sheen: 0.5, sheenRoughness: 0.6, sheenColor: new THREE.Color('#ffcfb0'), clearcoat: 0.18, clearcoatRoughness: 0.5 });
const skinPlain = new THREE.MeshPhysicalMaterial({ color: SKIN.clone().multiplyScalar(0.9), roughness: 0.5, sheen: 0.5, sheenRoughness: 0.6, sheenColor: new THREE.Color('#ffcfb0'), clearcoat: 0.15 });
const hairMat = new THREE.MeshPhysicalMaterial({ color: HAIR, roughness: 0.42, sheen: 1.0, sheenRoughness: 0.35, sheenColor: new THREE.Color('#6b4a38'), clearcoat: 0.25, clearcoatRoughness: 0.4 });
const darkMat = new THREE.MeshStandardMaterial({ color: '#120c0a', roughness: 0.7 });

const root = new THREE.Group(); scene.add(root);       // head rotation pivot
const head = new THREE.Group(); root.add(head);

// ---------------------------------------------------------------- head shape with morph targets
const ss = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
function shapeHead(v) {
  let { x, y, z } = v;
  x *= 0.94; y *= 1.05; z *= 0.97;
  if (y < 0) { const t = -y / 0.53; x *= 1 - 0.20 * t * t; z *= 1 - 0.08 * t * t; }
  if (y < -0.2 && z > 0) z += 0.035 * ss(-0.2, -0.45, y) * ss(0.1, 0.4, z);      // chin
  if (y > 0.3) y -= 0.03 * (y - 0.3) / 0.22;                                       // flatter crown
  z += 0.02 * ss(0.12, 0.3, Math.abs(x)) * ss(-0.36, -0.24, y) * (1 - ss(-0.06, 0.06, y)) * ss(0.22, 0.34, z); // cheeks (smooth)
  return new THREE.Vector3(x, y, z);
}
const headGeo = new THREE.SphereGeometry(0.5, 160, 120);
{
  const p = headGeo.attributes.position; const v = new THREE.Vector3();
  const base = [];
  for (let i = 0; i < p.count; i++) { v.fromBufferAttribute(p, i); const s = shapeHead(v); p.setXYZ(i, s.x, s.y, s.z); base.push(s); }
  const jaw = [], smile = [], puff = [];
  for (const b of base) {
    // jaw open: lower front face drops
    const wj = ss(-0.10, -0.32, b.y) * ss(-0.1, 0.25, b.z) * (1 - 0.6 * ss(0.25, 0.42, Math.abs(b.x)));
    jaw.push(0, -0.13 * wj, -0.035 * wj);
    // smile: cheeks rise and round out
    const ws = ss(0.12, 0.26, Math.abs(b.x)) * (1 - ss(0.32, 0.45, Math.abs(b.x))) * ss(-0.32, -0.12, b.y) * (1 - ss(0.0, 0.12, b.y)) * ss(0.15, 0.35, b.z);
    smile.push(Math.sign(b.x) * 0.012 * ws, 0.03 * ws, 0.028 * ws);
    const wp = ss(0.1, 0.25, Math.abs(b.x)) * (1 - ss(0.3, 0.45, Math.abs(b.x))) * ss(-0.35, -0.15, b.y) * (1 - ss(-0.05, 0.05, b.y)) * ss(0.1, 0.3, b.z);
    puff.push(Math.sign(b.x) * 0.05 * wp, 0, 0.03 * wp);
  }
  headGeo.morphAttributes.position = [new THREE.Float32BufferAttribute(jaw, 3), new THREE.Float32BufferAttribute(smile, 3), new THREE.Float32BufferAttribute(puff, 3)];
  headGeo.morphTargetsRelative = true;
  headGeo.computeVertexNormals();
}
const headMesh = new THREE.Mesh(headGeo, skinMat);
head.add(headMesh);

// surface helper: front surface point at (x, y) by ray casting the head
const ray = new THREE.Raycaster();
function surf(x, y, inset = 0) {
  ray.set(new THREE.Vector3(x, y, 3), new THREE.Vector3(0, 0, -1));
  const hit = ray.intersectObject(headMesh, false)[0];
  if (!hit) return new THREE.Vector3(x, y, 0.4);
  const n = hit.face.normal.clone();
  return hit.point.clone().addScaledVector(n, -inset);
}

// ---------------------------------------------------------------- ears
for (const s of [-1, 1]) {
  const ear = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 24), skinPlain);
  ear.scale.set(0.055, 0.115, 0.075); ear.position.set(s * 0.452, -0.04, -0.02); ear.rotation.y = s * 0.35;
  head.add(ear);
  const inner = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 16), new THREE.MeshPhysicalMaterial({ color: SKIN_DARK, roughness: 0.6 }));
  inner.scale.set(0.025, 0.07, 0.035); inner.position.set(s * 0.495, -0.04, 0.0); head.add(inner);
}

// ---------------------------------------------------------------- nose
{
  const p = surf(0, -0.075);
  const nose = new THREE.Mesh(new THREE.SphereGeometry(1, 48, 32), skinPlain);
  nose.scale.set(0.068, 0.058, 0.06); nose.position.copy(p).add(new THREE.Vector3(0, 0, 0.012)); head.add(nose);
  const bridge = new THREE.Mesh(new THREE.CapsuleGeometry(0.03, 0.09, 8, 16), skinPlain);
  // (no separate bridge: the rounded nose reads cleaner)
  for (const s of [-1, 1]) {
    const nos = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 12), new THREE.MeshStandardMaterial({ color: '#4a2a1e', roughness: 0.8 }));
    nos.scale.set(0.014, 0.008, 0.01); nos.position.copy(p).add(new THREE.Vector3(s * 0.024, -0.03, 0.045)); head.add(nos);
  }
}

// ---------------------------------------------------------------- eyes
function irisTexture() {
  const W = 1024, H = 512; const cv = document.createElement('canvas'); cv.width = W; cv.height = H;
  const g = cv.getContext('2d');
  g.fillStyle = '#f7f4f0'; g.fillRect(0, 0, W, H);
  const cx = 0.25 * W, cy = 0.5 * H, R = 96;
  const ig = g.createRadialGradient(cx, cy, 10, cx, cy, R);
  ig.addColorStop(0, '#5a3418'); ig.addColorStop(0.55, '#3b210f'); ig.addColorStop(0.9, '#22130a'); ig.addColorStop(1, '#120a06');
  g.fillStyle = ig; g.beginPath(); g.arc(cx, cy, R, 0, Math.PI * 2); g.fill();
  // iris fibres
  g.strokeStyle = 'rgba(140,90,50,0.25)'; g.lineWidth = 2;
  for (let a = 0; a < Math.PI * 2; a += 0.12) { g.beginPath(); g.moveTo(cx + Math.cos(a) * 42, cy + Math.sin(a) * 42); g.lineTo(cx + Math.cos(a) * 88, cy + Math.sin(a) * 88); g.stroke(); }
  g.fillStyle = '#050302'; g.beginPath(); g.arc(cx, cy, 40, 0, Math.PI * 2); g.fill();
  g.fillStyle = 'rgba(255,255,255,0.95)'; g.beginPath(); g.arc(cx - 34, cy - 30, 18, 0, Math.PI * 2); g.fill();
  g.fillStyle = 'rgba(255,255,255,0.7)'; g.beginPath(); g.arc(cx + 30, cy + 26, 8, 0, Math.PI * 2); g.fill();
  const t = new THREE.CanvasTexture(cv); t.colorSpace = THREE.SRGBColorSpace; return t;
}
const eyeMat = new THREE.MeshPhysicalMaterial({ map: irisTexture(), roughness: 0.08, clearcoat: 1, clearcoatRoughness: 0.03 });
const eyes = {};
const ER = 0.098;
for (const [s, side] of [[-1, 'R'], [1, 'L']]) {   // subject's right eye appears on screen-left (x<0)
  const c = surf(s * 0.165, 0.055); c.z -= 0.068;
  const g = new THREE.Group(); g.position.copy(c); head.add(g);
  const ball = new THREE.Mesh(new THREE.SphereGeometry(ER, 64, 48), eyeMat); g.add(ball);
  // lids: hemispheres slightly larger than the eyeball, rotated to open/close
  const upper = new THREE.Group(); g.add(upper);
  const ul = new THREE.Mesh(new THREE.SphereGeometry(ER * 1.02, 64, 32, 0, Math.PI * 2, 0, Math.PI / 2), skinPlain); upper.add(ul);
  const lash = new THREE.Mesh(new THREE.TorusGeometry(ER * 1.02, 0.0075, 10, 64, Math.PI), darkMat);
  lash.rotation.x = Math.PI / 2; upper.add(lash);   // arc across the front rim of the lid
  const lower = new THREE.Group(); g.add(lower);
  const ll = new THREE.Mesh(new THREE.SphereGeometry(ER * 1.015, 64, 32, 0, Math.PI * 2, Math.PI / 2, Math.PI / 2), skinPlain); lower.add(ll);
  eyes[side] = { g, ball, upper, lower, s };
}

// ---------------------------------------------------------------- eyebrows (chunky, tapered)
function blobChain(points, radii, mat) {
  const geos = [];
  const curve = new THREE.CatmullRomCurve3(points);
  const N = 22;
  for (let i = 0; i < N; i++) {
    const t = i / (N - 1); const p = curve.getPoint(t);
    const r = radii[0] + (radii[1] - radii[0]) * t;
    const sg = new THREE.SphereGeometry(r, 16, 12); sg.scale(1, 0.62, 0.7); sg.translate(p.x, p.y, p.z); geos.push(sg);
  }
  return new THREE.Mesh(mergeGeometries(geos), mat);
}
const brows = {};
for (const [s, side] of [[-1, 'R'], [1, 'L']]) {
  const pts = [0.06, 0.13, 0.2, 0.26].map((x, i) => { const p = surf(s * x, 0.185 + [0, 0.018, 0.02, 0.005][i], 0.004); return p; });
  const g = new THREE.Group(); head.add(g);
  const pivot = pts[0].clone();
  const m = blobChain(pts.map((p) => p.clone().sub(pivot)), [0.03, 0.016], darkMat);
  g.position.copy(pivot); g.add(m);
  brows[side] = { g, base: pivot.clone(), s };
}

// ---------------------------------------------------------------- mouth: drawn on a curved plane (canvas texture), updated per frame
const MW = 512, MH = 256;
const mouthCanvas = document.createElement('canvas'); mouthCanvas.width = MW; mouthCanvas.height = MH;
const mctx = mouthCanvas.getContext('2d');
const mouthTex = new THREE.CanvasTexture(mouthCanvas); mouthTex.colorSpace = THREE.SRGBColorSpace;
const mouthMat = new THREE.MeshStandardMaterial({ map: mouthTex, transparent: true, roughness: 0.55, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -4 });
const mouthPlane = new THREE.PlaneGeometry(0.34, 0.17, 48, 24);
{ // bend onto the face surface
  const p = mouthPlane.attributes.position;
  for (let i = 0; i < p.count; i++) { const x = p.getX(i), y = p.getY(i); const sp = surf(x, -0.205 + y, -0.003); p.setXYZ(i, x, -0.205 + y, sp.z); }
  mouthPlane.computeVertexNormals();
}
const mouthMesh = new THREE.Mesh(mouthPlane, mouthMat); mouthMesh.renderOrder = 2; head.add(mouthMesh);

function drawMouth(e) {
  const g = mctx; g.clearRect(0, 0, MW, MH);
  const cx = MW / 2, cy = MH * 0.42;
  const smile = 0.5 * ((e.mouthSmile_L || 0) + (e.mouthSmile_R || 0));
  const frown = 0.5 * ((e.mouthFrown_L || 0) + (e.mouthFrown_R || 0));
  const funnel = Math.max(e.mouthFunnel || 0, 0.8 * (e.mouthPucker || 0));
  const open = Math.min(1, (e.jawOpen || 0) * 1.6);
  const w = 120 * (1 + 0.28 * smile - 0.38 * funnel);
  const lift = 34 * smile - 26 * frown;
  const h = 110 * open * (1 - 0.15 * smile);
  const L = [cx - w, cy - lift], R = [cx + w, cy - lift];
  // lips (slightly darker skin)
  const lipCol = 'rgba(120,58,44,0.55)';
  if (h < 8) {
    g.strokeStyle = '#3a1a14'; g.lineWidth = 9; g.lineCap = 'round';
    g.beginPath(); g.moveTo(L[0], L[1]); g.bezierCurveTo(cx - w * 0.4, cy + 18 * smile + 4, cx + w * 0.4, cy + 18 * smile + 4, R[0], R[1]); g.stroke();
    g.fillStyle = lipCol; g.beginPath(); g.ellipse(cx, cy + 20, w * 0.55, 12, 0, 0, Math.PI * 2); g.fill();
  } else {
    const top = cy - 6 - 0.1 * h, bot = cy + h;
    const path = new Path2D();
    path.moveTo(L[0], L[1]);
    path.bezierCurveTo(cx - w * 0.45, top + 4 * smile, cx + w * 0.45, top + 4 * smile, R[0], R[1]);
    const bw = 1 - 0.55 * funnel;
    path.bezierCurveTo(cx + w * 0.55 * bw, bot + 22 * smile, cx - w * 0.55 * bw, bot + 22 * smile, L[0], L[1]);
    g.save(); g.fillStyle = '#3d0f16'; g.fill(path); g.clip(path);
    // teeth band under the upper lip
    g.fillStyle = '#f4f1ea'; g.beginPath(); g.ellipse(cx, top + 4, w * 0.8, Math.min(26, h * 0.32), 0, 0, Math.PI * 2); g.fill();
    // tongue
    const tg = g.createRadialGradient(cx, bot + 10, 4, cx, bot + 10, w * 0.8);
    tg.addColorStop(0, '#d9616f'); tg.addColorStop(1, '#8e2c3a');
    g.fillStyle = tg; g.beginPath(); g.ellipse(cx, bot + 6, w * 0.62, h * 0.42 + 8, 0, 0, Math.PI * 2); g.fill();
    g.restore();
    g.strokeStyle = 'rgba(70,28,22,0.9)'; g.lineWidth = 6; g.stroke(path);
    g.fillStyle = lipCol; g.beginPath(); g.ellipse(cx, bot + 12, w * 0.5, 9, 0, 0, Math.PI * 2); g.fill();
  }
  mouthTex.needsUpdate = true;
}

// ---------------------------------------------------------------- mustache (thick, classic)
const must = new THREE.Group(); head.add(must);
{
  const geos = [];
  for (const s of [-1, 1]) {
    const pts = [[0.0, -0.13], [0.05, -0.131], [0.095, -0.139], [0.13, -0.153], [0.152, -0.172]].map(([x, y]) => surf(s * x, y, -0.018));
    const m = blobChain(pts, [0.042, 0.02], hairMat);
    geos.push(m.geometry);
  }
  must.add(new THREE.Mesh(mergeGeometries(geos), hairMat));
}

// ---------------------------------------------------------------- hair: smooth sculpted volume + swept clumps
{
  const hairline = (x, z) => {
    if (z > 0.12) return 0.27 - 0.08 * Math.pow(Math.min(1, Math.abs(x) / 0.38), 2);       // front
    if (z > -0.1) return 0.02 + 0.12 * ss(-0.1, 0.12, z);                                  // temples / sideburns
    return -0.2;                                                                             // back
  };
  const cap = new THREE.SphereGeometry(0.5, 200, 140);
  const p = cap.attributes.position; const v = new THREE.Vector3();
  const dAbove = new Float32Array(p.count);
  for (let i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i); const s0 = shapeHead(v);
    const d = s0.y - hairline(s0.x, s0.z); dAbove[i] = d;
    // thickness grows smoothly from 0 at the hairline: no hard step on the forehead
    const top = ss(0.25, 0.5, s0.y);
    const k = 0.985 + (0.06 + 0.05 * top) * ss(-0.01, 0.07, d);
    p.setXYZ(i, s0.x * k, s0.y * k + 0.01 * top, s0.z * k);
  }
  const idx = cap.index.array; const keep = [];
  for (let i = 0; i < idx.length; i += 3) { if (dAbove[idx[i]] > -0.03 && dAbove[idx[i + 1]] > -0.03 && dAbove[idx[i + 2]] > -0.03) keep.push(idx[i], idx[i + 1], idx[i + 2]); }
  cap.setIndex(keep); cap.computeVertexNormals();
  head.add(new THREE.Mesh(cap, hairMat));
  // a few big swept clumps on top (quiff to one side)
  const geos = [];
  const clumps = [[-0.22, 0.36, 0.27, 0.15, 0.5], [-0.07, 0.4, 0.29, 0.16, 0.35], [0.09, 0.4, 0.27, 0.15, 0.2], [0.23, 0.35, 0.22, 0.13, 0.05],
                  [-0.14, 0.48, 0.1, 0.17, 0.4], [0.1, 0.49, 0.08, 0.17, 0.2], [-0.02, 0.5, -0.12, 0.17, 0.1]];
  for (const [x, y, z, r, sweep] of clumps) {
    const n = new THREE.Vector3(x, y, z).normalize();
    const sg = new THREE.SphereGeometry(r, 40, 28); sg.scale(1.55, 0.42, 1.0);
    sg.rotateY(sweep);
    const m = new THREE.Matrix4().lookAt(new THREE.Vector3(), n, new THREE.Vector3(0, 0, -1));
    sg.applyMatrix4(new THREE.Matrix4().makeRotationX(-Math.PI / 2));
    sg.applyMatrix4(m);
    const pos = shapeHead(n.clone().multiplyScalar(0.5)).multiplyScalar(1.07);
    sg.translate(pos.x, pos.y, pos.z); geos.push(sg);
  }
  head.add(new THREE.Mesh(mergeGeometries(geos), hairMat));
}

// ---------------------------------------------------------------- per-frame control
window.ready = Promise.resolve({ ok: true });

window.setFrame = (f) => {
  const e = f.bs;
  const [rx, ry, rz] = f.rot || [0, 0, 0];
  root.rotation.set(rx * 0.9, ry * 0.9, rz, 'XYZ');
  headMesh.morphTargetInfluences[0] = Math.min(1, (e.jawOpen || 0) * 1.2);
  headMesh.morphTargetInfluences[1] = 0.5 * ((e.mouthSmile_L || 0) + (e.mouthSmile_R || 0));
  headMesh.morphTargetInfluences[2] = e.cheekPuff || 0;
  // eyes
  for (const side of ['L', 'R']) {
    const E = eyes[side];
    const blink = e['eyeBlink_' + side] || 0, wide = e['eyeWide_' + side] || 0, squint = e['eyeSquint_' + side] || 0;
    E.upper.rotation.x = -0.95 - 0.25 * wide + (0.95 + 0.12) * blink + 0.25 * squint;
    E.lower.rotation.x = 0.45 - 0.35 * squint - 0.25 * (e['cheekSquint_' + side] || 0) - 0.3 * blink;
    const yaw = (side === 'L' ? (e.eyeLookOut_L - e.eyeLookIn_L) : (e.eyeLookIn_R - e.eyeLookOut_R)) || 0;
    const pitch = (e['eyeLookDown_' + side] - e['eyeLookUp_' + side]) || 0;
    E.ball.rotation.set(pitch * 0.45, yaw * 0.6, 0);
    // brows
    const B = brows[side];
    const up = e.browInnerUp || 0, outer = e['browOuterUp_' + side] || 0, down = e['browDown_' + side] || 0;
    B.g.position.set(B.base.x - B.s * 0.008 * down, B.base.y + 0.045 * up + 0.03 * outer - 0.018 * down, B.base.z);
    B.g.rotation.z = B.s * (0.25 * up - 0.12 * outer) - B.s * 0.06 * down * -1;
  }
  // mustache follows the upper lip
  const smile = 0.5 * ((e.mouthSmile_L || 0) + (e.mouthSmile_R || 0));
  must.position.set(0, -0.01 * (e.jawOpen || 0) + 0.008 * smile, 0);
  must.scale.set(1 + 0.12 * smile - 0.15 * (e.mouthPucker || 0), 1, 1);
  drawMouth(e);
};

window.headInfo = () => {
  // projected head centre and head height in pixels (for compositing)
  const top = new THREE.Vector3(0, 0.62, 0).applyMatrix4(head.matrixWorld).project(camera);
  const chin = new THREE.Vector3(0, -0.56, 0.1).applyMatrix4(head.matrixWorld).project(camera);
  const c = new THREE.Vector3(0, 0.0, 0).applyMatrix4(head.matrixWorld).project(camera);
  const px = (v) => [(v.x * 0.5 + 0.5) * S, (-v.y * 0.5 + 0.5) * S];
  return { centre: px(c), top: px(top), chin: px(chin) };
};

window.renderPNG = () => { renderer.render(scene, camera); return renderer.domElement.toDataURL('image/png'); };
