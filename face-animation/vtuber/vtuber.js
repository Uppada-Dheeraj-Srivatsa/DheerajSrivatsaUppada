import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';

const q = new URLSearchParams(location.search);
const W = +(q.get('w') || 720), H = +(q.get('h') || 1280);
const MODEL = q.get('model') || 'seed-san';
const BG = q.get('bg') || 'dusk';

const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(1);
renderer.setSize(W, H);
renderer.outputColorSpace = THREE.SRGBColorSpace;
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();

function backdrop(kind) {
  const cv = document.createElement('canvas'); cv.width = 720; cv.height = 1280;
  const g = cv.getContext('2d');
  const pal = {
    dusk: ['#2b2350', '#6b4a8a', '#f2a07b'],
    studio: ['#1d2533', '#33415a', '#8aa3c7'],
    peach: ['#ffd9c2', '#ffb8a8', '#d98aa8'],
  }[kind] || ['#2b2350', '#6b4a8a', '#f2a07b'];
  const lg = g.createLinearGradient(0, 0, 0, 1280);
  lg.addColorStop(0, pal[0]); lg.addColorStop(0.55, pal[1]); lg.addColorStop(1, pal[2]);
  g.fillStyle = lg; g.fillRect(0, 0, 720, 1280);
  // soft bokeh lights
  let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  for (let i = 0; i < 26; i++) {
    const x = rnd() * 720, y = rnd() * 760, r = 18 + rnd() * 60;
    const rg = g.createRadialGradient(x, y, 0, x, y, r);
    const a = 0.10 + rnd() * 0.18;
    rg.addColorStop(0, `rgba(255,236,210,${a})`); rg.addColorStop(0.7, `rgba(255,220,190,${a * 0.5})`); rg.addColorStop(1, 'rgba(255,220,190,0)');
    g.fillStyle = rg; g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2); g.fill();
  }
  // rim glow behind the character
  const glow = g.createRadialGradient(360, 520, 40, 360, 560, 520);
  glow.addColorStop(0, 'rgba(255,240,230,0.35)'); glow.addColorStop(1, 'rgba(255,240,230,0)');
  g.fillStyle = glow; g.fillRect(0, 0, 720, 1280);
  // vignette
  const vg = g.createRadialGradient(360, 600, 300, 360, 640, 900);
  vg.addColorStop(0, 'rgba(0,0,0,0)'); vg.addColorStop(1, 'rgba(0,0,0,0.45)');
  g.fillStyle = vg; g.fillRect(0, 0, 720, 1280);
  const t = new THREE.CanvasTexture(cv); t.colorSpace = THREE.SRGBColorSpace; return t;
}
scene.background = backdrop(BG);

const key = new THREE.DirectionalLight(0xfff4ec, Math.PI * 0.62);
key.position.set(0.5, 1.2, 1.6);
scene.add(key);
const rim = new THREE.DirectionalLight(0xffc4e0, Math.PI * 0.22);
rim.position.set(-1.2, 1.0, -1.0);
scene.add(rim);

const camera = new THREE.PerspectiveCamera(20, W / H, 0.05, 50);
let vrm = null;
const bones = {};
const restQ = {};
let headY = 1.4;

window.ready = new Promise((resolve, reject) => {
  const loader = new GLTFLoader();
  loader.register((parser) => new VRMLoaderPlugin(parser));
  loader.load(`./models/${MODEL}.vrm`, (gltf) => {
    vrm = gltf.userData.vrm;
    VRMUtils.removeUnnecessaryVertices(gltf.scene);
    if (VRMUtils.combineSkeletons) VRMUtils.combineSkeletons(gltf.scene);
    VRMUtils.rotateVRM0(vrm);
    vrm.scene.traverse((o) => { o.frustumCulled = false; });
    scene.add(vrm.scene);
    for (const name of ['hips', 'spine', 'chest', 'upperChest', 'neck', 'head', 'leftUpperArm', 'rightUpperArm', 'leftLowerArm', 'rightLowerArm', 'leftHand', 'rightHand', 'leftShoulder', 'rightShoulder']) {
      const b = vrm.humanoid.getNormalizedBoneNode(name);
      if (b) { bones[name] = b; restQ[name] = b.quaternion.clone(); }
    }
    vrm.lookAt.autoUpdate = false;
    vrm.update(0);
    const hp = new THREE.Vector3(); bones.head.getWorldPosition(hp); headY = hp.y;
    // bust shot: head in the upper third, shoulders and chest visible
    const lookY = headY - 0.025;
    const viewH = 0.62;
    const dist = (viewH / 2) / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
    camera.position.set(0, lookY + 0.02, dist);
    camera.lookAt(0, lookY, 0);
    window.__base = { lookY, dist };
    applyVariant().then((info) => resolve({ expressions: Object.keys(vrm.expressionManager.expressionMap), headY, variant: info }), (e) => reject(String(e)));
  }, undefined, (e) => reject(String(e)));
});

// texture variants: swap skin / eye textures on the loaded avatar
const VARIANT = q.get('variant') || '';
async function applyVariant() {
  if (!VARIANT) return 'none';
  const tl = new THREE.TextureLoader();
  const load = (url, like) => new Promise((res, rej) => tl.load(url, (t) => {
    t.flipY = false; t.colorSpace = THREE.SRGBColorSpace;
    if (like) { t.wrapS = like.wrapS; t.wrapT = like.wrapT; t.magFilter = like.magFilter; t.minFilter = like.minFilter; t.channel = like.channel; }
    res(t);
  }, undefined, rej));
  const swaps = { body_bake: `./${VARIANT}/body_bake_${VARIANT}.png`, eye: `./${VARIANT}/eye_${VARIANT}.png` };
  const cache = {};
  let n = 0;
  const mats = [];
  vrm.scene.traverse((o) => { if (o.isMesh) (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => mats.push(m)); });
  for (const m of mats) {
    const base = m.name.replace(' (Outline)', '');
    if (!swaps[base]) continue;
    const t = cache[base] ??= await load(swaps[base], m.map);
    if (m.map) m.map = t;
    if (m.shadeMultiplyTexture) m.shadeMultiplyTexture = t;
    if (base === 'body_bake' && m.shadeColorFactor) m.shadeColorFactor.setRGB(0.86, 0.73, 0.67);
    m.needsUpdate = true; n++;
  }
  if (VARIANT === 'indian') {
    // a regular guy: no backpack or robot arm
    const hide = new Set(['arm_mat', 'arm_plastic', 'green_emit', 'robo_face', 'glass', 'backpack_metal', 'backpack_nm', 'backpack_plastic']);
    vrm.scene.traverse((o) => { if (o.isMesh) { const ms = Array.isArray(o.material) ? o.material : [o.material]; if (ms.every((m) => hide.has(m.name.replace(' (Outline)', '')))) o.visible = false; } });
  }
  return `swapped ${n} materials`;
}

const E = new THREE.Euler();
const Qt = new THREE.Quaternion();
function setBone(name, x, y, z) {
  const b = bones[name]; if (!b) return;
  E.set(x, y, z, 'XYZ'); Qt.setFromEuler(E);
  b.quaternion.copy(restQ[name]).multiply(Qt);
}

let warmed = false;
// f: {expr:{name:value}, rot:[x,y,z], pos:[x,y,z], gaze:[yaw,pitch], t, i, fps}
window.setFrame = (f) => {
  const em = vrm.expressionManager;
  for (const k in f.expr) em.setValue(k, f.expr[k]);
  const [rx, ry, rz] = f.rot;
  const p = f.pos || [0, 0, 0];
  const time = (f.i || 0) / (f.fps || 30);
  const breath = Math.sin(time * 2 * Math.PI / 3.6);
  // head and neck share the rotation; chest follows a little (feels alive, like a VTuber rig)
  setBone('neck', rx * 0.4, ry * 0.4, rz * 0.4);
  setBone('head', rx * 0.6, ry * 0.6, rz * 0.6);
  setBone('upperChest', 0.02 * breath + rx * 0.08, ry * 0.12, rz * 0.15);
  setBone('spine', 0.0, 0.0, -p[0] * 0.35);
  // relaxed arms
  setBone('leftUpperArm', 0, 0, -1.22);
  setBone('rightUpperArm', 0, 0, 1.22);
  setBone('leftLowerArm', 0, -0.25, -0.12);
  setBone('rightLowerArm', 0, 0.25, 0.12);
  setBone('leftShoulder', 0, 0, -0.04 * breath);
  setBone('rightShoulder', 0, 0, 0.04 * breath);
  if (bones.hips) bones.hips.position.y = (restQ.hipsY ??= bones.hips.position.y) + 0.004 * breath;
  vrm.lookAt.yaw = f.gaze ? f.gaze[0] : 0;
  vrm.lookAt.pitch = f.gaze ? f.gaze[1] : 0;
  // gentle camera drift
  const b = window.__base;
  camera.position.set(Math.sin(time * 0.35) * 0.02, b.lookY + 0.02 + Math.sin(time * 0.27) * 0.008, b.dist * (1 - 0.04 * (f.t || 0)));
  camera.lookAt(0, b.lookY, 0);
  if (!warmed) { for (let i = 0; i < 90; i++) vrm.update(1 / 30); warmed = true; }
  vrm.update(1 / (f.fps || 30));
};

window.renderPNG = () => { renderer.render(scene, camera); return renderer.domElement.toDataURL('image/png'); };
