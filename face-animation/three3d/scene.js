import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { KTX2Loader } from 'three/addons/loaders/KTX2Loader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { GTAOPass } from 'three/addons/postprocessing/GTAOPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

const q = new URLSearchParams(location.search);
const W = +(q.get('w') || 720), H = +(q.get('h') || 1280);
const LOOK = q.get('look') || 'porcelain';
const AO = q.get('ao') === '1';

const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(1);
renderer.setSize(W, H);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.VSMShadowMap;
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();

// --- studio environment: dark room with large softboxes (clean, product-shot reflections)
function studioEnv(tint = [1, 1, 1]) {
  const env = new THREE.Scene();
  env.background = new THREE.Color(0x020203);
  const box = (w, h, x, y, z, k) => {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ side: THREE.DoubleSide }));
    m.material.color.setRGB(k * tint[0], k * tint[1], k * tint[2]);
    m.position.set(x, y, z); m.lookAt(0, 0, 0); env.add(m);
  };
  box(6, 3, 0, 6, 1, 2.2);          // overhead
  box(2, 6, -6, 1, 3, 3.2);         // key strip left-front
  box(1.2, 6, 6, 1, 2, 1.4);        // fill strip right
  box(1, 6, -5, 1, -5, 4.0);        // rim left-back
  box(1, 6, 5, 1, -5, 4.0);         // rim right-back
  box(8, 1.2, 0, -5, 3, 0.35);      // floor bounce
  const pm = new THREE.PMREMGenerator(renderer);
  return pm.fromScene(env, 0.02).texture;
}

function radialBg(c1, c2, c3) {
  const cv = document.createElement('canvas'); cv.width = 540; cv.height = 960;
  const g = cv.getContext('2d');
  const grd = g.createRadialGradient(270, 380, 10, 270, 470, 700);
  grd.addColorStop(0, c1); grd.addColorStop(0.5, c2); grd.addColorStop(1, c3);
  g.fillStyle = grd; g.fillRect(0, 0, 540, 960);
  const t = new THREE.CanvasTexture(cv); t.colorSpace = THREE.SRGBColorSpace; return t;
}

const LOOKS = {
  porcelain: {
    bg: ['#41444c', '#17181c', '#050506'], exposure: 0.95, env: 0.85, tint: [1, 0.98, 0.95],
    mat: () => new THREE.MeshPhysicalMaterial({ color: 0xf3eee8, roughness: 0.34, metalness: 0, clearcoat: 0.55, clearcoatRoughness: 0.28, sheen: 0.6, sheenRoughness: 0.5, sheenColor: new THREE.Color(0xffe8dc) }),
    teeth: () => new THREE.MeshPhysicalMaterial({ color: 0xfbf8f2, roughness: 0.25, clearcoat: 0.8 }),
    eyes: 'real',
    key: [0xfff1e4, 2.9], fill: [0xaabfff, 1.0], rimA: [0xffd2a8, 5.0], rimB: [0xa8ccff, 5.0],
    bloom: 0.06,
  },
  gold: {
    bg: ['#33270f', '#100b05', '#020101'], exposure: 0.82, env: 0.9, tint: [1, 0.92, 0.8],
    mat: () => new THREE.MeshPhysicalMaterial({ color: 0xe8b860, metalness: 1, roughness: 0.22, clearcoat: 0.3, clearcoatRoughness: 0.2 }),
    teeth: () => new THREE.MeshPhysicalMaterial({ color: 0xf3dfb0, metalness: 1, roughness: 0.18 }),
    eyes: 'same',
    key: [0xfff0d6, 2.4], fill: [0x9fb4ff, 0.5], rimA: [0xffa860, 3.5], rimB: [0xffd7a0, 3.0],
    bloom: 0.07,
  },
  obsidian: {
    bg: ['#122032', '#060a10', '#000000'], exposure: 1.1, env: 0.9, tint: [0.85, 0.95, 1.1],
    mat: () => new THREE.MeshPhysicalMaterial({ color: 0x0c0e13, metalness: 0.1, roughness: 0.16, clearcoat: 1, clearcoatRoughness: 0.06 }),
    teeth: () => new THREE.MeshPhysicalMaterial({ color: 0x1b1e26, roughness: 0.2, clearcoat: 1 }),
    eyes: 'glow',
    key: [0xdfe9ff, 1.8], fill: [0x5f7cff, 0.4], rimA: [0x39e1ff, 12], rimB: [0xff4fa3, 11],
    bloom: 0.45,
  },
};
const L = LOOKS[LOOK];
renderer.toneMappingExposure = L.exposure;
scene.background = radialBg(...L.bg);
scene.environment = studioEnv(L.tint);
scene.environmentIntensity = L.env;

const camera = new THREE.PerspectiveCamera(22, W / H, 0.5, 5000);
const pivot = new THREE.Group(); scene.add(pivot);
let head = null, eyeL = null, eyeR = null, dict = null;
const size = new THREE.Vector3();
const eyeRest = {};
let camBase = null, lookAt = null;

let composer = null;
function setupComposer() {
  composer = new EffectComposer(renderer);
  composer.setPixelRatio(1); composer.setSize(W, H);
  composer.addPass(new RenderPass(scene, camera));
  if (AO) {
    const ao = new GTAOPass(scene, camera, W, H);
    ao.output = GTAOPass.OUTPUT.Default;
    ao.blendIntensity = 0.85;
    ao.updateGtaoMaterial({ radius: size.y * 0.06, distanceExponent: 1.4, thickness: size.y * 0.02, scale: 1.0, samples: 16 });
    ao.updatePdMaterial({ lumaPhi: 10, depthPhi: 2, normalPhi: 3, radius: 6, rings: 2, samples: 16 });
    composer.addPass(ao);
  }
  if (L.bloom > 0) composer.addPass(new UnrealBloomPass(new THREE.Vector2(W, H), L.bloom, 0.6, 0.82));
  composer.addPass(new OutputPass());
}

function addLights(c, s) {
  const mk = (spec, x, y, z, shadow) => {
    const l = new THREE.DirectionalLight(spec[0], spec[1]);
    l.position.set(c.x + x * s, c.y + y * s, c.z + z * s); l.target.position.copy(c);
    if (shadow) {
      l.castShadow = true;
      l.shadow.mapSize.set(2048, 2048);
      const e = size.y * 0.9;
      Object.assign(l.shadow.camera, { left: -e, right: e, top: e, bottom: -e, near: 0.1, far: s * 4 });
      l.shadow.bias = -0.0002; l.shadow.normalBias = size.y * 0.002; l.shadow.radius = 14; l.shadow.blurSamples = 24;
    }
    scene.add(l, l.target);
  };
  mk(L.key, -1.0, 1.15, 1.5, true);
  mk(L.fill, 1.4, 0.1, 1.0, false);
  mk(L.rimA, -1.6, 0.7, -1.3, false);
  mk(L.rimB, 1.6, 0.5, -1.4, false);
}

window.ready = new Promise((resolve, reject) => {
  const ktx2 = new KTX2Loader().setTranscoderPath('./node_modules/three/examples/jsm/libs/basis/').detectSupport(renderer);
  new GLTFLoader().setKTX2Loader(ktx2).setMeshoptDecoder(MeshoptDecoder).load('./facecap.glb', (gltf) => {
    const root = gltf.scene;
    root.traverse((o) => {
      if (o.isMesh && o.morphTargetDictionary && Object.keys(o.morphTargetDictionary).length > 10) { head = o; dict = o.morphTargetDictionary; }
      if (o.name === 'eyeLeft') eyeL = o;
      if (o.name === 'eyeRight') eyeR = o;
    });
    const under = (o, pred) => { let p = o; while (p) { if (pred(p)) return true; p = p.parent; } return false; };
    root.traverse((o) => {
      if (!o.isMesh) return;
      o.castShadow = true; o.receiveShadow = true;
      const orig = o.material;
      const isEye = under(o, (p) => p === eyeL || p === eyeR);
      const isTeeth = under(o, (p) => p.name === 'teeth');
      if (isEye && L.eyes === 'real') { o.material = new THREE.MeshPhysicalMaterial({ map: orig.map, roughness: 0.12, clearcoat: 1, clearcoatRoughness: 0.02 }); return; }
      if (isEye && L.eyes === 'glow') { o.material = new THREE.MeshPhysicalMaterial({ color: 0x050608, emissive: 0x39e1ff, emissiveMap: orig.map, emissiveIntensity: 1.6, roughness: 0.1, clearcoat: 1 }); return; }
      o.material = isTeeth ? L.teeth(orig) : L.mat(orig);
    });
    root.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(head);
    box.getSize(size);
    const c = box.getCenter(new THREE.Vector3());
    const neck = new THREE.Vector3(c.x, box.min.y + size.y * 0.15, c.z - size.z * 0.1);
    root.position.sub(neck);
    pivot.add(root);
    lookAt = new THREE.Vector3(0, c.y - neck.y - size.y * 0.04, 0);
    const fov = THREE.MathUtils.degToRad(camera.fov);
    const dist = (size.y / 0.62) / 2 / Math.tan(fov / 2);
    camBase = new THREE.Vector3(0, lookAt.y + size.y * 0.06, dist + size.z * 0.3);
    camera.position.copy(camBase); camera.lookAt(lookAt);
    addLights(lookAt, Math.max(size.x, size.y) * 2.5);
    for (const e of [eyeL, eyeR]) if (e) eyeRest[e.uuid] = e.rotation.clone();
    setupComposer();
    composer.render();
    resolve({ targets: Object.keys(dict), size: size.toArray(), eyes: !!eyeL && !!eyeR });
  }, undefined, (e) => reject(String(e)));
});

// f: {bs: {name: value}, rot: [x,y,z], pos: [x,y,z], t: 0..1 (progress through clip)}
window.setFrame = (f) => {
  const inf = head.morphTargetInfluences;
  for (const k in f.bs) { const i = dict[k]; if (i !== undefined) inf[i] = f.bs[k]; }
  const r = f.rot || [0, 0, 0];
  pivot.rotation.set(r[0], r[1], r[2], 'XYZ');
  const p = f.pos || [0, 0, 0];
  pivot.position.set(p[0] * size.y * 0.6, p[1] * size.y * 0.6, p[2] * size.y * -0.6);
  const g = (k) => f.bs[k] || 0;
  const setEye = (e, s) => {
    if (!e) return;
    const yaw = (s === 'L' ? (g('eyeLookOut_L') - g('eyeLookIn_L')) : (g('eyeLookIn_R') - g('eyeLookOut_R'))) * 0.45;
    const pitch = (g('eyeLookDown_' + s) - g('eyeLookUp_' + s)) * 0.35;
    const r0 = eyeRest[e.uuid];
    e.rotation.set(r0.x + pitch, r0.y + yaw, r0.z);
  };
  setEye(eyeL, 'L'); setEye(eyeR, 'R');
  // slow cinematic push-in with a gentle drift
  const t = f.t || 0;
  const ease = t * t * (3 - 2 * t);
  camera.position.set(camBase.x + Math.sin(t * Math.PI) * size.x * 0.06, camBase.y, lookAt.z + (camBase.z - lookAt.z) * (1 - 0.08 * ease));
  camera.lookAt(lookAt);
};

window.renderPNG = () => { composer.render(); return renderer.domElement.toDataURL('image/png'); };
