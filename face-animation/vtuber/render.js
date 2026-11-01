// node render.js <vrm.json> <model> <out.mp4|out_prefix> [audio] [--frames 0,10] [--bg dusk] [--w 720 --h 1280] [--port 8766]
const { chromium } = require('playwright');
const fs = require('fs');
const { spawn } = require('child_process');

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const [jsonPath, model, out, audio] = args;
const W = +opt('--w', 720), H = +opt('--h', 1280);
const only = opt('--frames', null);
const bg = opt('--bg', 'dusk');
const port = +opt('--port', 8766);

(async () => {
  const d = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
  const browser = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const page = await browser.newPage({ viewport: { width: W, height: H } });
  page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') console.error('page:', m.text().slice(0, 300)); });
  page.on('pageerror', (e) => console.error('pageerror:', e.message));
  await page.goto(`http://127.0.0.1:${port}/index.html?model=${model}&bg=${bg}&w=${W}&h=${H}&variant=${opt('--variant', '')}`);
  const info = await page.evaluate(() => window.ready);
  console.log('model ready', JSON.stringify(info));
  const names = Object.keys(d.expr);
  const frameAt = (i) => {
    const expr = {}; names.forEach((k) => { expr[k] = d.expr[k][i]; });
    return { expr, rot: d.rot[i], pos: d.pos[i], gaze: d.gaze[i], i, fps: d.fps, t: d.n > 1 ? i / (d.n - 1) : 0 };
  };
  const grab = async () => Buffer.from((await page.evaluate(() => window.renderPNG())).split(',')[1], 'base64');

  if (only) {
    // advance through every frame so hair physics is continuous, but only save the requested ones
    const want = new Set(only.split(',').map(Number));
    const last = Math.max(...want);
    for (let i = 0; i <= last; i++) {
      await page.evaluate((f) => window.setFrame(f), frameAt(i));
      if (want.has(i)) { fs.writeFileSync(`${out}_${String(i).padStart(5, '0')}.png`, await grab()); console.log('wrote', i); }
    }
    await browser.close(); return;
  }
  const ff = spawn('ffmpeg', ['-y', '-v', 'error', '-f', 'image2pipe', '-c:v', 'png', '-framerate', String(d.fps), '-i', '-',
    ...(audio ? ['-i', audio, '-map', '0:v', '-map', '1:a?', '-c:a', 'aac', '-b:a', '192k', '-shortest'] : []),
    '-c:v', 'libx264', '-crf', '17', '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out], { stdio: ['pipe', 'inherit', 'inherit'] });
  const t0 = Date.now();
  for (let i = 0; i < d.n; i++) {
    await page.evaluate((f) => window.setFrame(f), frameAt(i));
    const buf = await grab();
    if (!ff.stdin.write(buf)) await new Promise((r) => ff.stdin.once('drain', r));
    if (i % 100 === 0) console.log(`frame ${i}/${d.n} ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  }
  ff.stdin.end(); await new Promise((r) => ff.on('close', r));
  await browser.close();
  console.log('wrote', out);
})().catch((e) => { console.error(e); process.exit(1); });
