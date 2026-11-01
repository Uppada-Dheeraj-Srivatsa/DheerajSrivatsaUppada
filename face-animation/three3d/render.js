// node render.js <arkit.json> <look> <out.mp4|out_prefix> [audio_src] [--frames 0,184,230] [--w 720 --h 1280] [--post]
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const [jsonPath, look, out, audio] = args;
const W = +opt('--w', 720), H = +opt('--h', 1280);
const only = opt('--frames', null);
const post = args.includes('--post');
const port = +opt('--port', 8765);

(async () => {
  const data = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
  const browser = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const page = await browser.newPage({ viewport: { width: W, height: H } });
  page.on('console', (m) => { if (m.type() === 'error') console.error('page:', m.text()); });
  page.on('pageerror', (e) => console.error('pageerror:', e.message));
  await page.goto(`http://127.0.0.1:${port}/index.html?look=${look}&w=${W}&h=${H}&ao=${opt("--ao", "0")}`);
  const info = await page.evaluate(() => window.ready);
  console.log('model ready', JSON.stringify({ targets: info.targets.length, size: info.size, eyes: info.eyes }));

  const frameAt = (i) => {
    const bs = {};
    data.names.forEach((k, j) => { bs[k] = data.frames[i][j]; });
    return { bs, rot: data.rot[i], pos: data.pos[i], t: data.n > 1 ? i / (data.n - 1) : 0 };
  };

  if (only) {
    for (const i of only.split(',').map(Number)) {
      await page.evaluate((f) => window.setFrame(f), frameAt(i));
      const url = await page.evaluate(() => window.renderPNG());
      const file = `${out}_${String(i).padStart(5, '0')}.png`;
      fs.writeFileSync(file, Buffer.from(url.split(',')[1], 'base64'));
      console.log('wrote', file);
    }
    await browser.close();
    return;
  }

  const vf = post ? ['-vf', 'noise=alls=5:allf=t,vignette=angle=PI/5'] : [];
  const ff = spawn('ffmpeg', ['-y', '-v', 'error', '-f', 'image2pipe', '-c:v', 'png', '-framerate', String(data.fps), '-i', '-',
    ...(audio ? ['-i', audio, '-map', '0:v', '-map', '1:a?', '-c:a', 'aac', '-b:a', '192k', '-shortest'] : []),
    ...vf, '-c:v', 'libx264', '-crf', '17', '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out],
    { stdio: ['pipe', 'inherit', 'inherit'] });
  const t0 = Date.now();
  for (let i = 0; i < data.n; i++) {
    await page.evaluate((f) => window.setFrame(f), frameAt(i));
    const url = await page.evaluate(() => window.renderPNG());
    const buf = Buffer.from(url.split(',')[1], 'base64');
    if (!ff.stdin.write(buf)) await new Promise((r) => ff.stdin.once('drain', r));
    if (i % 100 === 0) console.log(`frame ${i}/${data.n} ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  }
  ff.stdin.end();
  await new Promise((r) => ff.on('close', r));
  await browser.close();
  console.log('wrote', out);
})().catch((e) => { console.error(e); process.exit(1); });
