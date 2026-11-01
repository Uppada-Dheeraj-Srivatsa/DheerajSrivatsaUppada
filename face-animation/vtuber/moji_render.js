// node moji_render.js <arkit.json> <out_dir> [--frames 0,10] [--bg #334455] [--s 768]
// Renders the cartoon head per frame as RGBA PNGs (transparent background) for compositing.
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const [jsonPath, outDir] = args;
const only = opt('--frames', null);
const bg = opt('--bg', '');
const S = +opt('--s', 768);
const port = +opt('--port', 8766);

(async () => {
  const d = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
  fs.mkdirSync(outDir, { recursive: true });
  const browser = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const page = await browser.newPage({ viewport: { width: S, height: S } });
  page.on('console', (m) => { if (m.type() === 'error') console.error('page:', m.text().slice(0, 300)); });
  page.on('pageerror', (e) => console.error('pageerror:', e.message));
  await page.goto(`http://127.0.0.1:${port}/moji.html?s=${S}${bg ? '&bg=' + encodeURIComponent(bg) : ''}`);
  await page.evaluate(() => window.ready);
  const frameAt = (i) => { const bs = {}; d.names.forEach((k, j) => { bs[k] = d.frames[i][j]; }); return { bs, rot: d.rot[i] }; };
  const list = only ? only.split(',').map(Number) : [...Array(d.n).keys()];
  const info = [];
  const t0 = Date.now();
  for (const i of list) {
    await page.evaluate((f) => window.setFrame(f), frameAt(i));
    const url = await page.evaluate(() => window.renderPNG());
    info.push(await page.evaluate(() => window.headInfo()));
    fs.writeFileSync(path.join(outDir, `f_${String(i).padStart(5, '0')}.png`), Buffer.from(url.split(',')[1], 'base64'));
    if (i % 100 === 0) console.log(`frame ${i}/${d.n} ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  }
  fs.writeFileSync(path.join(outDir, 'info.json'), JSON.stringify(info[0]));
  await browser.close();
  console.log('done', list.length);
})().catch((e) => { console.error(e); process.exit(1); });
