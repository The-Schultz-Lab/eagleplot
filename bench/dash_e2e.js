// Same scenario as st_e2e.js, on the Dash prototype.
// Dash has no live preview: "change title -> visible" = edit title + click Generate.
const { chromium } = require('playwright');
const path = require('path');
const EXAMPLES = path.resolve(__dirname, '../plottle/example-data/Artificial');
const OUT = process.env.OUT || '.';
const BASE = 'http://127.0.0.1:8050';
const med = xs => { const s = [...xs].sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; };
const plotSrc = p => p.locator('#qp-plot-output img').first().getAttribute('src').catch(() => null);
async function waitPlotChange(p, before, timeout = 60000) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeout) {
    const now = await plotSrc(p);
    if (now && now !== before) {
      await p.locator('#qp-plot-output img').first().evaluate(img => img.complete || new Promise(r => img.onload = r));
      return;
    }
    await p.waitForTimeout(20);
  }
  throw new Error('plot never changed');
}
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2 });
  if (process.env.DARKLY_CSS) await p.route('**/bootswatch@5.3.6/dist/darkly/bootstrap.min.css', r => r.fulfill({ path: process.env.DARKLY_CSS, contentType: 'text/css' }));
  const errors = []; p.on('pageerror', e => errors.push(e.message));
  let t = Date.now();
  await p.goto(`${BASE}/data-upload`, { waitUntil: 'networkidle' });
  const firstLoad = Date.now() - t;
  await p.setInputFiles('#du-upload input[type=file]', EXAMPLES + '/ir_spectrum.csv');
  await p.waitForSelector('#du-upload-output .alert');
  t = Date.now();
  await p.click('.sidebar >> text=Quick Plot'); await p.waitForSelector('#qp-generate-btn');
  await p.waitForLoadState('networkidle');
  const nav = Date.now() - t;
  await p.click('#qp-plot-type');
  await p.locator('.dash-dropdown-content [role=option]', { hasText: '[Matplotlib] Line Plot' }).first().click();
  await p.waitForTimeout(300);
  const res = { 'open Quick Plot (ms)': nav, 'first page load (ms)': firstLoad, generate: [], 'change title -> visible on plot': [] };
  let before = await plotSrc(p);
  await p.click('#qp-generate-btn'); await waitPlotChange(p, before);
  for (let i = 0; i < 6; i++) {
    before = await plotSrc(p); t = Date.now();
    await p.fill('#qp-title', `Spectrum ${i}`);
    await p.click('#qp-generate-btn'); await waitPlotChange(p, before);
    res['change title -> visible on plot'].push(Date.now() - t);
  }
  for (let i = 0; i < 3; i++) {
    await p.fill('#qp-title', `Gen ${i}`);
    before = await plotSrc(p); t = Date.now();
    await p.click('#qp-generate-btn'); await waitPlotChange(p, before);
    res.generate.push(Date.now() - t);
  }
  await p.screenshot({ path: `${OUT}/dash_bench.png` });
  console.log(JSON.stringify({ label: 'dash', ...res,
    'change title -> visible on plot (median ms)': med(res['change title -> visible on plot']),
    'generate (median ms)': med(res.generate), pageerrors: errors.length }));
  await b.close();
})().catch(e => { console.error('FAILED', e.message); process.exit(1); });
