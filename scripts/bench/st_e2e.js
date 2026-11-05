// End-to-end latency of Quick Plot in a real browser (Streamlit variants).
// usage: node st_e2e.js <port> <label> [--live]
// Scenario: upload ir_spectrum.csv, open Quick Plot, Matplotlib line plot, then
// "change the title and see it on the plot" x6, and "Generate" x3.
const { chromium } = require('playwright');
const path = require('path');
const EXAMPLES = path.resolve(__dirname, '../../eagleplot/example-data/Artificial');
const OUT = process.env.OUT || '.';
const [port, label] = process.argv.slice(2);
const live = process.argv.includes('--live');
const BASE = `http://localhost:${port}`;
const med = xs => { const s = [...xs].sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; };

async function idle(p, quietMs = 400, timeout = 60000) {
  const t0 = Date.now(); let quietSince = null;
  while (Date.now() - t0 < timeout) {
    const running = await p.locator('[data-testid="stStatusWidget"]').count();
    if (running === 0) { quietSince ??= Date.now(); if (Date.now() - quietSince >= quietMs) return; }
    else quietSince = null;
    await p.waitForTimeout(25);
  }
  throw new Error('app never went idle');
}
const plotSrc = p => p.locator('[data-testid="stMain"] [data-testid="stImage"] img').first()
  .getAttribute('src').catch(() => null);
async function waitPlotChange(p, before, timeout = 60000) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeout) {
    const now = await plotSrc(p);
    if (now && now !== before) {
      await p.locator('[data-testid="stMain"] [data-testid="stImage"] img').first()
        .evaluate(img => img.complete || new Promise(r => img.onload = r));
      return;
    }
    await p.waitForTimeout(20);
  }
  throw new Error('plot never changed');
}
async function pick(p, selectLabel, option) {
  await p.getByLabel(selectLabel, { exact: true }).first().click();
  await p.getByRole('option', { name: option, exact: true }).click();
  await idle(p);
}

(async () => {
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2 });
  const errors = []; p.on('pageerror', e => errors.push(e.message));
  let t = Date.now();
  await p.goto(`${BASE}/data-upload`); await idle(p, 800);
  const firstLoad = Date.now() - t;
  await p.locator('input[type=file]').first().setInputFiles(EXAMPLES + '/ir_spectrum.csv');
  await idle(p, 800);
  t = Date.now();
  await p.locator('[data-testid="stSidebar"] a[href$="plot-basic"]').evaluate(a => a.click()); await idle(p, 800);
  const nav = Date.now() - t;
  await pick(p, 'Library', 'Matplotlib');
  await pick(p, 'Plot type', 'Line Plot');
  const res = { 'open Quick Plot (ms)': nav, 'first page load (ms)': firstLoad, generate: [], 'change title -> visible on plot': [] };
  // first plot
  let before = await plotSrc(p);
  await p.getByRole('button', { name: 'Generate Plot' }).click();
  if (live) await idle(p); else { await waitPlotChange(p, before); await idle(p); }
  const title = p.getByLabel('Title', { exact: true }).first();
  for (let i = 0; i < 6; i++) {
    before = await plotSrc(p);
    t = Date.now();
    await title.fill(`Spectrum ${i}`); await title.press('Enter');
    if (!live) { await idle(p, 150); await p.getByRole('button', { name: 'Generate Plot' }).click(); }
    await waitPlotChange(p, before);
    res['change title -> visible on plot'].push(Date.now() - t);
    await idle(p);
  }
  for (let i = 0; i < 3; i++) {
    await title.fill(`Gen ${i}`); await title.press('Enter'); await idle(p);
    before = await plotSrc(p); t = Date.now();
    await p.getByRole('button', { name: 'Generate Plot' }).click();
    if (live) await idle(p, 100); else await waitPlotChange(p, before);
    res.generate.push(Date.now() - t); await idle(p);
  }
  await p.screenshot({ path: `${OUT}/${label}.png` });
  console.log(JSON.stringify({ label, ...res,
    'change title -> visible on plot (median ms)': med(res['change title -> visible on plot']),
    'generate (median ms)': med(res.generate), pageerrors: errors.length }));
  await b.close();
})().catch(e => { console.error('FAILED', e.message); process.exit(1); });
