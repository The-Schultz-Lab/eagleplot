const { chromium } = require('playwright');
const path = require('path');
const EXAMPLES = path.resolve(__dirname, '../plottle/example-data/Artificial');
const OUT = process.env.OUT || '.';
async function idle(p, q = 500) { let s = null; const t0 = Date.now();
  while (Date.now() - t0 < 60000) { if (await p.locator('[data-testid="stStatusWidget"]').count() === 0) { s ??= Date.now(); if (Date.now() - s > q) return; } else s = null; await p.waitForTimeout(25); } }
async function pick(p, l, o) { await p.getByLabel(l, { exact: true }).first().click(); await p.getByRole('option', { name: o, exact: true }).click(); await idle(p); }
(async () => {
  const b = await chromium.launch(); const p = await b.newPage({ viewport: { width: 1440, height: 900 }, acceptDownloads: true });
  const errs = []; p.on('pageerror', e => errs.push(e.message));
  await p.goto(`http://localhost:${process.argv[2] || 8501}/data-upload`); await idle(p, 800);
  await p.locator('input[type=file]').first().setInputFiles(EXAMPLES + '/scatter_correlation.csv'); await idle(p, 800);
  await p.locator('[data-testid="stSidebar"] a[href$="plot-basic"]').evaluate(a => a.click()); await idle(p, 800);
  // 1) Matplotlib export download (lazy on Streamlit >= 1.52)
  await pick(p, 'Library', 'Matplotlib'); await pick(p, 'Plot type', 'Scatter Plot');
  await p.getByText('Export plot & data').click(); await idle(p);
  await pick(p, 'Format', 'PNG (300 dpi)');
  const [dl] = await Promise.all([p.waitForEvent('download'), p.getByRole('button', { name: 'Download', exact: true }).click()]);
  const fs = require('fs'); const path = await dl.path();
  console.log('export:', dl.suggestedFilename(), fs.statSync(path).size, 'bytes');
  // 2) Plotly click-to-annotate inside the fragment
  await p.screenshot({ path: `${OUT}/features.png` });
  await p.getByLabel('Library', { exact: true }).first().click(); await p.waitForTimeout(800); await p.screenshot({ path: `${OUT}/features.png` }); await p.keyboard.press('Escape');
  await pick(p, 'Library', 'Plotly'); await pick(p, 'Plot type', 'Interactive Scatter'); await idle(p, 800);
  await p.getByRole('button', { name: 'Generate Plot' }).click(); await idle(p, 800);
  const gd = p.locator('.js-plotly-plot').first(); await gd.waitFor();
  const ptLoc = p.locator('.js-plotly-plot .scatterlayer .points path.point').nth(5);
  await ptLoc.scrollIntoViewIfNeeded(); await p.waitForTimeout(500);
  const box = await ptLoc.boundingBox();
  const pt = { x: box.x + box.width / 2, y: box.y + box.height / 2 };
  await p.mouse.move(pt.x, pt.y); await p.waitForTimeout(300); await p.mouse.click(pt.x, pt.y); await idle(p, 800);
  await p.screenshot({ path: `${OUT}/features.png` });
  console.log('after click:', await p.getByText(/point\(s\) marked/).first().textContent().catch(() => 'no marker caption'));
  console.log('page errors:', errs.length);
  await b.close();
})().catch(e => { console.error('FAILED', e.message); process.exit(1); });
