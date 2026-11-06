// Smoke test for the EaglePlot desktop app (roadmap D5): launches the built Electron app.
//
//   node scripts/desktop/smoke.js EXECUTABLE [--out DIR]
//
// EXECUTABLE is the unpacked app's binary, e.g.
//   packaging/desktop/dist/linux-unpacked/eagleplot   (Linux; run under xvfb-run)
//
// Checks, with a fresh user-data folder:
// - the app starts and shows Home, with no exception and no page error;
// - it runs as the desktop build (Settings: no theme form, settings persist);
// - one click from Home's example gives a plot;
// - settings saved in one run are there after a restart;
// - an https link goes to the system browser (shell.openExternal), not nowhere;
// - nothing is requested over the network: everything is bundled.
// Writes DIR/desktop-smoke.json and screenshots (DIR defaults to build/desktop-smoke).
'use strict';
const fs = require('fs');
const os = require('os');
const path = require('path');
const { _electron: electron } = require('playwright');

const argv = process.argv.slice(2);
const exe = argv.find((a) => !a.startsWith('--'));
const out = argv.includes('--out') ? argv[argv.indexOf('--out') + 1] : 'build/desktop-smoke';
if (!exe) {
  console.log(fs.readFileSync(__filename, 'utf8').split('\n').slice(0, 15).join('\n'));
  process.exit(2);
}
fs.mkdirSync(out, { recursive: true });
const userData = fs.mkdtempSync(path.join(os.tmpdir(), 'eagleplot-desktop-'));
const problems = [];
const result = { exe, userData, runs: [] };

async function launch(label) {
  const t0 = Date.now();
  const app = await electron.launch({
    executablePath: exe,
    args: ['--no-sandbox', `--user-data-dir=${userData}`],
    timeout: 120000,
  });
  const page = await app.firstWindow();
  const run = { label, requests: [], pageErrors: [] };
  page.on('pageerror', (e) => run.pageErrors.push(String(e)));
  page.on('request', (r) => {
    const u = new URL(r.url());
    if (u.protocol === 'http:' || u.protocol === 'https:') run.requests.push(r.url());
  });
  await page.waitForSelector('[data-testid="stSidebar"]', { timeout: 300000 });
  await page.waitForTimeout(2500);
  run.readyMs = Date.now() - t0;
  result.runs.push(run);
  console.log(`${label}: ready in ${(run.readyMs / 1000).toFixed(1)} s`);
  return { app, page, run };
}

async function nav(page, slug) {
  await page.evaluate((s) => {
    const a = [...document.querySelectorAll('[data-testid="stSidebar"] a')].find((el) => (el.getAttribute('href') || '').replace(/^\/+/, '').endsWith(s));
    if (!a) throw new Error(`no sidebar link to ${s}`);
    a.click();
  }, slug);
  await page.waitForTimeout(3500);
}

const mainText = (page) => page.locator('[data-testid="stMain"]').innerText();
const check = (ok, what) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${what}`); if (!ok) problems.push(what); };

(async () => {
  // ── First run ──
  let { app, page, run } = await launch('first run');
  await page.screenshot({ path: path.join(out, 'home.png') });

  // Links: record instead of opening a browser.
  await app.evaluate(({ shell }) => {
    global.__opened = [];
    shell.openExternal = (url) => { global.__opened.push(url); return Promise.resolve(); };
  });
  await page.getByText('About', { exact: true }).first().click();
  await page.waitForTimeout(800);
  await page.getByRole('link', { name: 'NCCU Schultz Lab on GitHub' }).click();
  await page.waitForTimeout(1500);
  const opened = await app.evaluate(() => global.__opened);
  check(opened.some((u) => u.startsWith('https://github.com/')), `https link opens in the system browser (${JSON.stringify(opened)})`);
  check(!page.url().startsWith('https://'), 'the app window stays on the app');

  await nav(page, 'settings');
  let t = await mainText(page);
  check(t.includes('theme is set by the app'), 'Settings: runs as the desktop build (theme note)');
  check(!t.includes('until you close or reload'), 'Settings: settings are persistent');
  check(t.includes('No config.json file found'), 'Settings: a fresh user-data folder has no config.json');
  await page.getByRole('button', { name: 'Save defaults' }).click();
  await page.waitForTimeout(2500);
  check((await mainText(page)).includes('Defaults saved'), 'Settings: defaults saved');

  await nav(page, '');  // Home
  const t0 = Date.now();
  await page.getByRole('button', { name: 'Plot this example' }).first().click();
  await page.waitForSelector('[data-testid="stMain"] img[data-eagleplot-plot]', { timeout: 180000 }).catch(() => {});
  const plotted = (await page.locator('[data-testid="stMain"] img[data-eagleplot-plot]').count()) > 0;
  check(plotted, `Home → Quick Plot shows a plot (${((Date.now() - t0) / 1000).toFixed(1)} s)`);
  result.firstPlotMs = Date.now() - t0;
  await page.screenshot({ path: path.join(out, 'quick-plot.png') });
  check((await page.locator('[data-testid="stException"]').count()) === 0, 'no Streamlit exception');
  await app.close();

  // ── Restart: settings persist ──
  ({ app, page, run } = await launch('restart'));
  await nav(page, 'settings');
  t = await mainText(page);
  check(t.includes('View raw config.json'), 'after a restart, the saved config.json is still there');
  await app.close();

  for (const r of result.runs) {
    check(r.pageErrors.length === 0, `${r.label}: no page errors ${r.pageErrors.length ? JSON.stringify(r.pageErrors.slice(0, 3)) : ''}`);
    check(r.requests.length === 0, `${r.label}: no network requests ${r.requests.length ? JSON.stringify(r.requests.slice(0, 5)) : ''}`);
  }
  result.problems = problems;
  fs.writeFileSync(path.join(out, 'desktop-smoke.json'), JSON.stringify(result, null, 2));
  console.log(`\n${problems.length ? 'FAILED' : 'PASSED'}: ${problems.length} problem(s). Details: ${path.join(out, 'desktop-smoke.json')}`);
  process.exit(problems.length ? 1 : 0);
})().catch((e) => {
  console.error('desktop smoke test aborted:', e.message);
  process.exit(2);
});
