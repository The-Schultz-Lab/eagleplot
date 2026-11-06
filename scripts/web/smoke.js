// Smoke test for the in-browser (stlite) EaglePlot build (roadmap D2.2 / D3.3).
//
//   node scripts/web/smoke.js [URL] [--out DIR] [--example FILE] [--boot-timeout SECONDS]
//                             [--allow-origin ORIGIN ...]
//
// URL is the served site root (default http://127.0.0.1:8000/), e.g. after
//   python packaging/web/build_site.py build/web && python -m http.server -d build/web 8000
//
// Opens every page in the sidebar navigation twice: with an empty session, then after
// loading a built-in example dataset on Data Upload. On each page it records Streamlit
// exceptions ([data-testid="stException"]), uncaught page errors, console errors and
// failed network requests, and saves a screenshot. Writes DIR/smoke.json and exits 1 if
// any page shows a Streamlit exception or an uncaught page error (DIR defaults to
// build/web-smoke, which git ignores).
//
// It also checks three things the browser build promises:
// - The loading splash shows a step while the app starts, reaches its last step (stlite's
//   loadFinished event) and is gone once the app is up.
// - An on-click download (Quick Plot, PNG 300 dpi, built by a callable) arrives and leaves
//   the app running. Under stlite these go through a workaround in index.html.
// - The app is installable (Chromium's installability check passes) and, after this one
//   online visit, opens offline (service worker, roadmap D4.2).
// - Data never leaves the browser: every request goes to the site's own origin, or to an
//   origin allowed with --allow-origin (the Pyodide CDN, https://cdn.jsdelivr.net, is
//   allowed by default).
//
// Needs Node with the `playwright` package and a Chromium it can launch. stlite shows no
// "running" indicator, so a page counts as done when its URL has changed, no element is
// stale and the DOM has been quiet for a while.
'use strict';
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

function parseArgs(argv) {
  const opts = {
    url: 'http://127.0.0.1:8000/',
    out: 'build/web-smoke',
    example: 'sine_cosine_waves.csv',
    bootTimeout: 300,
    allowOrigins: ['https://cdn.jsdelivr.net'],
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--out') opts.out = argv[++i];
    else if (a === '--example') opts.example = argv[++i];
    else if (a === '--boot-timeout') opts.bootTimeout = Number(argv[++i]);
    else if (a === '--allow-origin') opts.allowOrigins.push(argv[++i]);
    else if (a === '-h' || a === '--help') { console.log(fs.readFileSync(__filename, 'utf8').split('\n').slice(0, 26).join('\n')); process.exit(0); }
    else if (a.startsWith('--')) throw new Error(`unknown option ${a}`);
    else opts.url = a;
  }
  if (!opts.url.endsWith('/')) opts.url += '/';
  return opts;
}

const opts = parseArgs(process.argv.slice(2));
const MIN_PAGES = 14; // Home + the 13 pages in eagleplot/pages/
const QUIET_MS = 1500; // no DOM change for this long = page settled
const PAGE_TIMEOUT_MS = 180000;
const SIDEBAR_LINKS = '[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"]';

const slug = (s) => s.replace(/^\/+|\/+$/g, '').replace(/[^a-z0-9]+/gi, '-') || 'home';
const clip = (s, n = 400) => (s.length > n ? s.slice(0, n) + '…' : s);

// Record the time of the last DOM mutation, so "settled" can be measured in the page.
const QUIET_PROBE = () => {
  window.__epLastMutation = performance.now();
  new MutationObserver(() => { window.__epLastMutation = performance.now(); })
    .observe(document, { subtree: true, childList: true, attributes: true, characterData: true });
};

async function settle(page, { quietMs = QUIET_MS, timeout = PAGE_TIMEOUT_MS } = {}) {
  await page.waitForFunction(
    (q) => {
      if (document.querySelector('[data-stale="true"], [data-testid="stSpinner"], [data-testid="stSkeleton"]')) return false;
      return performance.now() - (window.__epLastMutation || 0) >= q;
    },
    quietMs,
    { timeout, polling: 100 },
  );
}

// Streamlit writes page links as hrefs relative to the app root ("" is Home), so they
// are resolved against the site URL, not against the page currently shown.
async function sidebarPages(page) {
  const hrefs = await page
    .locator(SIDEBAR_LINKS)
    .evaluateAll((as) => as.map((a) => [a.getAttribute('href') || '', a.textContent.trim()]));
  const seen = new Map();
  for (const [href, label] of hrefs) {
    const p = new URL(href, opts.url).pathname;
    if (!seen.has(p)) seen.set(p, label);
  }
  return [...seen].map(([pathname, label]) => ({ pathname, label }));
}

async function openPage(page, pathname) {
  const t0 = Date.now();
  const found = await page.locator(SIDEBAR_LINKS).evaluateAll((as, [p, base]) => {
    const a = as.find((el) => new URL(el.getAttribute('href') || '', base).pathname === p);
    if (!a) return false;
    a.click(); // links in collapsed expanders are not "visible" to Playwright
    return true;
  }, [pathname, opts.url]);
  if (!found) throw new Error(`no sidebar link to ${pathname}`);
  // Streamlit may drop the trailing slash of the app root (/app/ -> /app).
  await page.waitForFunction(
    (p) => location.pathname.replace(/\/+$/, '') === p.replace(/\/+$/, ''),
    pathname,
    { timeout: PAGE_TIMEOUT_MS },
  );
  await settle(page);
  return Date.now() - t0;
}

async function collect(page, events) {
  const exceptions = await page.locator('[data-testid="stException"]').evaluateAll((els) => els.map((e) => e.innerText));
  const errorAlerts = await page
    .locator('[data-testid="stAlertContentError"]')
    .evaluateAll((els) => els.map((e) => e.innerText));
  const h1 = await page.locator('[data-testid="stMain"] h1').first().innerText().catch(() => '');
  return {
    title: h1.trim(),
    exceptions: exceptions.map((s) => clip(s, 1500)),
    errorAlerts: errorAlerts.map((s) => clip(s)),
    pageErrors: events.pageErrors.splice(0),
    consoleErrors: events.consoleErrors.splice(0),
    failedRequests: events.failedRequests.splice(0),
  };
}

// Quick Plot with the example data: pick PNG 300 dpi (built on click by a callable, unlike
// PNG 150, which reuses the display image) and download it.
async function checkDownload(page, base, events, results) {
  const qp = new URL('plot-basic', base).pathname;
  const ms = await openPage(page, qp);
  await page.getByText('Export plot & data').first().click();
  await settle(page, { quietMs: 500 });
  const exportBlock = page.locator('[data-testid="stExpander"]').filter({ hasText: 'Export plot & data' }).first();
  await exportBlock.locator('[data-testid="stSelectbox"]').first().click();
  await page.getByRole('option', { name: 'PNG (300 dpi)' }).click();
  await settle(page, { quietMs: 500 });
  const t0 = Date.now();
  const [download] = await Promise.all([
    page.waitForEvent('download', { timeout: 120000 }),
    exportBlock.getByRole('button', { name: 'Download', exact: true }).click(),
  ]);
  const file = path.join(opts.out, 'download-' + download.suggestedFilename());
  await download.saveAs(file);
  const bytes = fs.statSync(file).size;
  const head = fs.readFileSync(file).subarray(0, 4).toString('latin1');
  await settle(page);
  const r = await collect(page, events);
  const alive = (await page.locator(SIDEBAR_LINKS).count()) > 0 &&
    new URL(page.url()).pathname.replace(/\/+$/, '') === qp.replace(/\/+$/, '');
  if (!alive) r.pageErrors.push(`app not running after the download (now at ${page.url()})`);
  if (bytes < 1000 || head !== '\x89PNG') r.pageErrors.push(`download is not a PNG (${bytes} bytes, starts ${JSON.stringify(head)})`);
  results.push({ phase: 'download', page: qp, label: 'Quick Plot PNG 300 dpi', ms: ms + (Date.now() - t0), bytes, ...r });
  console.log(`${r.pageErrors.length ? 'FAIL' : 'ok  '} download  ${download.suggestedFilename()} ${bytes} bytes, app ${alive ? 'still running' : 'GONE'}`);
}

async function checkSplash(page, firstStep, results) {
  const gone = await page
    .waitForFunction(() => !document.getElementById('loading'), null, { timeout: 10000 })
    .then(() => true, () => false);
  const state = await page.evaluate(() => ({
    stage: window.eaglePlotSplash ? window.eaglePlotSplash.stage : null,
    readyMs: window.eaglePlotReadyMs,
  }));
  const errors = [];
  if (!firstStep) errors.push('splash: no step shown while the app was starting');
  if (state.stage !== 3) errors.push(`splash: reached step ${state.stage}, expected 3 (did stlite's loadFinished event fire?)`);
  if (!gone) errors.push('splash: #loading still in the page after the app started');
  if (typeof state.readyMs !== 'number') errors.push('splash: window.eaglePlotReadyMs not set');
  results.push({ phase: 'splash', page: '-', label: 'Loading splash', exceptions: [], errorAlerts: [],
    consoleErrors: [], failedRequests: [], pageErrors: errors });
  console.log(`${errors.length ? 'FAIL' : 'ok  '} splash    first step "${firstStep}", last step ${state.stage + 1} of 4, ${gone ? 'removed' : 'STILL SHOWN'}`);
}

// Installable, and opens offline after this visit: reload with the network off.
async function checkOffline(page, events, results) {
  const r = { phase: 'offline', page: '-', label: 'Installable + offline reload', exceptions: [], errorAlerts: [],
    consoleErrors: [], failedRequests: [], pageErrors: [] };
  const cdp = await page.context().newCDPSession(page);
  const { installabilityErrors } = await cdp.send('Page.getInstallabilityErrors');
  const state = await page.evaluate(() => window.eaglePlotOffline);
  if (installabilityErrors.length) r.pageErrors.push(`not installable: ${installabilityErrors.map((e) => e.errorId).join(', ')}`);
  if (state !== 'ready') r.pageErrors.push(`service worker not controlling the page (${state})`);
  const t0 = Date.now();
  await page.context().setOffline(true);
  try {
    await page.reload();
    await page.waitForFunction(() => (window.eaglePlotBootLog || []).some((e) => e.event === 'ready'), null, { timeout: 180000, polling: 200 });
    const exc = await page.locator('[data-testid="stException"]').count();
    if (exc) r.pageErrors.push(`${exc} exception(s) after the offline reload`);
  } catch (e) {
    r.pageErrors.push(`offline reload: ${e.message.split('\n')[0]}`);
  } finally {
    await page.context().setOffline(false);
  }
  r.ms = Date.now() - t0;
  events.failedRequests.splice(0); // requests failing while offline are expected
  results.push(r);
  console.log(`${r.pageErrors.length ? 'FAIL' : 'ok  '} offline   installable ${installabilityErrors.length ? 'NO' : 'yes'}, ` +
    `service worker ${state}, offline reload ${r.pageErrors.length ? 'failed' : `ready in ${(r.ms / 1000).toFixed(1)} s`}`);
}

async function loadExample(page, base, example, events, results) {
  const dataPath = new URL('data-upload', base).pathname;
  const ms = await openPage(page, dataPath);
  await page.getByRole('tab', { name: 'Example Datasets' }).click();
  await settle(page, { quietMs: 500 });
  // Each example is a bordered card with a caption and a "Load" button.
  const card = page
    .locator('[data-testid="stVerticalBlock"]')
    .filter({ has: page.getByRole('button', { name: 'Load', exact: true }) })
    .filter({ hasText: exampleLabel(example) })
    .last();
  const t0 = Date.now();
  await card.getByRole('button', { name: 'Load', exact: true }).click();
  await page.waitForFunction(
    (f) => document.querySelector('[data-testid="stMain"]')?.innerText.includes(f),
    example,
    { timeout: PAGE_TIMEOUT_MS },
  );
  await settle(page);
  const r = await collect(page, events);
  results.push({ phase: 'load-example', page: dataPath, label: 'Load example', ms: ms + (Date.now() - t0), ...r });
  await page.screenshot({ path: path.join(opts.out, 'with-data-00-load-example.png'), fullPage: true });
}

// Card labels as shown on Data Upload (see _EXAMPLES in eagleplot/pages/1_Data_Upload.py).
function exampleLabel(file) {
  const known = {
    'normal_distribution.csv': 'Normal Distribution',
    'bimodal_distribution.csv': 'Bimodal Distribution',
    'sine_cosine_waves.csv': 'Sine & Cosine Waves',
    'scatter_correlation.csv': 'Scatter Correlation',
    'grouped_categorical.csv': 'Grouped Categorical',
    'molecular_properties.csv': 'Molecular Properties',
  };
  return known[file] || file;
}

(async () => {
  fs.mkdirSync(opts.out, { recursive: true });
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  await page.addInitScript(QUIET_PROBE);
  const events = { pageErrors: [], consoleErrors: [], failedRequests: [] };
  page.on('pageerror', (e) => events.pageErrors.push(clip(String(e.stack || e.message), 1500)));
  page.on('console', (m) => { if (m.type() === 'error') events.consoleErrors.push(clip(m.text())); });
  page.on('requestfailed', (r) => events.failedRequests.push(`${r.failure()?.errorText} ${r.url()}`));
  page.on('response', (r) => { if (r.status() >= 400) events.failedRequests.push(`HTTP ${r.status()} ${r.url()}`); });
  const origins = new Map(); // origin -> request count
  page.on('request', (r) => {
    const u = new URL(r.url());
    if (u.protocol === 'data:' || u.protocol === 'blob:') return;
    origins.set(u.origin, (origins.get(u.origin) || 0) + 1);
  });

  const results = [];
  const base = opts.url;
  const t0 = Date.now();
  await page.goto(base);
  // The splash draws its first step from an inline script, before stlite loads.
  const splashStep = await page
    .waitForFunction(() => document.querySelector('#loading #ep-step small')?.parentElement.textContent, null, { timeout: 30000 })
    .then((h) => h.jsonValue(), () => null);
  await page.waitForSelector(SIDEBAR_LINKS, { timeout: opts.bootTimeout * 1000 });
  await settle(page);
  const bootMs = Date.now() - t0;
  console.log(`booted in ${(bootMs / 1000).toFixed(1)} s`);
  results.push({ phase: 'boot', page: new URL(base).pathname, label: 'Boot', ms: bootMs, ...(await collect(page, events)) });
  await checkSplash(page, splashStep, results);

  const pages = await sidebarPages(page);
  console.log(`${pages.length} pages in the sidebar: ${pages.map((p) => p.label).join(', ')}`);
  if (pages.length < MIN_PAGES) throw new Error(`expected at least ${MIN_PAGES} pages, found ${pages.length}`);

  async function visitAll(phase) {
    let i = 0;
    for (const { pathname, label } of pages) {
      i += 1;
      let ms = null;
      let navError = null;
      try { ms = await openPage(page, pathname); } catch (e) { navError = e.message.split('\n')[0]; }
      const r = await collect(page, events);
      if (navError) r.pageErrors.push(`navigation: ${navError}`);
      results.push({ phase, page: pathname, label, ms, ...r });
      const shot = path.join(opts.out, `${phase}-${String(i).padStart(2, '0')}-${slug(pathname.slice(new URL(base).pathname.length))}.png`);
      await page.screenshot({ path: shot, fullPage: true });
      const bad = r.exceptions.length + r.pageErrors.length;
      console.log(`${bad ? 'FAIL' : 'ok  '} ${phase.padEnd(9)} ${label.padEnd(22)} ${ms === null ? '   -  ' : (ms / 1000).toFixed(1).padStart(5) + ' s'}  ` +
        `exc=${r.exceptions.length} pageerr=${r.pageErrors.length} console=${r.consoleErrors.length} ` +
        `net=${r.failedRequests.length} alerts=${r.errorAlerts.length}`);
    }
  }

  await visitAll('empty');
  await loadExample(page, base, opts.example, events, results);
  await visitAll('with-data');
  await checkDownload(page, base, events, results);
  await checkOffline(page, events, results);

  await browser.close();
  const allowed = new Set([new URL(base).origin, ...opts.allowOrigins]);
  const foreign = [...origins].filter(([o]) => !allowed.has(o));
  console.log(`requests by origin: ${[...origins].map(([o, n]) => `${o} ${n}`).join(', ')}`);
  if (foreign.length) {
    results.push({ phase: 'privacy', page: '-', label: 'Requests to other origins', exceptions: [], errorAlerts: [],
      consoleErrors: [], failedRequests: [], pageErrors: foreign.map(([o, n]) => `${n} request(s) to ${o}`) });
  }
  const failures = results.filter((r) => r.exceptions.length || r.pageErrors.length);
  const summary = { url: base, bootMs, pages: pages.length, origins: Object.fromEntries(origins), failures: failures.length, results };
  fs.writeFileSync(path.join(opts.out, 'smoke.json'), JSON.stringify(summary, null, 2));
  for (const f of failures) {
    console.log(`\n--- ${f.phase} ${f.label} (${f.page})`);
    for (const e of f.exceptions) console.log('  exception:', e.replace(/\n/g, '\n    '));
    for (const e of f.pageErrors) console.log('  page error:', e);
  }
  console.log(`\n${failures.length ? 'FAILED' : 'PASSED'}: ${results.length} page loads, ${failures.length} with exceptions or page errors. Details: ${path.join(opts.out, 'smoke.json')}`);
  process.exit(failures.length ? 1 : 0);
})().catch((e) => {
  console.error('smoke test aborted:', e.message);
  process.exit(2);
});
