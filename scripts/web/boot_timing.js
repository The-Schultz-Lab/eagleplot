// Start-up timing for the in-browser (stlite) EaglePlot build (roadmap D4.1).
//
//   node scripts/web/boot_timing.js [URL] [--runs N] [--throttle MBPS,LATENCY_MS] [--out FILE]
//
// URL is the served app (default http://127.0.0.1:8000/app/). Each run opens a fresh browser
// profile (a cold, first visit), waits until the app is ready, then reloads in the same
// profile (a warm visit: the browser cache is kept). For each visit it reports the time to
// "ready" (the sidebar is drawn), the start-up timeline from index.html
// (window.eaglePlotBootLog: stlite's progress messages, loadFinished, ready) and the bytes
// downloaded. --throttle emulates a network, e.g. 20,40 for 20 Mbit/s with 40 ms latency.
//
// Needs Node with the `playwright` package and a Chromium it can launch.
'use strict';
const fs = require('fs');
const { chromium } = require('playwright');

function parseArgs(argv) {
  const opts = { url: 'http://127.0.0.1:8000/app/', runs: 3, throttle: null, out: null };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--runs') opts.runs = Number(argv[++i]);
    else if (a === '--throttle') {
      const [mbps, latency] = argv[++i].split(',').map(Number);
      opts.throttle = { mbps, latency: latency || 0 };
    } else if (a === '--out') opts.out = argv[++i];
    else if (a === '-h' || a === '--help') {
      console.log(fs.readFileSync(__filename, 'utf8').split('\n').slice(0, 12).join('\n'));
      process.exit(0);
    } else if (a.startsWith('--')) throw new Error(`unknown option ${a}`);
    else opts.url = a;
  }
  return opts;
}

const opts = parseArgs(process.argv.slice(2));
const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length ? s[Math.floor((s.length - 1) / 2)] : null;
};

async function visit(page, cdp, reload) {
  // Counted from the page's requestfinished events, which include the Web Worker that runs
  // Python (Pyodide, packages, wheels); a page-level CDP session would miss those.
  let bytes = 0;
  let requests = 0;
  const pending = [];
  const onDone = (req) => {
    requests += 1;
    pending.push(req.sizes().then((sz) => { bytes += sz.responseBodySize + sz.responseHeadersSize; }).catch(() => {}));
  };
  page.on('requestfinished', onDone);
  const t0 = Date.now();
  if (reload) await page.reload();
  else await page.goto(opts.url);
  await page.waitForFunction(() => (window.eaglePlotBootLog || []).some((e) => e.event === 'ready'), null, {
    timeout: 600000,
    polling: 100,
  });
  const wall = Date.now() - t0;
  const log = await page.evaluate(() => window.eaglePlotBootLog);
  page.off('requestfinished', onDone);
  await Promise.all(pending);
  return { wall, readyMs: log.find((e) => e.event === 'ready').t, log, mb: +(bytes / 1e6).toFixed(1), requests };
}

(async () => {
  const browser = await chromium.launch();
  const results = { url: opts.url, throttle: opts.throttle, cold: [], warm: [] };
  for (let r = 0; r < opts.runs; r++) {
    const ctx = await browser.newContext();
    const page = await ctx.newPage();
    const cdp = await ctx.newCDPSession(page);
    await cdp.send('Network.enable');
    if (opts.throttle) {
      await cdp.send('Network.emulateNetworkConditions', {
        offline: false,
        latency: opts.throttle.latency,
        downloadThroughput: (opts.throttle.mbps * 1e6) / 8,
        uploadThroughput: (opts.throttle.mbps * 1e6) / 8,
      });
    }
    const cold = await visit(page, cdp, false);
    const warm = await visit(page, cdp, true);
    results.cold.push(cold);
    results.warm.push(warm);
    console.log(`run ${r + 1}: cold ${(cold.readyMs / 1000).toFixed(1)} s (${cold.mb} MB), ` +
      `warm ${(warm.readyMs / 1000).toFixed(1)} s (${warm.mb} MB)`);
    await ctx.close();
  }
  await browser.close();

  // Median timeline of the last cold and warm run, as phase durations.
  for (const kind of ['cold', 'warm']) {
    const runs = results[kind];
    console.log(`\n${kind}: ready median ${(median(runs.map((v) => v.readyMs)) / 1000).toFixed(1)} s, ` +
      `downloaded median ${median(runs.map((v) => v.mb))} MB`);
    const log = runs[runs.length - 1].log;
    let prev = 0;
    for (const e of log) {
      console.log(`  ${String(e.t).padStart(6)} ms  +${String(e.t - prev).padStart(5)}  ${e.event}${e.detail ? ': ' + e.detail : ''}`);
      prev = e.t;
    }
  }
  if (opts.out) fs.writeFileSync(opts.out, JSON.stringify(results, null, 2));
})().catch((e) => {
  console.error('boot timing aborted:', e.message);
  process.exit(2);
});
