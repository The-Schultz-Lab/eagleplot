// Render the EaglePlot brand PNGs and logo.ico from the SVGs in brand/.
//
//   node scripts/brand/render_brand.js
//
// Writes eagleplot/assets/logo.png (the head, 960 px), icon.png (app icon, 512 px),
// logo.ico (app icon at 16-256 px) and brand/lockup-*@2x.png. Needs Node with
// `playwright` and a Chromium it can launch; backgrounds stay transparent.
'use strict';
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

const REPO = path.resolve(__dirname, '..', '..');
const BRAND = path.join(REPO, 'brand');
const ASSETS = path.join(REPO, 'eagleplot', 'assets');
const ICO_SIZES = [16, 24, 32, 48, 64, 128, 256];

async function render(browser, svgFile, width, height) {
  const svg = fs.readFileSync(path.join(BRAND, svgFile), 'utf8')
    .replace(/width="\d+" height="\d+"/, `width="${width}" height="${height}"`);
  const page = await browser.newPage({ viewport: { width, height } });
  await page.setContent(`<html><body style="margin:0;background:transparent">${svg}</body></html>`);
  const png = await page.screenshot({ omitBackground: true, clip: { x: 0, y: 0, width, height } });
  await page.close();
  return png;
}

// An .ico file may hold PNG images directly: a 6-byte header, a 16-byte entry per image.
function ico(pngs) {
  const header = Buffer.alloc(6 + 16 * pngs.length);
  header.writeUInt16LE(0, 0); header.writeUInt16LE(1, 2); header.writeUInt16LE(pngs.length, 4);
  let offset = header.length;
  pngs.forEach(({ size, data }, i) => {
    const e = 6 + 16 * i;
    header.writeUInt8(size >= 256 ? 0 : size, e); header.writeUInt8(size >= 256 ? 0 : size, e + 1);
    header.writeUInt16LE(1, e + 4); header.writeUInt16LE(32, e + 6);
    header.writeUInt32LE(data.length, e + 8); header.writeUInt32LE(offset, e + 12);
    offset += data.length;
  });
  return Buffer.concat([header, ...pngs.map((p) => p.data)]);
}

(async () => {
  const browser = await chromium.launch();
  fs.writeFileSync(path.join(ASSETS, 'logo.png'), await render(browser, 'logo.svg', 960, 960));
  fs.writeFileSync(path.join(ASSETS, 'icon.png'), await render(browser, 'icon.svg', 512, 512));
  const icons = [];
  for (const size of ICO_SIZES) icons.push({ size, data: await render(browser, 'icon.svg', size, size) });
  fs.writeFileSync(path.join(ASSETS, 'logo.ico'), ico(icons));
  for (const f of fs.readdirSync(BRAND).filter((n) => /^lockup-.*\.svg$/.test(n))) {
    const m = fs.readFileSync(path.join(BRAND, f), 'utf8').match(/width="(\d+)" height="(\d+)"/);
    const out = f.replace(/\.svg$/, '@2x.png');
    fs.writeFileSync(path.join(BRAND, out), await render(browser, f, 2 * m[1], 2 * m[2]));
  }
  await browser.close();
  console.log('wrote eagleplot/assets/{logo.png,icon.png,logo.ico} and brand/lockup-*@2x.png');
})().catch((e) => { console.error(e); process.exit(1); });
