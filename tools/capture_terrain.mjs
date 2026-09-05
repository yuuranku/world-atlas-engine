// Actual browser screenshots, with identical source coordinates for comparisons.
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';
import { mkdir, writeFile } from 'node:fs/promises';

const [rootArgument, playwrightEntry] = process.argv.slice(2);
if (!rootArgument || !playwrightEntry) throw new Error('Usage: node capture_terrain.mjs <terrain-root> <playwright-entry>');
const { chromium } = await import(pathToFileURL(resolve(playwrightEntry)).href);
const root = resolve(rootArgument), captures = resolve(root, 'review/screenshots');
await mkdir(captures, { recursive: true });
const browser = await chromium.launch({ channel: 'msedge', headless: true });
const page = await browser.newPage({ viewport: { width: 1760, height: 1020 }, deviceScaleFactor: 1 });
const errors = [];
page.on('pageerror', error => errors.push(error.message));
page.on('requestfailed', request => errors.push(request.url()));
const report = { errors, screenshots: [], checks: {} };
async function capture(name) {
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  await page.screenshot({ path: resolve(captures, `${name}.png`), animations: 'disabled' });
  report.screenshots.push(name);
}
try {
  const url = pathToFileURL(resolve(root, 'review/index.html'));
  await page.goto(url.href, { waitUntil: 'load' });
  await page.evaluate(() => Promise.all([...document.images].map(image => image.decode())));
  report.checks.imagesLoaded = await page.evaluate(() => [...document.images].every(image => image.naturalWidth > 0));
  await capture('terrain-overview');
  for (const [name, column, row, zoom] of [['northwest-coast',380,300,2.5],['central-inlets',900,510,2.5],['southern-coast',1020,790,2.5],['reference-coast',2000,520,1.8]]) {
    await page.evaluate(({column,row,zoom}) => {scale=zoom;x=innerWidth/2-column*scale;y=innerHeight/2-row*scale;clamp();schedule();}, {column,row,zoom});
    await capture(name);
  }
  await page.evaluate(() => { scale=6; x=innerWidth/2-925*scale; y=innerHeight/2-415*scale; clamp(); schedule(); });
  await capture('maximum-zoom');
  const previous = await page.locator('#scene').getAttribute('style');
  await page.mouse.move(900,550); await page.mouse.down();
  await page.mouse.move(1020,620,{steps:10}); await page.mouse.up();
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(resolve)));
  report.checks.dragWorks = previous !== await page.locator('#scene').getAttribute('style');
  await page.locator('#toggle-contours').click();
  report.checks.contourToggle = await page.locator('#toggle-contours').getAttribute('aria-pressed') === 'false';
  await page.locator('#fit').click();
  report.checks.fitWorks = await page.evaluate(() => Math.abs(scale-fitScale()) < 1e-8);
  report.checks.noErrorPanel = !(await page.locator('#error').isVisible());
  await page.setViewportSize({width:700,height:1020});
  await page.evaluate(() => {scale=2.1;x=350-2000*scale;y=510-510*scale;clamp();schedule();});
  await capture('reference-detail');
  report.url = url.href;
  if (errors.length || Object.values(report.checks).some(value => !value)) throw new Error(JSON.stringify(report));
} finally {
  await writeFile(resolve(root,'review/browser-check.json'), JSON.stringify(report,null,2));
  await browser.close();
}
console.log(JSON.stringify(report,null,2));
