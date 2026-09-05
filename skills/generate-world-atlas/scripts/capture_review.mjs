import { chromium } from "./browser/node_modules/@playwright/test/index.mjs";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

if (!process.argv[2]) throw new Error("Usage: node capture_review.mjs <world-output> [msedge|chrome|chromium]");
const root = resolve(process.argv[2]);
const review = resolve(root, "review");
const society = JSON.parse(await readFile(resolve(review, "society.json"), "utf8"));
const captures = resolve(review, "screenshots");
await mkdir(captures, { recursive: true });
const errors = [];
const channel = process.argv[3] || "msedge";
const browser = await chromium.launch({ ...(channel === "chromium" ? {} : { channel }), headless: true });
const page = await browser.newPage({ viewport: { width: 1800, height: 1100 }, deviceScaleFactor: 1.5 });
page.on("pageerror", error => errors.push(error.message));
page.on("requestfailed", request => errors.push(`${request.url()}: ${request.failure()?.errorText}`));
page.on("console", message => { if (message.type() === "error") errors.push(message.text()); });
const url = pathToFileURL(resolve(review, "index.html"));
url.searchParams.set("theme", "political");
const checks = { themes: [], errors, captures: [] };

async function settle() {
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(450);
}

async function capture(name, mapOnly = false) {
  await settle();
  const path = resolve(captures, `${name}.png`);
  if (mapOnly) await page.locator("#map-frame").screenshot({ path, animations: "disabled" });
  else await page.screenshot({ path, animations: "disabled" });
  checks.captures.push(name);
}

try {
  await page.goto(url.href, { waitUntil: "load", timeout: 90000 });
  await page.locator("#zoom-reset").click();
  await capture("political-overview", true);
  for (const theme of ["none", "climate", "biome", "watershed", "potential", "population", "civilizations", "languages", "religions", "political", "provinces"]) {
    await page.locator("#physical-theme").selectOption(theme);
    await settle();
    checks.themes.push({ theme, selected: await page.locator("#physical-theme").inputValue() });
    if (theme === "none") await capture("physical-overview", true);
    if (theme === "civilizations") await capture("civilizations-overview", true);
  }
  await page.locator("#physical-theme").selectOption("political");
  await page.locator("#toggle-cities").check();
  await page.locator("#toggle-transport").check();
  const coreIds = new Set(society.states.map(state => state.core_settlement_id));
  const capital = society.settlements.filter(city => coreIds.has(city.identifier))
    .sort((a, b) => b.population_max - a.population_max)[0];
  const focus = await page.evaluate(({ column, row }) => {
    const svg = document.querySelector("#overlay");
    const point = new DOMPoint(column + .5, row + .5).matrixTransform(svg.getScreenCTM());
    const rect = document.querySelector("#viewport").getBoundingClientRect();
    return { x: point.x, y: point.y, cx: rect.x + rect.width / 2, cy: rect.y + rect.height / 2 };
  }, capital);
  await page.mouse.move(focus.x, focus.y);
  await page.mouse.down();
  await page.mouse.move(focus.cx, focus.cy, { steps: 12 });
  await page.mouse.up();
  for (let i = 0; i < 4; i++) await page.locator("#zoom-in").click();
  checks.focus = { name: capital.name, row: capital.row, column: capital.column };
  await capture("political-transport-detail");
  await page.locator("#physical-theme").selectOption("provinces");
  await capture("provinces-detail");
  await page.locator("#physical-theme").selectOption("religions");
  await capture("religions-detail");
  await page.locator("#view-tectonic").click();
  await settle();
  checks.tectonicLoads = await page.locator("#tectonic-controls").isVisible();
  await page.locator("#view-monsoon").click();
  await settle();
  checks.monsoonLoads = await page.locator("#monsoon-controls").isVisible();
  checks.capitalLabels = await page.locator('#city-labels [data-national-capital="true"]').count();
  checks.holySymbols = await page.locator('#city-symbols polygon[data-holy-city="true"]').count();
  checks.title = await page.title();
  if (errors.length) throw new Error(errors.join("\n"));
  if (checks.capitalLabels !== society.states.length || checks.holySymbols !== society.religions.length) {
    throw new Error("capital or holy-city symbols do not match world data");
  }
} finally {
  await writeFile(resolve(review, "browser-check.json"), JSON.stringify(checks, null, 2));
  await browser.close();
}
console.log(JSON.stringify(checks, null, 2));
