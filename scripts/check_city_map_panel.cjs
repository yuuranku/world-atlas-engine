/* Verify lazy city generation in the existing place card, above the world map. */
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const {chromium} = require('C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

async function mainSettled(page) {
  await page.waitForTimeout(160);
  await page.waitForFunction(() => {
    const tiles = tileManager.stats(), overview = overviewManager.stats();
    return tiles.manifestLoaded && tiles.pendingRequests === 0 && tiles.queuedTiles === 0
      && overview.pendingRasters === 0 && overview.queuedRasters === 0;
  }, null, {polling: 100, timeout: 60000});
}

async function atlasCamera(page) {
  return page.evaluate(() => ({scale, translateX, translateY,
    transform: document.getElementById('atlas-camera').getAttribute('transform'),
    selectedSettlementId, place: new URLSearchParams(location.search).get('place')}));
}

async function searchSelect(page, place) {
  const input = page.locator('#city-search');
  await input.fill(place.name);await input.focus();
  await page.waitForFunction(id => Array.from(document.querySelectorAll('#city-results li'))
    .some(node => node.dataset.placeId === id), place.id);
  await page.locator(`#city-results [data-place-id="${place.id}"]`).click();
  await page.waitForFunction(id => ui.selectedPlaceId === id, place.id);
  assert.equal(await page.locator('#place-name').textContent(), place.name);
}

async function citySettled(page, place) {
  await page.waitForFunction(({id, sourceId}) => {
    const stats = cityMap.stats();
    return stats.selected === id && stats.recipeId === sourceId && !stats.loading
      && stats.buildings > 0 && stats.camera.width > 0 && stats.camera.height > 0
      && document.querySelector('#city-fabric-layer [data-city-detail-id]');
  }, {id: place.id, sourceId: place.sourceId}, {polling: 100, timeout: 60000});
  await mainSettled(page);
  return page.evaluate(() => cityMap.stats());
}

async function drawing(page) {
  return page.locator('#city-fabric-layer').evaluate(layer => {
    const root = layer.querySelector('[data-city-detail-id]');
    const sum = attribute => Array.from(layer.querySelectorAll(`[${attribute}]`))
      .reduce((total, node) => total + Number(node.getAttribute(attribute)), 0);
    let fingerprint = 2166136261;
    for (const node of layer.querySelectorAll('.city-blocks path, .city-buildings path')) {
      for (const character of node.getAttribute('d') || '')fingerprint = Math.imul(fingerprint ^ character.charCodeAt(0), 16777619);
    }
    return {cities: layer.querySelectorAll('[data-city-detail-id]').length,
      id: root?.dataset.cityDetailId, datasets: root ? {...root.dataset} : null,
      buildings: sum('data-building-count'), farmland: sum('data-farmland-count'), fingerprint: fingerprint >>> 0,
      innerWalls: Array.from(layer.querySelectorAll('.city-inner-walls')).filter(node => node.getAttribute('d')).length,
      innerBlocks: layer.querySelectorAll('.city-block-old-town, .city-block-civic').length,
      outerBlocks: layer.querySelectorAll('.city-blocks .city-block').length,
      waterSource: layer.querySelector('.city-local-site')?.dataset.waterSource,
      terrain: layer.querySelectorAll('.city-local-terrain path').length,
      contours: layer.querySelectorAll('.city-local-contours path').length,
      panelCamera: Boolean(layer.closest('#city-map-camera')),
      worldCamera: Boolean(layer.closest('#atlas-camera')),
      bounds: root ? (() => {const box = root.getBBox();return {x: box.x, y: box.y, width: box.width, height: box.height};})() : null};
  });
}

async function panelLayout(page) {
  return page.evaluate(() => {
    const rect = id => {const box = document.getElementById(id).getBoundingClientRect();
      return {x: box.x, y: box.y, width: box.width, height: box.height, bottom: box.bottom, right: box.right};};
    return {summary: rect('place-summary'), map: rect('city-map-viewport'), card: rect('place-card'),
      width: innerWidth, height: innerHeight, pageWidth: document.documentElement.scrollWidth,
      pageHeight: document.documentElement.scrollHeight};
  });
}

async function svgReferences(page) {
  const result = await page.locator('#city-fabric-layer').evaluate(layer => {
    const ids = Array.from(document.querySelectorAll('[id]'), node => node.id);
    const duplicateIds = ids.filter((id, index) => ids.indexOf(id) !== index), missing = [];
    for (const node of layer.querySelectorAll('*'))for (const attribute of node.attributes) {
      for (const match of attribute.value.matchAll(/url\(\s*["']?#([^"'()\s]+)/g)) {
        if (!document.getElementById(match[1])) missing.push(match[1]);
      }
      if (attribute.localName === 'href' && attribute.value.startsWith('#')
          && !document.getElementById(attribute.value.slice(1))) missing.push(attribute.value.slice(1));
    }
    return {ids: ids.length, duplicateIds, missing};
  });
  assert.deepEqual(result.duplicateIds, [], 'City and world SVG IDs must be unique');
  assert.deepEqual(result.missing, [], 'Every generated clip and paint reference must resolve');
  return result;
}

async function assertPanel(page, place, stats) {
  assert.equal(await page.locator('dialog, #city-map-dialog').count(), 0, 'The original place card contains the city');
  assert.equal(await page.locator('#place-card #place-summary').count(), 1);
  assert.equal(await page.locator('#place-card #city-map-section #city-map-viewport #city-map-canvas').count(), 1);
  assert.equal(await page.locator('#viewport [data-city-detail-id], #viewport .city-local-site').count(), 0,
    'Generated city detail must not be painted onto the world map');
  assert.equal(stats.selected, place.id);assert.equal(stats.recipeId, place.sourceId);assert.equal(stats.active, 1);
  assert.ok(Number.isSafeInteger(stats.seed), 'Each opening has a random layout seed');
  assert.deepEqual([stats.population.minimum, stats.population.maximum], place.population);
  assert.ok(stats.population.estimate >= stats.population.minimum && stats.population.estimate <= stats.population.maximum);
  assert.ok(Math.abs(stats.anchor.column - place.native[0]) < 1e-7
    && Math.abs(stats.anchor.row - place.native[1]) < 1e-7, 'Generation retains its saved geographic anchor');
  const geometry = await drawing(page);
  assert.equal(geometry.cities, 1);assert.equal(geometry.id, place.sourceId);
  assert.equal(geometry.buildings, stats.buildings);assert.equal(geometry.farmland, stats.farmland);
  assert.equal(geometry.innerBlocks, stats.innerCity);assert.equal(geometry.outerBlocks, stats.outerCity);
  assert.equal(geometry.waterSource, stats.waterSource.kind);
  assert.equal(Number(geometry.datasets.seed), stats.seed);
  assert.equal(Number(geometry.datasets.population), stats.population.estimate);
  assert.equal(Number(geometry.datasets.nativeRow), stats.anchor.row);
  assert.equal(Number(geometry.datasets.nativeColumn), stats.anchor.column);
  assert.equal(geometry.panelCamera, true);assert.equal(geometry.worldCamera, false);
  assert.ok(geometry.bounds.width > 0 && geometry.bounds.height > 0);
  assert.ok(geometry.terrain > 0, 'The city panel includes the local terrain');
  const layout = await panelLayout(page);
  assert.ok(layout.summary.bottom <= layout.map.y + 1, 'City data is above its map');
  assert.ok(layout.map.width >= 280 && layout.map.height >= 220, 'City geometry has a usable viewing area');
  assert.equal(layout.pageWidth, layout.width, 'The panel must fit within the page width');
  assert.ok(layout.card.x >= 0 && layout.card.right <= layout.width + 1);
  return {stats, drawing: geometry, layout, svg: await svgReferences(page), worldCamera: await atlasCamera(page)};
}

function assertRequests(requests, offset, place) {
  const fresh = requests.slice(offset);
  assert.ok(fresh.every(request => request.sourceId === place.sourceId),
    `Selecting ${place.name} must only fetch this city's local data`);
  return fresh;
}

async function regenerateCheck(page, place, before, requests) {
  const worldCamera = await atlasCamera(page), offset = requests.length;
  await page.locator('#place-regenerate').click();
  await page.waitForFunction(seed => !cityMap.stats().loading && cityMap.stats().seed !== seed
    && cityMap.stats().buildings > 0, before.stats.seed, {polling: 100, timeout: 60000});
  const after = await assertPanel(page, place, await citySettled(page, place));
  assert.notEqual(after.stats.seed, before.stats.seed);
  assert.notEqual(after.drawing.fingerprint, before.drawing.fingerprint, 'The random seed changes actual city geometry');
  assert.deepEqual(after.stats.anchor, before.stats.anchor);assert.deepEqual(after.stats.population, before.stats.population);
  assert.equal(after.stats.generatedCount, before.stats.generatedCount + 1);
  assert.deepEqual(after.worldCamera, worldCamera, 'Regenerating a panel city preserves the world-map camera');
  assertRequests(requests, offset, place);return after;
}

async function independentControls(page) {
  const worldBefore = await atlasCamera(page), original = await page.evaluate(() => cityMap.stats());
  await page.locator('#city-map-zoom-in').click();
  const enlarged = await page.evaluate(() => cityMap.stats());
  assert.ok(enlarged.camera.scale > original.camera.scale, 'The panel zoom button enlarges only the city');
  assert.deepEqual(await atlasCamera(page), worldBefore);
  await page.locator('#city-map-zoom-out').click();
  const reduced = await page.evaluate(() => cityMap.stats());
  assert.ok(reduced.camera.scale < enlarged.camera.scale);
  const box = await page.locator('#city-map-viewport').boundingBox();
  const x = box.x + box.width * .55, y = box.y + box.height * .55;
  await page.mouse.move(x, y);await page.mouse.down();
  await page.mouse.move(x + 42, y + 30, {steps: 12});await page.mouse.up();
  const dragged = await page.evaluate(() => cityMap.stats());
  assert.ok(dragged.camera.translateX !== reduced.camera.translateX || dragged.camera.translateY !== reduced.camera.translateY,
    'Dragging the city changes its own camera');
  assert.deepEqual(await atlasCamera(page), worldBefore, 'City drag must not pan the background');
  await page.mouse.move(x, y);await page.mouse.wheel(0, -200);await page.waitForTimeout(160);
  const wheeled = await page.evaluate(() => cityMap.stats());
  assert.ok(wheeled.camera.scale > dragged.camera.scale);
  assert.deepEqual(await atlasCamera(page), worldBefore, 'City wheel must not zoom the background');
  await page.locator('#city-map-region').click();
  const region = await page.evaluate(() => cityMap.stats());assert.equal(region.camera.extent, 'region');
  await page.locator('#city-map-city').click();
  const city = await page.evaluate(() => cityMap.stats());assert.equal(city.camera.extent, 'city');
  assert.ok(city.camera.scale > region.camera.scale, 'City extent shows more street detail than surroundings extent');
  await page.locator('#city-map-reset').click();
  const reset = await page.evaluate(() => cityMap.stats());
  assert.equal(reset.seed, original.seed);assert.equal(reset.generatedCount, original.generatedCount);
  const panelBeforeWorldZoom = reset.camera;
  await page.locator('#zoom-in').click();await mainSettled(page);
  const worldEnlarged = await atlasCamera(page);
  assert.ok(worldEnlarged.scale > worldBefore.scale);
  assert.deepEqual((await page.evaluate(() => cityMap.stats())).camera, panelBeforeWorldZoom,
    'Background zoom must not change the city panel camera');
  await page.locator('#place-recenter').click();await mainSettled(page);
  const located = await page.evaluate(() => cityMap.stats());
  assert.equal(located.seed, original.seed);assert.equal(located.generatedCount, original.generatedCount);
  assert.deepEqual(located.camera, panelBeforeWorldZoom, 'World location action retains the current panel drawing');
  return {original, enlarged, reduced, dragged, wheeled, region, city, reset, worldEnlarged};
}

async function reopenCheck(page, place, previous, requests) {
  await page.locator('#place-close').click();
  await page.waitForFunction(() => cityMap.stats().selected === null
    && document.getElementById('city-fabric-layer').children.length === 0);
  const offset = requests.length;await searchSelect(page, place);
  const reopened = await assertPanel(page, place, await citySettled(page, place));
  assert.notEqual(reopened.stats.seed, previous.stats.seed);
  assert.notEqual(reopened.drawing.fingerprint, previous.drawing.fingerprint, 'Reopening changes actual city geometry');
  assert.deepEqual(reopened.stats.anchor, previous.stats.anchor);assert.deepEqual(reopened.stats.population, previous.stats.population);
  assert.equal(reopened.stats.generatedCount, previous.stats.generatedCount + 1);
  assertRequests(requests, offset, place);return reopened;
}

async function mobileCheck(browser, url, place, output, errors) {
  const context = await browser.newContext({viewport: {width: 390, height: 844},
    isMobile: true, hasTouch: true, deviceScaleFactor: 1});
  try {
    const page = await context.newPage();page.on('pageerror', error => errors.push(String(error)));
    await page.goto(url, {waitUntil: 'load', timeout: 60000});await mainSettled(page);
    await searchSelect(page, place);
    const city = await assertPanel(page, place, await citySettled(page, place));
    assert.equal(city.layout.pageHeight, 844, 'Mobile scroll belongs inside the place panel');
    await page.locator('#city-map-zoom-in').tap();
    assert.ok((await page.evaluate(() => cityMap.stats())).camera.scale > city.stats.camera.scale);
    assert.deepEqual(await atlasCamera(page), city.worldCamera, 'Mobile panel controls retain the background camera');
    await page.locator('#city-map-reset').tap();
    await page.screenshot({path: path.join(output, 'city-mobile.png')});
    const regenerated = await regenerateCheck(page, place, city, []);
    await page.locator('#place-close').tap({timeout: 5000});
    await page.waitForFunction(() => cityMap.stats().selected === null
      && document.getElementById('city-fabric-layer').children.length === 0);
    return {city, regenerated};
  } finally {await context.close();}
}

async function interruptedLoads(browser, url, first, second, review, output) {
  const context = await browser.newContext({viewport: {width: 1500, height: 1000}});
  let releaseResponse = () => {};
  try {
    const page = await context.newPage();await page.goto(url, {waitUntil: 'load', timeout: 60000});await mainSettled(page);
    const pattern = `**/city-maps/${encodeURIComponent(first.sourceId)}.json`;
    await page.route(pattern, route => route.fulfill({status: 503, contentType: 'application/json', body: '{}'}));
    await searchSelect(page, first);
    await page.waitForFunction(() => !cityMap.stats().loading
      && document.getElementById('city-map-status').textContent.includes('重新生成'));
    assert.equal((await drawing(page)).buildings, 0, 'A failed load leaves no partial drawing');
    const error = await page.locator('#city-map-status').textContent();
    await page.screenshot({path: path.join(output, 'city-load-error.png')});
    await page.unroute(pattern);await page.locator('#place-regenerate').click();
    const recovered = await assertPanel(page, first, await citySettled(page, first));
    await page.locator('#place-close').click();
    const body = await fs.readFile(path.join(review, 'city-maps', `${first.sourceId}.json`), 'utf8');
    let receivedRequest;const received = new Promise(resolve => {receivedRequest = resolve;});
    const gate = new Promise(resolve => {releaseResponse = resolve;});
    await page.route(pattern, async route => {receivedRequest();await gate;
      try {await route.fulfill({status: 200, contentType: 'application/json', body});} catch {}});
    await searchSelect(page, first);
    await Promise.race([received, new Promise((_, reject) => setTimeout(() => reject(new Error('Recipe request was not intercepted')), 10000))]);
    assert.equal(await page.evaluate(() => cityMap.stats().loading), true);
    await searchSelect(page, second);
    const next = await assertPanel(page, second, await citySettled(page, second));
    releaseResponse();
    await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    assert.equal(await page.evaluate(() => cityMap.stats().recipeId), second.sourceId,
      'An old response cannot replace the newer selected city');
    assert.equal((await drawing(page)).id, second.sourceId);
    return {error, recovered, selectedAfterCancelledRequest: next};
  } finally {releaseResponse();await context.close();}
}

async function checkCityMapPanel(review, baseURL) {
  const output = path.join(review, 'city-panel-check');await fs.mkdir(output, {recursive: true});
  const places = JSON.parse(await fs.readFile(path.join(review, 'place-index.json'), 'utf8'));
  const cities = places.filter(place => place.kind === 'city');
  const reference = cities.find(place => place.sourceId === 'settlement-0677');
  const largest = cities.filter(place => place.id !== reference?.id).sort((a, b) => b.population[1] - a.population[1])[0];
  const harbor = cities.filter(place => ['海港', '岛港'].includes(place.siteLabel)
    && place.id !== reference?.id && place.id !== largest?.id).sort((a, b) => b.population[1] - a.population[1])[0];
  const nonCity = places.find(place => place.kind === 'state');
  assert.ok(reference && largest && harbor && nonCity, 'Reference, large, harbor and state test records must exist');
  const report = {schema: 'city-panel-browser-check-v1', status: 'failed', baseURL, cities: [], recipeRequests: [], errors: []};
  let browser, page;
  try {
    browser = await chromium.launch({headless: true,
      executablePath: 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
      args: ['--enable-unsafe-swiftshader']});
    page = await browser.newPage({viewport: {width: 1500, height: 1000}});
    page.on('pageerror', error => report.errors.push(String(error)));
    page.on('request', request => {
      const match = new URL(request.url()).pathname.match(/\/city-maps\/([^/]+)\.json$/);
      if (match) report.recipeRequests.push({sourceId: decodeURIComponent(match[1]), url: request.url()});
    });
    page.on('response', response => {if (response.status() >= 400) report.errors.push(`${response.status()} ${response.url()}`);});
    page.on('requestfailed', request => {if (request.failure()?.errorText !== 'net::ERR_ABORTED')
      report.errors.push(`${request.url()}: ${request.failure()?.errorText}`);});
    const url = new URL('index.html', baseURL).href;
    await page.goto(url, {waitUntil: 'load', timeout: 60000});await mainSettled(page);
    assert.deepEqual(report.recipeRequests, [], 'Page load must not fetch local city data');
    assert.equal(await page.locator('[data-city-detail-id]').count(), 0, 'Initial page must not generate city detail');
    assert.equal(await page.evaluate(() => cityMap.stats().generatedCount), 0);
    for (const [label, place] of [['reference', reference], ['large', largest], ['harbor', harbor]]) {
      process.stdout.write(JSON.stringify({stage: 'city-panel', city: place.name, label}) + '\n');
      const offset = report.recipeRequests.length, beforeCount = await page.evaluate(() => cityMap.stats().generatedCount);
      await searchSelect(page, place);
      const city = await assertPanel(page, place, await citySettled(page, place));
      assert.equal(city.stats.generatedCount, beforeCount + 1);
      const fresh = assertRequests(report.recipeRequests, offset, place);assert.equal(fresh.length, 1);
      await page.screenshot({path: path.join(output, `city-${label}.png`)});
      const regenerated = await regenerateCheck(page, place, city, report.recipeRequests);
      const reopened = await reopenCheck(page, place, regenerated, report.recipeRequests);
      if (label === 'reference') report.controls = await independentControls(page);
      report.cities.push({place: {id: place.id, name: place.name, population: place.population, native: place.native, site: place.siteLabel},
        initial: city, regenerated, reopened, recipeRequests: fresh});
    }
    const offset = report.recipeRequests.length;await searchSelect(page, nonCity);await mainSettled(page);
    assert.equal((await drawing(page)).cities, 0);assert.equal(await page.locator('#city-map-section').isVisible(), false);
    assert.equal(await page.locator('#place-regenerate').isVisible(), false);
    assert.equal(report.recipeRequests.length, offset, 'A non-city selection fetches no local city data');
    report.mobile = await mobileCheck(browser, url, reference, output, report.errors);
    report.interruptedLoads = await interruptedLoads(browser, url, reference, largest, review, output);
    assert.deepEqual(report.errors, []);report.status = 'ok';return report;
  } catch (error) {
    report.failure = String(error.stack || error);
    if (page)try {report.diagnostics = await page.evaluate(() => ({tiles: tileManager.stats(), overview: overviewManager.stats(),
      city: cityMap.stats(), status: document.getElementById('city-map-status').textContent, scale}));} catch {}
    throw error;
  } finally {
    const reportPath = path.join(output, 'browser-check.json');await fs.writeFile(reportPath, JSON.stringify(report, null, 2));
    process.stdout.write(JSON.stringify({status: report.status, cities: report.cities.length, errors: report.errors, reportPath, failure: report.failure}) + '\n');
    if (browser)await browser.close();
  }
}

module.exports = {checkCityMapPanel};
if (require.main === module) {
  if (!process.argv[2] || !process.argv[3])throw new Error('Usage: node scripts/check_city_map_panel.cjs <reviewDirectory> <baseURL>');
  checkCityMapPanel(path.resolve(process.argv[2]), process.argv[3]).catch(error => {console.error(error);process.exitCode = 1;});
}
