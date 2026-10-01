const {chromium} = require('C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const path = require('node:path');
const fs = require('node:fs/promises');
const assert = require('node:assert/strict');
const {spawnSync} = require('node:child_process');
const {observeGlobeCompositions}=require('../scripts/check_review_browser.cjs');

(async () => {
  const root = path.resolve(process.argv[2]);
  const baseURL=process.argv[4];
  assert.ok(baseURL&&/^https?:\/\//.test(baseURL),
    'Pass the served review HTTP URL after the review and canonical grid directories');
  const gridRoot = path.resolve(process.argv[3] || path.join(root, '..', 'grid'));
  const browser = await chromium.launch({headless: true,
    executablePath: 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    args: ['--enable-unsafe-swiftshader']});
  try {
    const page = await browser.newPage({viewport: {width: 1200, height: 900}, deviceScaleFactor: 2});
    await observeGlobeCompositions(page);
    const errors = [];
    page.on('pageerror', error => errors.push(String(error)));
    page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
    page.on('requestfailed', request => {
      if(request.failure()?.errorText!=='net::ERR_ABORTED') errors.push(request.url() + ': ' + JSON.stringify(request.failure()));
    });
    page.on('response', response=>{if(response.status()>=400) errors.push(`${response.status()} ${response.url()}`);});
    await page.goto(new URL('globe.html',baseURL).href, {waitUntil: 'load', timeout: 120000});
    await page.waitForFunction(() => window.WorldAtlasGlobeDiagnostics?.().appliedTheme === 'terrain' && WorldAtlasGlobeDiagnostics().textures === 1, undefined, {timeout: 120000});
    const initial = await page.evaluate(() => {
      const canvas = document.querySelector('canvas');
      const bounds = canvas.getBoundingClientRect();
      return {...WorldAtlasGlobeDiagnostics(), backingWidth: canvas.width, backingHeight: canvas.height,
        cssWidth: bounds.width, cssHeight: bounds.height, devicePixelRatio,
        sourceType: new DOMParser().parseFromString(WorldAtlasGlobe.surface, 'image/svg+xml').documentElement.localName,
        surfaceBytes: WorldAtlasGlobe.surface.length,
        overlayURLs: WorldAtlasGlobe.textures};
    });
    const expectedThemes=['terrain','climate','biome','watershed','potential','habitability','vegetation','population',
      'civilizations','languages','religions','political','provinces'];
    assert.deepEqual(Object.keys(initial.overlayURLs),expectedThemes);
    initial.overlayBytes={};
    for(const theme of expectedThemes){
      assert.equal(initial.overlayURLs[theme],`globe-theme-${theme}.svg`);
      initial.overlayBytes[theme]=(await fs.stat(path.join(root,initial.overlayURLs[theme]))).size;
      assert.ok(initial.overlayBytes[theme]>0,'Every lazy globe theme must have a delivered SVG');
    }
    assert.equal(initial.sourceType, 'svg');
    assert.equal(initial.pixelRatio, 2, 'high-DPI rendering must use the actual screen resolution');
    assert.equal(initial.backingWidth, initial.cssWidth * initial.devicePixelRatio);
    assert.equal(initial.backingHeight, initial.cssHeight * initial.devicePixelRatio);
    assert.equal(initial.cssWidth, 1200, 'high-DPI canvas must fit its viewport');
    assert.equal(initial.cssHeight, 900);
    assert.ok(initial.textureWidth >= 4096, 'globe detail needs a full-resolution vector texture');
    assert.equal(initial.textureWidth, initial.textureHeight * 2);
    assert.equal(initial.textures, 1);
    const expectedResult = spawnSync(path.join(__dirname,'..','.venv','Scripts','python.exe'),
      [path.join(__dirname,'..','scripts','globe_seam_oracle.py'),root,gridRoot,
       String(initial.textureWidth),String(initial.textureHeight)],
      {encoding:'utf8',env:{...process.env,PYTHONPATH:path.join(__dirname,'..','src')}});
    assert.equal(expectedResult.status,0,expectedResult.stderr);
    const expectedSeam=JSON.parse(expectedResult.stdout);
    assert.equal(expectedSeam.oracle,'delivered-fine-physical-svg-exact-pixel-footprints');
    assert.ok(expectedSeam.rows.length>=8);
    const seamPixels = await page.evaluate(async ({samples, width, height}) => {
      const svg = new DOMParser().parseFromString(WorldAtlasGlobe.surface, 'image/svg+xml').documentElement;
      svg.setAttribute('width', width); svg.setAttribute('height', height);
      const image = new Image();
      // This independent surface-only oracle is not a product texture
      // composition. Keep the observer scoped to the renderer's Image.src.
      image.setAttribute('src','data:image/svg+xml;charset=utf-8,' + encodeURIComponent(new XMLSerializer().serializeToString(svg)));
      await image.decode();
      const canvas = document.createElement('canvas'); canvas.width=width; canvas.height=height;
      const context = canvas.getContext('2d', {willReadFrequently:true});
      context.drawImage(image, 0, 0);
      return samples.rows.map(sample => {
        const y = Math.floor((sample.row + .5) / samples.height * height);
        return {...sample, textureRow:y,
          left:Array.from(context.getImageData(0,y,1,1).data),
          right:Array.from(context.getImageData(width-1,y,1,1).data)};
      });
    }, {samples:expectedSeam, width:initial.textureWidth, height:initial.textureHeight});
    for (const sample of seamPixels) {
      for (const side of ['left','right']) {
        assert.deepEqual(sample[side], [...sample.color,255], `${side} seam pixel at source row ${sample.row} must keep its physical ocean band`);
      }
    }
    await page.waitForTimeout(250);
    const idleBefore = await page.evaluate(() => WorldAtlasGlobeDiagnostics().frames);
    await page.waitForTimeout(400);
    assert.equal(await page.evaluate(() => WorldAtlasGlobeDiagnostics().frames), idleBefore, 'stationary globe must not redraw');
    await page.screenshot({path: path.join(root, 'check-globe-terrain-clear.png')});
    const themes = await page.locator('#globe-theme option').evaluateAll(options => options.map(option => option.value));
    assert.deepEqual(themes,expectedThemes);
    for (const theme of themes) {
      await page.locator('#globe-theme').selectOption(theme);
      await page.waitForFunction(key => WorldAtlasGlobeDiagnostics().appliedTheme === key && WorldAtlasGlobeDiagnostics().textures === 1, theme, {timeout: 120000});
      assert.equal(await page.evaluate(() => WorldAtlasGlobeDiagnostics().textures), 1, 'retain only the active GPU texture');
      assert.ok((await page.evaluate(()=>WorldAtlasGlobeDiagnostics())).cachedThemeSources<=4,
        'Retain at most four fetched thematic source documents');
    }
    // Exercise cancellation while vector images are decoding, including return
    // to the active theme. The final selection must own both status and texture.
    await page.evaluate(keys => {
      const selector = document.querySelector('#globe-theme');
      for (const key of [...keys].reverse().concat(keys.at(-1))) {
        selector.value = key;
        selector.dispatchEvent(new Event('change'));
      }
    }, themes);
    await page.waitForFunction(key => WorldAtlasGlobeDiagnostics().appliedTheme === key && WorldAtlasGlobeDiagnostics().textures === 1, themes.at(-1), {timeout: 120000});
    const rasterizations=await page.evaluate(()=>observedGlobeRasterizations);
    assert.deepEqual(rasterizations.slice(0,13).map(value=>value.theme),expectedThemes,
      'Each globe selector rasterizes its actual published thematic source');
    for(const raster of rasterizations){
      assert.equal(raster.width,initial.textureWidth);assert.equal(raster.height,initial.textureHeight);
      assert.equal(raster.landClipDefinitions,1);
      assert.equal(raster.thematicPartitions,raster.theme==='terrain'?0:1);
      assert.ok(raster.clipMatchesPublishedSurface&&raster.allThematicPartitionsClipped&&raster.sharedInkIsLastLayer,
        'Actual texture painting shares the exact authoritative coast and one common ink layer');
    }
    await page.waitForTimeout(300);
    assert.ok((await page.locator('#globe-status').textContent()).includes('同源矢量地形'));
    await page.screenshot({path: path.join(root, 'check-globe-clear.png')});
    // The texture's ±180° join must also be reviewed in a real globe view.
    await page.mouse.move(600, 450);
    await page.mouse.down();
    await page.mouse.move(825, 450, {steps: 8});
    await page.mouse.up();
    await page.waitForTimeout(200);
    await page.screenshot({path: path.join(root, 'check-globe-clear-seam.png')});
    await page.locator('#globe-reset').click();
    await page.locator('#globe-graticule').check();
    await page.waitForFunction(() => WorldAtlasGlobeDiagnostics().graticule);
    await page.locator('canvas').click({position: {x: 600, y: 450}});
    assert.match(await page.locator('#globe-coordinates').textContent(), /°/);
    await page.mouse.move(600, 450);
    await page.mouse.wheel(0, -450);
    await page.waitForTimeout(300);
    await page.screenshot({path: path.join(root, 'check-globe-clear-zoom.png')});
    const final = await page.evaluate(() => WorldAtlasGlobeDiagnostics());
    await page.evaluate(() => window.dispatchEvent(new PageTransitionEvent('pagehide',{persisted:false})));
    const releasedTextures = await page.evaluate(() => WorldAtlasGlobeDiagnostics().textures);
    assert.equal(releasedTextures,0,'Leaving the globe releases its actual GPU texture');
    assert.deepEqual(errors, []);
    const report = {schema:'globe-clarity-browser-v2',status:'ok',initial, themes, rasterizations,
      seamPixelSamples:seamPixels.length,seamOracle:expectedSeam.oracle, idleFrames: idleBefore, final, releasedTextures, errors};
    await fs.writeFile(path.join(root, 'globe-clarity-check.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report));
  } finally {
    await browser.close();
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
