/* Verify real image density, camera reuse, races and LOD handoff in the browser. */
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const {chromium} = require('C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

(async () => {
  const review = path.resolve(process.argv[2]), baseURL = process.argv[3];
  const browser = await chromium.launch({headless:true,
    executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
  const results = [], errors = [];
  try {
    for (const pixelRatio of [1, 2]) {
      const page = await browser.newPage({viewport:{width:1674,height:1249}, deviceScaleFactor:pixelRatio});
      page.on('pageerror',error=>errors.push(String(error)));
      const settle = () => page.waitForFunction(() => {
        const overview = overviewManager.stats(), tiles = tileManager.stats();
        return overview.currentKey && overview.currentKey===overview.wantedKey
          && overview.pendingRasters===0 && overview.queuedRasters===0
          && tiles.pendingRequests===0 && tiles.queuedTiles===0;
      },null,{timeout:120000});
      const focus = (zoom, x=300, y=650) => page.evaluate(({zoom,x,y}) => {
        scale=zoom;translateX=viewport.clientWidth/2-x*atlasScaleX*scale;
        translateY=viewport.clientHeight/2-(y*atlasScaleY+polarSceneBand)*scale;
        constrainCamera();applyTransform();scheduleAdaptiveUpdate(true);
      },{zoom,x,y});
      const quality = () => page.evaluate(async () => {
        const node=document.querySelector('[data-atlas-overview-refinement]');
        const image=new Image();image.src=node.getAttribute('href');await image.decode();
        const visible = id => getComputedStyle(document.getElementById(id)).display!=='none';
        const nativeWidth=Number(node.getAttribute('width')),nativeHeight=Number(node.getAttribute('height'));
        return {zoom:scale,screenDensity:scale*Math.max(atlasScaleX,atlasScaleY)*window.devicePixelRatio,
          density:Math.min(image.naturalWidth/nativeWidth,image.naturalHeight/nativeHeight),
          pixelWidth:image.naturalWidth,pixelHeight:image.naturalHeight,
          covered:overviewManager.stats().viewportCovered,
          overviewVisible:visible('overview-world'),detailVisible:visible('detail-world')};
      });
      await page.goto(new URL('index.html?revision=clarity&zoom=5.5&x=300&y=650',baseURL).href,
        {waitUntil:'load',timeout:120000});
      await settle();
      for (const zoom of [5.5,2.25,3.9,7.9]) {
        await focus(zoom);await settle();
        const state=await quality();
        assert.equal(state.covered,true);assert.ok(state.density>=state.screenDensity);
        assert.equal(state.overviewVisible,true);assert.equal(state.detailVisible,false);
        results.push({pixelRatio,...state});
        if (zoom===5.5) await page.screenshot({path:path.join(review,`check-v100-overview-crisp-dpr${pixelRatio}.png`)});
      }
      const before=await page.evaluate(()=>overviewManager.stats());
      await focus(7.9,302,650);await settle();
      assert.equal(await page.evaluate(()=>overviewManager.stats().viewportRasterizations),before.viewportRasterizations,
        'a nearby pan reuses the already sharp buffer');
      await focus(7.9,520,650);await settle();
      assert.equal((await quality()).covered,true,'distant panning sharpens the new region');
      const rendered=await page.evaluate(()=>overviewManager.stats().viewportRasterizations);
      await page.waitForTimeout(350);
      assert.equal(await page.evaluate(()=>overviewManager.stats().viewportRasterizations),rendered,'idle does not repaint');
      await focus(5.5,300,650);await settle();
      for(const theme of ['political','provinces','vegetation']) {
        await page.evaluate(theme=>{activeTheme=theme;updateThematicState();},theme);await settle();
        const state=await quality();assert.ok(state.density>=state.screenDensity);assert.equal(state.covered,true);
      }
      // Superseded crop and thematic requests must not overwrite the last choice.
      await page.evaluate(()=>{activeTheme='population';updateThematicState();
        scale=3.25;applyTransform();activeTheme='none';updateThematicState();});
      await focus(5.5,300,650);await settle();
      assert.equal(await page.evaluate(()=>JSON.parse(overviewManager.stats().currentKey)[1]),'none');
      assert.equal((await quality()).covered,true);
      await focus(6.5);await settle();
      assert.equal(await page.evaluate(()=>tileManager.stats().baseTilesReady),true,'regional tiles preload before the handoff');
      await focus(8.25);await settle();
      assert.equal(await page.locator('#overview-world').evaluate(node=>getComputedStyle(node).display==='none'),true);
      assert.equal(await page.locator('#detail-world').evaluate(node=>getComputedStyle(node).display!=='none'),true);
      // Delay an uncached vector region and verify that its previous base stays visible.
      await page.route('**/tiles/regional/*.json',async route=>{
        await new Promise(resolve=>setTimeout(resolve,150));await route.continue();
      });
      await focus(5.5,1650,450);await settle();
      await focus(8.25,1650,450);
      assert.equal(await page.locator('#overview-world').evaluate(node=>getComputedStyle(node).display!=='none'),true,
        'handoff retains a painted base while vector tiles are pending');
      await settle();
      assert.equal(await page.evaluate(()=>tileManager.stats().baseTilesReady),true);
      assert.equal(await page.locator('#overview-world').evaluate(node=>getComputedStyle(node).display==='none'),true);
      await page.unroute('**/tiles/regional/*.json');
      await focus(5.5,300,650);await settle();
      await focus(.8);await settle();
      assert.equal(await page.locator('[data-atlas-overview-raster]').evaluate(node=>getComputedStyle(node).display!=='none'),true,
        'zooming out restores the world preview');
      await page.close();
    }
    assert.deepEqual(errors,[]);
    const report={status:'passed',checks:['screen-density PNGs','DPR1 and DPR2','cached pans','idle',
      'theme changes','superseded requests','tile preloading','no blank LOD handoff','zoom-out'],results,errors};
    await fs.writeFile(path.join(review,'overview-clarity-check.json'),JSON.stringify(report,null,2)+'\n');
    process.stdout.write(JSON.stringify({status:report.status,checks:report.checks,
      results:results.map(({pixelRatio,zoom,density,screenDensity})=>({pixelRatio,zoom,density,screenDensity})),errors})+'\n');
  } finally {await browser.close();}
})().catch(error=>{process.stderr.write(error.stack+'\n');process.exitCode=1;});
