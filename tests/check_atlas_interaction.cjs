const assert = require('node:assert/strict');
const path = require('node:path');
const {chromium} = require('C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const source = path.resolve(__dirname, '../src/world_atlas/core/web/atlas-interaction.js');
const {wheelPixels} = require(source);
assert.equal(wheelPixels({deltaY:1,deltaMode:0}, 800), 1);
assert.equal(wheelPixels({deltaY:1,deltaMode:1}, 800), 16);
assert.equal(wheelPixels({deltaY:-1,deltaMode:2}, 800), -240);
const close = (a,b) => assert.ok(Math.abs(a-b)<1e-7, `${a} != ${b}`);

(async () => {
  const browser = await chromium.launch({channel:'msedge',headless:true});
  try {
    const page = await browser.newPage({viewport:{width:1000,height:800}});
    page.on('pageerror', error => console.error('Browser:',error.message));
    await page.setContent('<style>body{margin:0}#map{width:1000px;height:700px;touch-action:none;user-select:none}</style><div id="map" tabindex="0"><span id="city">真实地点</span></div><input id="search" value="可选择的搜索文字">');
    await page.addScriptTag({path:source});
    await page.evaluate(() => {
      window.camera = {scale:2,translateX:20,translateY:30,minimumScale:.5,maximumScale:64};
      window.taps = []; window.commits = 0; window.changes = 0;
      const map = document.querySelector('#map');
      window.zoomAt = (factor,x,y) => {
        const next = Math.min(camera.maximumScale,Math.max(camera.minimumScale,camera.scale*factor));
        camera.translateX = x-(x-camera.translateX)/camera.scale*next;
        camera.translateY = y-(y-camera.translateY)/camera.scale*next;
        camera.scale = next; changes++;
      };
      window.gestures = WorldAtlasInteraction.attach({viewport:map,
        getCamera:()=>({...camera}),setCamera:state=>{Object.assign(camera,state);changes++;},
        zoomAt,zoomBy:factor=>zoomAt(factor,500,350),reset:()=>{},
        onCommit:()=>commits++,onTap:event=>taps.push(event.target.id)});
    });
    // Wheel magnitude is respected, and the point under the cursor is invariant.
    const wheel = async delta => {
      await page.evaluate(delta => {
        camera = {scale:2,translateX:20,translateY:30,minimumScale:.5,maximumScale:64};
        window.previousChanges = changes;
        document.querySelector('#map').dispatchEvent(new WheelEvent('wheel',
          {deltaY:delta,deltaMode:0,clientX:310,clientY:220,bubbles:true,cancelable:true}));
      },delta);
      await page.waitForFunction(()=>changes>previousChanges);
      return page.evaluate(()=>({...camera}));
    };
    const large = await wheel(-120), small = await wheel(-1);
    close(large.scale,2*Math.exp(.3)); close(small.scale,2*Math.exp(.0025));
    for (const state of [large,small]) {
      close((310-state.translateX)/state.scale,145);
      close((220-state.translateY)/state.scale,95);
    }
    // A real captured mouse drag pans and cannot become a place/ruler tap.
    await page.mouse.move(5,8); await page.mouse.down(); await page.mouse.move(105,68,{steps:5}); await page.mouse.up();
    assert.equal(await page.evaluate(()=>taps.length),0);
    await page.mouse.click(12,8);
    assert.deepEqual(await page.evaluate(()=>taps),['city']);
    // Two pointers compose scale and centroid movement in a single camera step.
    const pinch = await page.evaluate(() => {
      camera = {scale:2,translateX:20,translateY:30,minimumScale:.5,maximumScale:64};
      const map = document.querySelector('#map');
      map.setPointerCapture = () => {};
      const event = (type,id,x,y) => map.dispatchEvent(new PointerEvent(type,
        {pointerId:id,clientX:x,clientY:y,button:0,pointerType:'touch',bubbles:true,cancelable:true}));
      event('pointerdown',11,200,200); event('pointerdown',12,400,200);
      const before = {...camera};
      event('pointermove',12,500,260);
      const after = {...camera};
      event('pointerup',11,200,200); event('pointerup',12,500,260);
      return {before,after,taps:taps.length};
    });
    close((300-pinch.before.translateX)/pinch.before.scale,(350-pinch.after.translateX)/pinch.after.scale);
    close((200-pinch.before.translateY)/pinch.before.scale,(230-pinch.after.translateY)/pinch.after.scale);
    assert.equal(pinch.taps,1);
    const doubled = await page.evaluate(() => {
      const before = {...camera}, previousCommits = commits;
      document.querySelector('#map').dispatchEvent(new MouseEvent('dblclick',
        {clientX:320,clientY:280,bubbles:true,cancelable:true}));
      return {before,after:{...camera},committed:commits-previousCommits};
    });
    close(doubled.after.scale,doubled.before.scale*2);
    close((320-doubled.after.translateX)/doubled.after.scale,(320-doubled.before.translateX)/doubled.before.scale);
    close((280-doubled.after.translateY)/doubled.after.scale,(280-doubled.before.translateY)/doubled.before.scale);
    assert.equal(doubled.committed,1,'Double-click zoom must commit its new viewpoint');
    // Text input still supports normal selection, while map text does not select.
    await page.locator('#search').focus(); await page.keyboard.press('Control+A');
    assert.ok(await page.locator('#search').evaluate(input=>input.selectionEnd-input.selectionStart===input.value.length));
    await page.waitForTimeout(180);
    const idle = await page.evaluate(()=>({changes,commits}));
    await page.waitForTimeout(250);
    assert.deepEqual(await page.evaluate(()=>({changes,commits})),idle);
    await page.evaluate(()=>gestures.destroy());
    console.log('Map interaction: wheel units/anchor, drag/tap, pinch centroid, input selection and idle passed.');
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
