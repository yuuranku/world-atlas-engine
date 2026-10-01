const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const http = require('node:http');
const path = require('node:path');
const {chromium} = require('C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const source = path.resolve(__dirname, '../src/world_atlas/core/web/atlas-ruler.js');
const {greatCircleDistanceKm:distance, greatCircleArc:arc} = require(source);
const close = (actual, expected, tolerance = 1e-8) => assert.ok(Math.abs(actual-expected) <= tolerance,
  `${actual} differs from ${expected} by more than ${tolerance}`);

// The distances use the saved world's radius, not Earth's default radius.
const radius = 6400;
close(distance([0, 0], [90, 0], radius), Math.PI*radius/2);
close(distance([179, 0], [-179, 0], radius), 2*Math.PI/180*radius);
close(distance([0, 90], [37, -90], radius), Math.PI*radius);
close(distance([35, 90], [-124, 90], radius), 0, 0);
close(distance([20, -35], [20, -35], radius), 0, 0);
close(distance([0, 0], [90, 0], 2*radius), 2*distance([0, 0], [90, 0], radius));
for (const [a, b] of [[[0, 0], [180, 0]], [[45, 30], [-135, -30]],
  [[0, 0], [179.999999999, .000000001]], [[-90, 80], [90, 80]],
  [[179, 62], [-179, 61]], [[0, 0], [1e-9, 1e-9]]]) {
  close(distance(a, b, radius), distance(b, a, radius), 1e-9);
  const forward = arc(a, b), backward = arc(b, a).reverse();
  assert.equal(forward.length, backward.length);
  assert.ok(forward.every(point => point.every(Number.isFinite) && Math.abs(point[1]) <= 90));
  assert.deepEqual(forward[0], a); assert.deepEqual(forward.at(-1), b);
  let accumulated = 0;
  forward.forEach((point, i) => {
    close(distance(point, backward[i], radius), 0, 1e-6);
    if (i) accumulated += distance(forward[i-1], point, radius);
  });
  close(accumulated, distance(a, b, radius), 1e-6);
}
assert.deepEqual(arc([5, 6], [5, 6]), [[5, 6], [5, 6]]);
assert.throws(() => distance([0, 91], [0, 0], radius), RangeError);
assert.throws(() => distance([0, NaN], [0, 0], radius), TypeError);
assert.throws(() => distance([0, 0], [0, 0], 0), RangeError);
assert.throws(() => arc([0, 0], [0, 0], {maxStepDegrees:0}), RangeError);

(async () => {
  let browser, server;
  try {
    const script = await fs.readFile(source);
    const html = `<!doctype html><html><head><meta charset="utf-8"><link rel="icon" href="data:"></head><body style="margin:0">
<svg xmlns="http://www.w3.org/2000/svg" id="map" width="720" height="360" viewBox="0 0 360 180">
<rect width="360" height="180" fill="#091627"/><g id="camera"><g id="ruler"/></g></svg>
<output id="distance"></output><script src="/atlas-ruler.js"></script><script>
window.changes=[];
window.ruler=WorldAtlasRuler.create({svgOverlay:document.getElementById('ruler'),worldWidth:360,radiusKm:6400,
mapNativeToGeo:point=>[point[0]-180,90-point[1]],onChange:state=>{
changes.push(state);document.getElementById('distance').textContent=state.totalKm.toFixed(3)+' km';}});
</script></body></html>`;
    server = http.createServer((request, response) => {
      response.setHeader('content-type', request.url === '/atlas-ruler.js' ? 'text/javascript; charset=utf-8' : 'text/html; charset=utf-8');
      response.end(request.url === '/atlas-ruler.js' ? script : html);
    });
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    browser = await chromium.launch({headless:true,
      executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
    const page = await browser.newPage({viewport:{width:760, height:420}});
    const errors = [];
    page.on('pageerror', error => errors.push(String(error)));
    await page.addInitScript(() => {
      window.rulerAnimationRequests = 0;
      window.rulerInputListeners = 0;
      const original = requestAnimationFrame;
      window.requestAnimationFrame = callback => {rulerAnimationRequests++; return original(callback);};
      const originalListener = EventTarget.prototype.addEventListener;
      EventTarget.prototype.addEventListener = function(type, ...arguments) {
        if (['resize','wheel','click','pointerdown','pointermove'].includes(type)) rulerInputListeners++;
        return originalListener.call(this, type, ...arguments);
      };
    });
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    const lifecycle = await page.evaluate(() => {
      const assertions = [], check = (value, message) => {if (!value) throw Error(message); assertions.push(message);};
      check(!ruler.active && !ruler.addNativePoint([180, 90]), 'inactive cannot add');
      check(ruler.start() && !ruler.start(), 'start only changes once');
      check(!ruler.addNativePoint([180, -1]) && !ruler.addNativePoint([180, 181])
        && !ruler.addNativePoint([NaN, 90]), 'polar band and invalid input rejected');
      check(ruler.addNativePoint([180, 90]) && ruler.addNativePoint([270, 90])
        && ruler.addNativePoint([270, 0]), 'three true stations added');
      const snapshot = ruler.state();
      check(snapshot.stations.length === 3 && snapshot.segments.length === 2, 'multi-point route');
      check(Math.abs(snapshot.totalKm - Math.PI*6400) < 1e-8, 'cumulative spherical distance');
      snapshot.stations[0].native[0] = -999;
      check(ruler.state().stations[0].native[0] === 180, 'snapshot cannot mutate route');
      check(ruler.undo() && Math.abs(ruler.state().totalKm-Math.PI*3200)<1e-8, 'undo subtracts last segment');
      check(ruler.finish() && !ruler.active && !ruler.addNativePoint([0, 0]), 'finish retains route and disables adding');
      check(ruler.state().stations.length === 2 && ruler.undo(), 'finished route remains editable');
      check(ruler.clear() && !ruler.clear() && ruler.state().totalKm === 0 && !ruler.active, 'clear zeroes finished route');
      ruler.start();ruler.addNativePoint([359, 90]);ruler.addNativePoint([1, 90]);
      check(ruler.active, 'clear and start semantics');
      return {assertions, state:ruler.state(), changes:changes.length};
    });
    close(lifecycle.state.totalKm, 2*Math.PI/180*radius);
    const wrap = await page.evaluate(() => {
      const data = document.querySelector('[data-ruler-part="arc"]').getAttribute('d');
      const parts = data.split('M').filter(Boolean).map(part => {
        const numbers = part.match(/-?\d+(?:\.\d+)?(?:e[+-]?\d+)?/gi).map(Number);
        return numbers.filter((value, index) => index%2 === 0);
      });
      return {parts, label:document.querySelector('[data-atlas-ruler]').getAttribute('aria-label'),
        titles:Array.from(document.querySelectorAll('[data-ruler-station] title'), node => node.textContent),
        distance:document.getElementById('distance').textContent,
        nonScaling:Array.from(document.querySelectorAll('[data-atlas-ruler] path'), node => node.getAttribute('vector-effect'))};
    });
    assert.equal(wrap.parts.length, 2, 'dateline must have two disjoint pieces');
    assert.ok(wrap.parts.every(part => Math.max(...part)-Math.min(...part) <= 1.0000001), 'no route crosses the unfolded world');
    assert.match(wrap.label, /2 个站点/); assert.match(wrap.label, /223\.402 km/);
    assert.match(wrap.distance, /223\.402 km/);
    assert.ok(wrap.titles[0].includes('179.000000°') && wrap.titles[1].includes('-179.000000°'));
    assert.ok(wrap.nonScaling.every(value => value === 'non-scaling-stroke'));

    const edgeCases = await page.evaluate(() => {
      const paths = () => document.querySelector('[data-ruler-part="arc"]')?.getAttribute('d').split('M')
        .filter(Boolean).map(part => {
          const numbers=part.match(/-?\d+(?:\.\d+)?(?:e[+-]?\d+)?/gi).map(Number);
          return Array.from({length:numbers.length/2},(_,i)=>numbers.slice(i*2,i*2+2));
        }) || [];
      ruler.clear();ruler.addNativePoint([359,28]);ruler.addNativePoint([1,29]);
      const curvedSeam=paths();
      ruler.clear();ruler.addNativePoint([90,10]);ruler.addNativePoint([270,10]);
      const throughPole=paths();
      ruler.clear();ruler.addNativePoint([0,45]);ruler.addNativePoint([360,45]);
      return {curvedSeam, throughPole, identicalSeam:ruler.state(), identicalSeamPaths:paths()};
    });
    assert.equal(edgeCases.curvedSeam.length, 2);
    const lastBeforeSeam=edgeCases.curvedSeam[0].at(-1), firstAfterSeam=edgeCases.curvedSeam[1][0];
    assert.equal(lastBeforeSeam[0],360); assert.equal(firstAfterSeam[0],0);
    close(lastBeforeSeam[1],firstAfterSeam[1],0);
    const seamGeo=[180,90-lastBeforeSeam[1]];
    close(distance([179,62],seamGeo,radius)+distance(seamGeo,[-179,61],radius),
      distance([179,62],[-179,61],radius),1e-7);
    assert.equal(edgeCases.throughPole.length, 2, 'a pole crossing must not draw across longitude columns');
    assert.ok(edgeCases.throughPole.every(part=>Math.max(...part.map(point=>point[0]))-Math.min(...part.map(point=>point[0]))<1e-8));
    close(edgeCases.identicalSeam.totalKm,0,0);
    assert.deepEqual(edgeCases.identicalSeamPaths,[], 'coincident seam coordinates must not create a false arc');

    // Actual SVG raster pixels verify the point radius, not just its attribute.
    const markerSizes = await page.evaluate(async () => {
      ruler.clear();ruler.addNativePoint([180, 90]);
      const sizes = [];
      for (const scale of [1, 5, 8192]) {
        document.getElementById('camera').setAttribute('transform', `translate(180 90) scale(${scale}) translate(-180 -90)`);
        const svg = new XMLSerializer().serializeToString(document.getElementById('map'));
        const url = URL.createObjectURL(new Blob([svg], {type:'image/svg+xml'}));
        const image = new Image();image.src=url;await image.decode();
        const canvas=document.createElement('canvas');canvas.width=720;canvas.height=360;
        const context=canvas.getContext('2d');context.drawImage(image,0,0);URL.revokeObjectURL(url);
        const pixel=context.getImageData(360,160,1,40).data;
        let whiteOrBlue=0,blue=0;
        for(let i=0;i<pixel.length;i+=4){
          if(pixel[i]>200 && pixel[i+1]>200 && pixel[i+2]>200) whiteOrBlue++;
          if(pixel[i]<80 && pixel[i+1]>65 && pixel[i+2]>140){blue++;whiteOrBlue++;}
        }
        sizes.push({scale,whiteOrBlue,blue});
      }
      return sizes;
    });
    assert.ok(markerSizes.every(size => size.whiteOrBlue >= 6 && size.whiteOrBlue <= 10), JSON.stringify(markerSizes));
    assert.ok(markerSizes.every(size => size.blue >= 4 && size.blue <= 6), JSON.stringify(markerSizes));
    assert.equal(new Set(markerSizes.map(size => size.whiteOrBlue)).size, 1);
    assert.equal(new Set(markerSizes.map(size => size.blue)).size, 1);
    await page.evaluate(() => {
      ruler.finish();window.rulerIdleMutations=0;
      window.rulerObserver=new MutationObserver(records=>rulerIdleMutations+=records.length);
      rulerObserver.observe(document.getElementById('ruler'),{subtree:true,childList:true,attributes:true,characterData:true});
    });
    await page.waitForTimeout(250);
    const idle = await page.evaluate(() => {
      rulerIdleMutations+=rulerObserver.takeRecords().length;rulerObserver.disconnect();
      const state={mutations:rulerIdleMutations,animationRequests:rulerAnimationRequests,inputListeners:rulerInputListeners};
      ruler.destroy();ruler.destroy();
      if(document.querySelector('[data-atlas-ruler]') || ruler.start()) throw Error('destroy must remove overlay and disable controller');
      const invalidGeo = WorldAtlasRuler.create({svgOverlay:document.getElementById('ruler'),worldWidth:360,radiusKm:6400,mapNativeToGeo:()=>[0,91]});
      invalidGeo.start();if(invalidGeo.addNativePoint([180,90])) throw Error('illegal mapped latitude accepted');invalidGeo.destroy();
      return state;
    });
    assert.deepEqual(idle, {mutations:0, animationRequests:0,inputListeners:0});
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({status:'ok', radiusKm:radius, knownDistances:true, lifecycle:lifecycle.assertions,
      wrappedParts:wrap.parts, markerSizes, idle, errors}));
  } finally {
    if (browser) await browser.close();
    if (server) await new Promise(resolve => server.close(resolve));
  }
})().catch(error => {console.error(error);process.exitCode=1;});
