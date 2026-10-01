const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const http = require('node:http');
const os = require('node:os');
const path = require('node:path');
const {spawnSync} = require('node:child_process');
const {chromium} = require('C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

(async () => {
  const project = path.resolve(__dirname, '..');
  const temporary = await fs.mkdtemp(path.join(os.tmpdir(), 'world-atlas-globe-lazy-'));
  let browser, server;
  try {
    const fixture = spawnSync(path.join(project, '.venv', 'Scripts', 'python.exe'), ['-c', `
from pathlib import Path
import sys
from world_atlas.core.globe import write_globe
themes=['terrain','climate','biome','watershed','potential','habitability','vegetation','population','civilizations','languages','religions','political','provinces']
root='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 8">{}</svg>'
base=root.format('<rect id="base-ground" width="16" height="8" fill="#aad080"/>')
ink=root.format('<path id="common-ink" d="M1.25,2.5 L3.125,4.75" stroke="#123456"/>')
textures={key:root.format('<path id="overlay-'+key+'" d="M1.23456789,2.34567891 L3.45678912,4.56789123" fill="#ffc080"/>') for key in themes}
write_globe(Path(sys.argv[1]),surface=base,ink=ink,textures=textures,grid_digest='fixture',world_name='测试')
`, temporary], {encoding:'utf8', env:{...process.env, PYTHONPATH:path.join(project, 'src')}});
    assert.equal(fixture.status, 0, fixture.stderr);
    const requests = [];
    let delayReligions = false;
    server = http.createServer(async (request, response) => {
      const name = path.basename(new URL(request.url, 'http://localhost').pathname);
      if (name.startsWith('globe-theme-')) requests.push(name);
      try {
        const bytes = await fs.readFile(path.join(temporary, name));
        const send = () => {
          response.setHeader('content-type', name.endsWith('.svg') ? 'image/svg+xml' : name.endsWith('.js') ? 'text/javascript' : 'text/html');
          response.end(bytes);
        };
        if (delayReligions && name === 'globe-theme-religions.svg') setTimeout(send, 500);
        else send();
      } catch {
        response.statusCode = 404; response.end();
      }
    });
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    browser = await chromium.launch({headless:true,
      executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
      args:['--enable-unsafe-swiftshader']});
    const page = await browser.newPage({viewport:{width:640,height:480}});
    const errors = [];
    page.on('pageerror', error => errors.push(String(error)));
    await page.addInitScript(() => {
      const originalFetch = window.fetch;
      window.fixtureThemeFetches = [];
      window.fetch = (url, ...arguments) => {
        if (String(url).startsWith('globe-theme-')) fixtureThemeFetches.push(url);
        return originalFetch(url, ...arguments);
      };
      // Test request/cache/composition behavior with small GPU allocations.
      // The production DPR/8K regression independently tests real resolution.
      for (const Class of [window.WebGLRenderingContext, window.WebGL2RenderingContext]) {
        if (!Class) continue;
        const original = Class.prototype.getParameter;
        Class.prototype.getParameter = function(parameter) {
          return parameter === this.MAX_TEXTURE_SIZE ? 256 : original.call(this, parameter);
        };
      }
      const descriptor = Object.getOwnPropertyDescriptor(HTMLImageElement.prototype, 'src');
      Object.defineProperty(HTMLImageElement.prototype, 'src', {...descriptor, set(value) {
        if (value.startsWith('data:image/svg+xml;charset=utf-8,')) window.lastGlobeVectorSource = decodeURIComponent(value.split(',').slice(1).join(','));
        descriptor.set.call(this, value);
      }});
    });
    const baseURL=`http://127.0.0.1:${server.address().port}/globe.html`;
    await page.goto(baseURL, {waitUntil:'load'});
    await page.waitForFunction(() => window.WorldAtlasGlobeDiagnostics?.().appliedTheme === 'terrain'&&WorldAtlasGlobeDiagnostics().textures===1);
    assert.deepEqual(requests, ['globe-theme-terrain.svg'], 'initial view must fetch only its chosen theme');
    const options = await page.locator('#globe-theme option').evaluateAll(nodes => nodes.map(node => ({value:node.value,label:node.textContent})));
    assert.equal(options.length, 13);
    assert.ok(options.every(option => /[\u4e00-\u9fff]/.test(option.label)));
    async function select(theme) {
      await page.locator('#globe-theme').selectOption(theme);
      await page.waitForFunction(key => WorldAtlasGlobeDiagnostics().appliedTheme === key&&WorldAtlasGlobeDiagnostics().textures===1, theme);
      const diagnostics = await page.evaluate(() => WorldAtlasGlobeDiagnostics());
      assert.equal(diagnostics.textures, 1);
      assert.ok(diagnostics.cachedThemeSources <= 4);
    }
    for (const theme of ['habitability','vegetation','population','provinces','climate','watershed']) await select(theme);
    const terrainRequests = await page.evaluate(() => fixtureThemeFetches.filter(name => name === 'globe-theme-terrain.svg').length);
    await select('terrain');
    assert.equal(await page.evaluate(() => fixtureThemeFetches.filter(name => name === 'globe-theme-terrain.svg').length), terrainRequests+1, 'an evicted source is fetched again, allowing the HTTP cache');
    delayReligions = true;
    await Promise.all([
      page.waitForRequest(request => request.url().endsWith('/globe-theme-religions.svg')),
      page.locator('#globe-theme').selectOption('religions'),
    ]);
    await select('vegetation');
    await page.waitForTimeout(650);
    assert.equal(await page.evaluate(() => WorldAtlasGlobeDiagnostics().appliedTheme), 'vegetation', 'an obsolete request cannot replace the latest theme');
    assert.ok((await page.locator('#globe-status').textContent()).includes('同源矢量地形'));
    delayReligions = false;
    for (const {value} of options) await select(value);
    const composition = await page.evaluate(() => {
      const svg = new DOMParser().parseFromString(lastGlobeVectorSource, 'image/svg+xml').documentElement;
      return {viewBox:svg.getAttribute('viewBox'), width:svg.getAttribute('width'),
        base:!!svg.querySelector('#base-ground'), ink:!!svg.querySelector('#common-ink'),
        path:svg.querySelector('#overlay-provinces').getAttribute('d'),
        lastChild:svg.lastElementChild.querySelector('#common-ink')?.id};
    });
    assert.deepEqual(composition, {viewBox:'0 0 16 8',width:'256',base:true,ink:true,
      path:'M1.23456789,2.34567891 L3.45678912,4.56789123',lastChild:'common-ink'});
    const before = await page.evaluate(() => fixtureThemeFetches.length);
    await select('provinces');
    assert.equal(await page.evaluate(() => fixtureThemeFetches.length), before, 'returning to the applied theme performs no fetch');
    // A material may retain its disposed map object. The diagnostic must report
    // the actual released GPU resource, rather than that object's presence.
    await page.evaluate(()=>window.dispatchEvent(new PageTransitionEvent('pagehide',{persisted:false})));
    assert.equal(await page.evaluate(()=>WorldAtlasGlobeDiagnostics().textures),0,'Disposing the globe releases the measured GPU texture');
    const initialThemes=[];
    for(const [query,expected] of [['?theme=vegetation','vegetation'],['?theme=political','political'],
      ['?theme=provinces','provinces'],['?theme=none','terrain'],['?theme=__proto__','terrain'],['?theme=unknown','terrain']]){
      await page.goto(baseURL+query,{waitUntil:'load'});
      await page.waitForFunction(key=>window.WorldAtlasGlobeDiagnostics?.().appliedTheme===key&&WorldAtlasGlobeDiagnostics().textures===1,expected);
      assert.equal(await page.locator('#globe-theme').inputValue(),expected);
      assert.deepEqual(await page.evaluate(()=>fixtureThemeFetches),[`globe-theme-${expected}.svg`],
        'An initial theme uses the published inventory and fetches only the selected source');
      initialThemes.push({query,applied:expected,textures:(await page.evaluate(()=>WorldAtlasGlobeDiagnostics())).textures});
    }
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({status:'ok',themes:13,initialThemeRequests:1,
      cachedThemeSources:(await page.evaluate(() => WorldAtlasGlobeDiagnostics())).cachedThemeSources,
      textures:1,actualGpuDisposal:true,initialThemes,latestSelectionPreserved:true,exactOverlayGeometry:true}));
  } finally {
    await browser?.close();
    await new Promise(resolve => server ? server.close(resolve) : resolve());
    const relative = path.relative(os.tmpdir(), temporary);
    assert.ok(relative && !relative.startsWith('..') && !path.isAbsolute(relative));
    await fs.rm(temporary, {recursive:true,force:true});
  }
})().catch(error => {console.error(error);process.exitCode=1;});
