// Exercise the independent interface with genuine saved-model test records.
const assert=require('node:assert/strict');
const fs=require('node:fs/promises');
const http=require('node:http');
const path=require('node:path');
const {spawnSync}=require('node:child_process');
const {chromium}=require('C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

(async()=>{
  const project=path.resolve(__dirname,'..');
  let browser,server;
  try{
    const generated=spawnSync(path.join(project,'.venv/Scripts/python.exe'),['-c',`
import json
from test_atlas_ui import saved_fixture,saved_locations
from world_atlas.core.atlas_ui import build_place_index,ui_markup,THEMES
grid,society=saved_fixture()
legends={key:'<div class="legend"><span class="swatch" style="background:#336699"></span>'+label+'</div>' for key,label in THEMES}
legends.update(monsoon='<p>四季相同降水色标</p>',tectonic='<p>源板块边界</p>')
print(json.dumps({'places':build_place_index(grid,society,settlement_locations=saved_locations(society)),
'markup':ui_markup(world_name='界面行为测试',summary='2 国家 · 2 省份',legends=legends,info_html='<p>保存模型与坐标说明</p>')},ensure_ascii=False))
`],{encoding:'utf8',env:{...process.env,PYTHONUTF8:'1',PYTHONIOENCODING:'utf-8',PYTHONPATH:path.join(project,'src')+path.delimiter+path.join(project,'tests')}});
    assert.equal(generated.status,0,generated.stderr);
    const fixture=JSON.parse(generated.stdout);
    const script=await fs.readFile(path.join(project,'src/world_atlas/core/web/atlas-ui.js'));
    const css=await fs.readFile(path.join(project,'src/world_atlas/core/web/atlas-ui.css'));
    const html=`<!doctype html><html><head><meta charset="utf-8"><link rel="icon" href="data:"><link rel="stylesheet" href="/atlas-ui.css"></head><body><main><section id="viewport" tabindex="0"><svg id="map-canvas"></svg></section>${fixture.markup}</main><script src="/atlas-ui.js"></script><script>
window.actionLog=[];window.placeLoads=0;window.fixturePlaces=${JSON.stringify(fixture.places)};
window.mapState={view:'physical',theme:'none',season:'vernal'};
window.ui=WorldAtlasUI.create({places:async()=>{placeLoads++;await new Promise(resolve=>setTimeout(resolve,20));return fixturePlaces;},
onSelectPlace:(place,options)=>actionLog.push({action:'select',id:place.id,restore:options.restore,center:options.center,generate:options.generate}),
onRegenerateCity:place=>actionLog.push({action:'regenerate',id:place.id}),
onClosePlace:options=>actionLog.push({action:'close',restore:options.restore}),
onThemeChange:theme=>{actionLog.push({action:'theme',theme});mapState.theme=theme;ui.sync(mapState);},
onViewChange:view=>{actionLog.push({action:'view',view});mapState.view=view;ui.sync(mapState);},
onSeasonChange:season=>{actionLog.push({action:'season',season});mapState.season=season;ui.sync(mapState);},
onLayerChange:(layer,enabled)=>actionLog.push({action:'layer',layer,enabled}),onZoom:action=>actionLog.push({action:'zoom',type:action}),
onRulerAction:action=>actionLog.push({action:'ruler',type:action})});
</script></body></html>`;
    server=http.createServer((request,response)=>{
      const resource=request.url.split('?')[0];
      response.setHeader('content-type',resource.endsWith('.js')?'text/javascript; charset=utf-8':resource.endsWith('.css')?'text/css; charset=utf-8':'text/html; charset=utf-8');
      response.end(resource==='/atlas-ui.js'?script:resource==='/atlas-ui.css'?css:html);
    });
    await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
    browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
    const page=await browser.newPage({viewport:{width:1280,height:850}});
    const errors=[];page.on('pageerror',error=>errors.push(String(error)));
    const url=`http://127.0.0.1:${server.address().port}/`;
    await page.goto(url);
    assert.equal(await page.locator('#city-focus').count(),0,'obsolete native select must be removed');
    assert.equal(await page.locator('[data-theme-button]').count(),13);
    await page.locator('#city-search').fill('旧东');
    await page.waitForFunction(()=>document.querySelector('#city-results').children.length>=4);
    const resultKinds=await page.locator('#city-results [data-place-id]').evaluateAll(nodes=>nodes.map(node=>node.dataset.placeId.split(':')[0]));
    assert.deepEqual([...new Set(resultKinds)].sort(),['city','geography','province','state']);
    assert.equal(await page.evaluate(()=>actionLog.length),0,'searching must not focus the map');
    await page.locator('#city-search').press('ArrowUp');
    const active=await page.evaluate(()=>document.getElementById(document.querySelector('#city-search').getAttribute('aria-activedescendant')).dataset.placeId);
    const last=await page.locator('#city-results li').last().getAttribute('data-place-id');
    assert.equal(active,last,'first ArrowUp must select the last result');
    await page.locator('#city-search').press('Escape');
    assert.equal(await page.locator('#city-search').getAttribute('aria-expanded'),'false');
    await page.locator('#city-search').fill('旧东港');
    await page.waitForFunction(()=>document.querySelector('#city-results').children.length===1);
    await page.locator('#city-search').press('ArrowDown');await page.locator('#city-search').press('Enter');
    await page.waitForFunction(()=>ui.selectedPlaceId==='city:east-port');
    assert.equal(await page.locator('#city-map-section').isVisible(),true,'the selected city has a map below its summary');
    assert.equal(await page.locator('#place-extra').getAttribute('open'),null,'administrative facts initially stay compact');
    await page.locator('#place-extra summary').click();
    const card=await page.locator('#place-card').innerText();
    assert.match(card,/20,000–40,000 人/);assert.match(card,/旧东府/);assert.match(card,/首都/);assert.match(card,/旧日教/);assert.match(card,/112\.500°W/);
    assert.equal(await page.evaluate(()=>actionLog.at(-1).center),true,'search selections must request an appropriate view');
    await page.evaluate(()=>ui.selectPlace('city:east-port',{center:false}));
    assert.equal(await page.evaluate(()=>actionLog.at(-1).center),false,'a map click must preserve the current zoom');
    await page.locator('#place-recenter').click();
    assert.equal(await page.evaluate(()=>actionLog.at(-1).action),'select','the location action focuses the world map');
    assert.equal(await page.evaluate(()=>actionLog.at(-1).id),'city:east-port');
    assert.equal(await page.evaluate(()=>actionLog.at(-1).generate),false,'locating an existing city must not regenerate its map');
    await page.locator('#place-regenerate').click();
    assert.equal(await page.evaluate(()=>actionLog.at(-1).action),'regenerate');
    const selections=()=>page.evaluate(()=>actionLog.filter(item=>item.action==='select').length);
    const beforeClose=await selections();await page.locator('#place-close').click();
    assert.equal(await selections(),beforeClose,'closing a place card must not refocus the map');
    assert.equal(await page.evaluate(()=>new URLSearchParams(location.search).has('place')),false);
    await page.goBack();await page.waitForFunction(()=>ui.selectedPlaceId==='city:east-port');
    assert.equal(await page.evaluate(()=>actionLog.at(-1).restore),true);
    await page.evaluate(()=>ui.selectPlace('state:1'));assert.match(await page.locator('#place-card').innerText(),/50,000–90,000 人/);
    assert.equal(await page.locator('#city-map-section').isVisible(),false,'other places retain their factual card');
    await page.goBack();await page.waitForFunction(()=>ui.selectedPlaceId==='city:east-port');
    await page.locator('#toggle-controls').click();
    assert.equal(await page.locator('#atlas-controls').evaluate(node=>node.inert),false);
    const beforeTheme=await selections();await page.locator('[data-theme-button="vegetation"]').click();
    await page.waitForFunction(()=>document.querySelector('[data-theme-button="vegetation"]').getAttribute('aria-pressed')==='true');
    assert.equal(await selections(),beforeTheme,'changing themes must keep the current place without focusing it');
    assert.equal(await page.evaluate(()=>ui.selectedPlaceId),'city:east-port');
    assert.equal(await page.locator('[data-legend-key="vegetation"]').isVisible(),true);
    const actionCount=await page.evaluate(()=>actionLog.length);
    await page.evaluate(()=>ui.sync({view:'physical',theme:'potential',season:'vernal',layers:{rivers:false}}));
    assert.equal(await page.evaluate(()=>actionLog.length),actionCount,'UI sync must not dispatch map actions');
    assert.equal(await page.locator('#toggle-rivers').isChecked(),false);
    await page.locator('#view-monsoon').click();await page.waitForFunction(()=>!document.querySelector('#monsoon-controls').hidden);
    await page.locator('#season-december').click();await page.locator('#toggle-monsoon-wind').uncheck();
    assert.equal(await page.evaluate(()=>actionLog.some(item=>item.action==='season'&&item.season==='december')),true);
    assert.equal(await page.evaluate(()=>actionLog.some(item=>item.action==='layer'&&item.layer==='monsoon-wind'&&!item.enabled)),true);
    assert.equal(await page.locator('[data-legend-key="monsoon"]').isVisible(),true);
    await page.locator('#view-tectonic').click();await page.waitForFunction(()=>!document.querySelector('#tectonic-controls').hidden);
    assert.equal(await page.locator('[data-legend-key="tectonic"]').isVisible(),true);
    await page.keyboard.press('Escape');assert.equal(await page.locator('#atlas-controls').evaluate(node=>node.inert),true);
    await page.locator('#ruler-toggle').click();assert.equal(await page.evaluate(()=>actionLog.at(-1).type),'start');
    await page.evaluate(()=>ui.setRulerState({active:true,stations:[{},{}],segments:[{distanceKm:.123}],totalKm:.123}));
    assert.equal(await page.locator('#ruler-summary').innerText(),'123 m');
    assert.match(await page.locator('#ruler-hint').innerText(),/2 个站点/);
    await page.locator('#ruler-finish').click();assert.equal(await page.evaluate(()=>actionLog.at(-1).type),'finish');
    await page.evaluate(()=>ui.setRulerState({active:false,stations:[{},{}],segments:[{distanceKm:.123}],totalKm:.123}));
    assert.equal(await page.locator('#ruler-card').isVisible(),true,'finish must keep the result');
    await page.locator('#zoom-in').click();assert.equal(await page.evaluate(()=>actionLog.at(-1).type),'in');
    await page.locator('#map-info-toggle').click();assert.equal(await page.locator('#map-info').isVisible(),true);await page.locator('#map-info-close').click();
    assert.equal(await page.evaluate(()=>placeLoads),1,'saved place data loads once');

    // A new page restores the factual place directly from its typed URL.
    await page.goto(url+'?place=geography%3Awest-river&theme=vegetation');
    await page.waitForFunction(()=>ui.selectedPlaceId==='geography:west-river');
    assert.match(await page.locator('#place-card').innerText(),/标注位置/);
    assert.equal(await page.evaluate(()=>actionLog[0].restore),true);
    assert.equal(await page.locator('[data-theme-button="vegetation"]').getAttribute('aria-pressed'),'true');
    await page.setViewportSize({width:390,height:844});
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth),390,'mobile shell must not overflow');
    await page.locator('#toggle-controls').click();await page.locator('[data-theme-button="population"]').click();
    await page.waitForFunction(()=>document.querySelector('#atlas-controls').hidden);
    assert.equal(await page.locator('#place-card').isVisible(),true);
    const beforeDestroy=await page.evaluate(()=>actionLog.length);await page.evaluate(()=>ui.destroy());
    await page.locator('#zoom-in').click();assert.equal(await page.evaluate(()=>actionLog.length),beforeDestroy,'destroy must detach the controller');
    assert.deepEqual(errors,[]);
    console.log(JSON.stringify({status:'ok',searchKinds:[...new Set(resultKinds)],keyboard:true,factualCard:true,history:true,
      contextualLegends:true,mapSelectionWithoutCentering:true,syncWithoutFeedback:true,mobileWidth:390,rulerSummary:'123 m',errors}));
  }finally{if(browser)await browser.close();if(server)await new Promise(resolve=>server.close(resolve));}
})().catch(error=>{console.error(error);process.exitCode=1;});
