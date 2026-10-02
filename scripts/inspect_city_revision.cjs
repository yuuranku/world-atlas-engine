const {chromium}=require('C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('node:fs/promises'),path=require('node:path');
const assert=require('node:assert/strict');
(async()=>{
  const output=path.resolve(process.argv[2]);await fs.mkdir(output,{recursive:true});
  const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
  try{
    const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
    page.on('pageerror',e=>errors.push(String(e)));
    await page.addInitScript(()=>{let value=20261002;crypto.getRandomValues=a=>{a[0]=value++;return a;};});
    const base=process.argv[3]||'http://127.0.0.1:18769/';
    await page.goto(base+'index.html?place=city%3Asettlement-0005');
    await page.waitForFunction(()=>window.WorldAtlasCityCharacter&&typeof cityMap!=='undefined'&&!cityMap.stats().loading&&cityMap.stats().buildings>0,{},{timeout:60000});
    const report={errors,cities:[]};
    for(const id of ['settlement-0024','settlement-0059','settlement-0373','settlement-0646','settlement-0019','settlement-0581','settlement-0721','settlement-0301','settlement-0109','settlement-0627','settlement-0004','settlement-0005','settlement-0013','settlement-0558','settlement-0677','settlement-0041','settlement-0585','settlement-0809','settlement-0667','settlement-0022','settlement-0829','settlement-0006','settlement-0010']){
      await page.evaluate(id=>ui.selectPlace('city:'+id),id);
      await page.waitForFunction(id=>cityMap.stats().recipeId===id&&!cityMap.stats().loading,id,{timeout:60000});
      await page.waitForTimeout(120);
      const stats=await page.evaluate(()=>{
        const r=cityMap.recipe,p=WorldAtlasCities.plan(r,{grid:{width:2176,height:1088}}),m=r.urban.gridCellKilometres;
        const b=p.buildings.flatMap(b=>b.points),xs=b.map(q=>(q.column-r.location.column)*m.column),ys=b.map(q=>(q.row-r.location.row)*m.row);
        return {...cityMap.stats(),form:r.localSite.form,civicKinds:[...new Set(p.landmarks.map(l=>l.kind))],
          siteType:r.siteType,foundationRelief:Math.max(0,...p.buildings.map(b=>b.foundation?.reliefMetres||0)),palacePrecincts:p.plazas.filter(p=>p.precinct).length,
          palaceParts:p.landmarks.filter(l=>/^royal-palace|^imperial-palace/.test(l.kind)).length,
          extentKm:{x:Math.max(...xs)-Math.min(...xs),y:Math.max(...ys)-Math.min(...ys)},ruralRoads:r.localSite.ruralRoads.length};
      });report.cities.push(stats);
      assert.ok(stats.buildings>0,id+' has buildings on usable ground');
      assert.ok(stats.foundationRelief<=3.5,id+' buildings have coherent foundations');
      if(['port','island-port','lake-port'].includes(stats.siteType))assert.ok(stats.harbor,id+' has an actual harbor');
      await page.locator('#place-card').screenshot({path:path.join(output,id+'-city.png')});
      await page.locator('#city-map-zoom-in').click();await page.locator('#city-map-zoom-in').click();
      await page.locator('#place-card').screenshot({path:path.join(output,id+'-streets.png')});
      await page.locator('#city-map-region').click();
      await page.locator('#place-card').screenshot({path:path.join(output,id+'-region.png')});
    }
    const before=await page.evaluate(()=>cityMap.stats().generatedCount);
    await page.locator('#place-regenerate').click();
    await page.waitForFunction(before=>!cityMap.stats().loading&&cityMap.stats().generatedCount>before,before);
    await page.evaluate(()=>cityMap.zoomBy(1e9));
    const camera=await page.evaluate(()=>cityMap.stats().camera);
    assert.equal(camera.scale,camera.maximumScale);assert.ok(await page.locator('#city-map-zoom-in').isDisabled());
    assert.equal(errors.length,0);
    await page.evaluate(()=>ui.selectPlace('city:settlement-0041'));
    await page.waitForFunction(()=>cityMap.stats().recipeId==='settlement-0041'&&!cityMap.stats().loading);
    await page.screenshot({path:path.join(output,'world-and-city.png')});
    await fs.writeFile(path.join(output,'inspection.json'),JSON.stringify(report,null,2));
    console.log(JSON.stringify(report));
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
