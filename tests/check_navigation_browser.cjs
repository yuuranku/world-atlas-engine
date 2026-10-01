const assert=require('node:assert/strict'),fs=require('node:fs/promises'),path=require('node:path');
const {chromium}=require('C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const {create}=require('../src/world_atlas/core/web/atlas-navigation-worker.js');
(async()=>{
 const review=path.resolve(process.argv[2]),base=process.argv[3];
 const data=JSON.parse(await fs.readFile(path.join(review,'navigation-network.json'),'utf8')),engine=create(data);
 const candidates=data.cities.filter(c=>c.node!==null);let pair,expected,flightPair,flightResult,seaPair,seaResult;
 for(const a of candidates.slice(0,80)){for(const b of candidates.slice(1,80)){
   if(a.id===b.id)continue;
   try{const r=engine.route({start:{cityId:a.id},end:{cityId:b.id},mode:'carriage'});if(r.distanceKm>30&&r.distanceKm<600){pair=[a,b];expected=r;break;}}catch{}
 }if(pair)break;}
 assert.ok(pair,'actual carriage-connected cities');
 for(const a of data.cities.slice(0,250)){for(const b of data.cities.slice(0,250)){
   if(a.id===b.id)continue;
   try{const r=engine.route({start:{cityId:a.id},end:{cityId:b.id},mode:'dragon'});if(r.restCount){flightPair=[a,b];flightResult=r;break;}}catch{}
 }if(flightPair)break;}
 assert.ok(flightPair,'a direct flight with daily range and recovery');
 const ports=data.cities.filter(c=>c.portNode!==null);
 for(const a of ports.slice(0,80)){for(const b of ports.slice(1,80)){
   if(a.id===b.id)continue;
   try{const r=engine.route({start:{cityId:a.id},end:{cityId:b.id},mode:'boat'});if(r.seaDistanceKm>50&&r.distanceKm<1200&&r.ports.length>=2){seaPair=[a,b];seaResult=r;break;}}catch{}
 }if(seaPair)break;}
 assert.ok(seaPair,'actual harbors connected by a sea itinerary');
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
 const page=await browser.newPage({viewport:{width:1600,height:1000}}),errors=[],requests=[];
 page.on('pageerror',e=>errors.push(String(e)));page.on('response',r=>{if(r.status()>=400)errors.push(`${r.status()} ${r.url()}`)});
 page.on('request',r=>requests.push(r.url()));
 try{
   await page.goto(new URL('index.html?theme=political',base).href,{waitUntil:'load',timeout:120000});
   assert.equal(requests.some(u=>u.endsWith('navigation-network.json')),false,'network stays lazy until visible entry opens');
   await page.locator('#navigation-toggle').click();await page.waitForFunction(()=>document.querySelector('#navigation-mode').options.length===6);
   assert.equal(await page.locator('#navigation-panel').isVisible(),true);
   const verifySymbolSizes=async()=>{
     const sizes=await page.locator('#city-symbols [data-city-symbol-tier]').evaluateAll(nodes=>nodes
       .filter(node=>getComputedStyle(node).display!=='none')
       .map(node=>{const r=node.getBoundingClientRect();return Math.max(r.width,r.height)}));
     assert.ok(sizes.length&&Math.max(...sizes)<32,'city symbols must retain readable screen sizes after focus');
   };
   const plan=async(a,b,mode)=>{await page.locator('#navigation-start').fill(a.name);await page.locator('#navigation-end').fill(b.name);
     await page.locator('#navigation-mode').selectOption(mode);await page.locator('#navigation-go').click();
     await page.waitForFunction(({a,b,mode})=>window.navigation.result?.start.cityId===a&&navigation.result?.end.cityId===b&&navigation.result.mode===mode,{a:a.id,b:b.id,mode},{timeout:120000});
     return page.evaluate(()=>navigation.result);};
   const carriage=await plan(...pair,'carriage');assert.ok(Math.abs(carriage.distanceKm-expected.distanceKm)<1e-8);
   assert.ok(carriage.distanceKm>=carriage.directKm-1e-7);assert.ok(await page.locator('[data-atlas-navigation] path').count()===2);
   await page.locator('#navigation-swap').click();await page.waitForFunction(id=>navigation.result?.start.cityId===id,pair[1].id);
   const swapped=await page.evaluate(()=>navigation.result);assert.ok(Math.abs(swapped.distanceKm-carriage.distanceKm)<1e-8);
   await page.locator('#navigation-mode').selectOption('horse');await page.waitForFunction(()=>navigation.result?.mode==='horse');
   await page.locator('#navigation-mode').selectOption('walk');await page.waitForFunction(()=>navigation.result?.mode==='walk');
   await page.waitForFunction(()=>tileManager.stats().pendingRequests===0&&tileManager.stats().queuedTiles===0&&overviewManager.stats().pendingRasters===0);await page.waitForTimeout(200);
   await verifySymbolSizes();
   await page.screenshot({path:path.join(review,'check-v100-navigation-road.png')});
   const flights=await plan(...flightPair,'dragon');assert.equal(flights.restCount,flightResult.restCount);
   assert.equal(flights.distanceKm,flights.directKm);
   assert.ok(flights.elapsedHours>=data.dayHours);assert.equal(flights.dailyRangeKm,390);
   assert.ok(await page.locator('#navigation-status').textContent().then(t=>t.includes('途中休息')));
   assert.ok(await page.locator('[data-navigation-flight]').first().getAttribute('d').then(d=>d.includes('Q')));
   await page.waitForFunction(()=>tileManager.stats().pendingRequests===0&&tileManager.stats().queuedTiles===0&&overviewManager.stats().pendingRasters===0);await page.waitForTimeout(200);
   await verifySymbolSizes();
   await page.screenshot({path:path.join(review,'check-v100-navigation-dragon.png')});
   const userStart=data.cities.find(c=>c.name==='拉玛丹加'),userEnd=data.cities.find(c=>c.name==='乌纳孙盖');
   assert.ok(userStart&&userEnd,'the exact failed flight from the user screenshot');
   const magic=await plan(userStart,userEnd,'magic-flight');assert.equal(magic.distanceKm,magic.directKm);assert.ok(magic.restCount>0);
   await page.waitForFunction(()=>tileManager.stats().pendingRequests===0&&tileManager.stats().queuedTiles===0&&overviewManager.stats().pendingRasters===0);
   await page.screenshot({path:path.join(review,'check-v100-navigation-magic.png')});
   const sailing=await plan(...seaPair,'boat');assert.deepEqual(sailing.ports,seaResult.ports);assert.ok(sailing.seaDistanceKm>50);
   assert.ok(await page.locator('#navigation-status').textContent().then(t=>t.includes('上下船港口')));
   assert.ok(await page.locator('#navigation-detail').textContent().then(t=>t.includes('海路')));
   await page.waitForFunction(()=>tileManager.stats().pendingRequests===0&&tileManager.stats().queuedTiles===0&&overviewManager.stats().pendingRasters===0);
   await verifySymbolSizes();
   await page.screenshot({path:path.join(review,'check-v100-navigation-sea.png')});
   await page.locator('#navigation-clear').click();assert.equal(await page.evaluate(()=>navigation.result),null);
   const edge=data.edges.find(e=>e.kind==='road'&&e.maximumGrade<.05&&e.points.length>=2),point=edge.points[Math.floor(edge.points.length/2)];
   await page.locator('#navigation-mode').selectOption('walk');await page.locator('#navigation-pick-start').click();
   await page.evaluate(point=>{scale=24;translateX=viewport.clientWidth/2-point[0]*atlasScaleX*scale;translateY=viewport.clientHeight/2-(point[1]*atlasScaleY+polarSceneBand)*scale;applyTransform();},point);
   const box=await page.locator('#viewport').boundingBox();await page.mouse.click(box.x+box.width/2,box.y+box.height/2);
   assert.ok((await page.locator('#navigation-start').inputValue()).startsWith('地图选点'));
   await page.locator('#navigation-pick-end').click();await page.keyboard.press('Escape');assert.equal(await page.evaluate(()=>navigation.picking),false);
   const off=data.cities.find(c=>c.node===null);await page.locator('#navigation-start').fill(off.name);await page.locator('#navigation-end').fill(pair[0].name);
   await page.locator('#navigation-go').click();await page.waitForFunction(()=>document.querySelector('#navigation-status').textContent.includes('尚未接入道路网络'));
   await page.locator('#navigation-close').click();assert.equal(await page.locator('#navigation-panel').isVisible(),false);
   assert.deepEqual(errors,[]);
   const report={status:'ok',url:base,checks:['visible entry','lazy network','typed cities','actual geometry and distance','swap','walking and horse modes','direct dragon range and rest','user magic flight regression','curved flight overlay','ports and sailing','map click','Escape','disconnected location','close panel'],
     road:{names:pair.map(c=>c.name),distanceKm:carriage.distanceKm,hours:carriage.hours},flight:{names:flightPair.map(c=>c.name),distanceKm:flights.distanceKm,elapsedHours:flights.elapsedHours,restCount:flights.restCount},
     magic:{names:[userStart.name,userEnd.name],distanceKm:magic.distanceKm,restCount:magic.restCount},
     sailing:{names:seaPair.map(c=>c.name),distanceKm:sailing.distanceKm,seaDistanceKm:sailing.seaDistanceKm,ports:sailing.ports,elapsedHours:sailing.elapsedHours},errors};
   await fs.writeFile(path.join(review,'navigation-browser-check.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
