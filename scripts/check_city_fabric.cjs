/* Compare the pure city planner against one saved city's physical context. */
const {chromium}=require('C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs/promises');
const path=require('node:path');
const {pathToFileURL}=require('node:url');

(async()=>{
  if(!process.argv[2]||!process.argv[3])throw new Error('Usage: node scripts/check_city_fabric.cjs <review-directory> <output-directory>');
  const sourceReview=path.resolve(process.argv[2]),output=path.resolve(process.argv[3]);
  const payload=JSON.parse(await fs.readFile(path.join(sourceReview,'city-maps','settlement-0677.json'),'utf8'));
  assert.equal(payload.schema,'world-atlas-city-map-v1');
  assert.equal(payload.recipe.id,'settlement-0677');
  assert.ok(payload.grid.width>0&&payload.grid.height>0);
  assert.equal(payload.siteContext.schema,'city-site-context-v1');
  await fs.mkdir(output,{recursive:true});
  const web=path.join(__dirname,'..','src','world_atlas','core','web');
  const [siteScript,detailScript]=await Promise.all([
    fs.readFile(path.join(web,'city-site.js'),'utf8'),fs.readFile(path.join(web,'city-detail.js'),'utf8')]);
  const eras=['tribal','ancient','medieval','early-modern','industrial','contemporary'];
  const labels=['部落 · 疏散院落','古典 · 内城与外墙','中古 · 曲折街路','近世 · 城墙与坊巷','工业 · 旧城与新区','近现代 · 扩张区'];
  const cards=eras.map((era,index)=>`<article><h2>${labels[index]}</h2><svg id="city-${index}" xmlns="http://www.w3.org/2000/svg"></svg></article>`).join('');
  const fixture=JSON.stringify({payload,eras}).replaceAll('<','\\u003c');
  await fs.writeFile(path.join(output,'city-fabric.html'),`<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>城市形态检查</title><style>body{margin:0;background:#efe7d4;color:#5c5147;font:14px "Microsoft YaHei",system-ui}main{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:#c4b9a4}article{background:#f6f0df;padding:16px}h2{margin:0;font-size:16px;font-weight:500}svg{display:block;width:100%;height:455px}header{padding:18px}p{margin:6px 0 0;color:#837760}</style><header>同一城市 · 不同时代的街区生成<p>同一人口、公里尺度、世界位置与生成 seed；时代改变街路、建筑和城防。</p></header><main>${cards}</main><script>${await fs.readFile(path.join(web,'city-character.js'),'utf8')}</script><script>${siteScript}</script><script>${detailScript}</script><script>window.cityFixture=${fixture};</script></html>`);
  const report={schema:'city-fabric-browser-check-v2',status:'failed',sourceCity:payload.recipe.id,
    population:payload.recipe.population,grid:payload.grid,seed:payload.recipe.seed,cities:[],errors:[]};
  let browser;
  try{
    browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',args:['--enable-unsafe-swiftshader']});
    const page=await browser.newPage({viewport:{width:1500,height:1120}});
    page.on('pageerror',error=>report.errors.push(String(error)));
    await page.goto(pathToFileURL(path.join(output,'city-fabric.html')).href);
    report.cities=await page.evaluate(()=>{
      const {payload,eras}=cityFixture,NS='http://www.w3.org/2000/svg';
      window.fabricRecipes=[];
      window.renderFabric=(svg,recipe,prefix)=>{
        const site=document.createElementNS(NS,'g');site.setAttribute('class','fixture-local-terrain');
        for(const band of recipe.localSite.terrainBands){
          const node=document.createElementNS(NS,'path');node.setAttribute('d',band.d);node.setAttribute('fill',band.fill);site.appendChild(node);
        }
        for(const channel of recipe.localSite.channels){
          const node=document.createElementNS(NS,'path');
          node.setAttribute('d',channel.points.map((point,index)=>`${index?'L':'M'}${point.column} ${point.row}`).join(' '));
          node.setAttribute('fill','none');node.setAttribute('stroke','#8bb9c5');
          node.setAttribute('stroke-width',channel.widthMetres/1000/((recipe.urban.gridCellKilometres.row+recipe.urban.gridCellKilometres.column)/2));
          site.appendChild(node);
        }
        const drawing=WorldAtlasCities.render(recipe,{grid:payload.grid,clipPrefix:prefix});
        svg.replaceChildren(site,drawing);
        const view=svg.viewBox.baseVal,scale=Math.min(svg.clientWidth/view.width,svg.clientHeight/view.height);
        for(const label of drawing.querySelectorAll('[data-city-screen-label]')){
          label.setAttribute('font-size',12/scale);label.setAttribute('stroke-width',2.5/scale);
        }
        return drawing;
      };
      return eras.map((era,index)=>{
        const start=performance.now();
        const recipe=WorldAtlasCitySite.generate({...payload.recipe,id:'era-'+era,era},payload.siteContext,{seed:payload.recipe.seed});
        fabricRecipes.push(recipe);
        const svg=document.getElementById('city-'+index),{north,west,south,east}=recipe.urban.bounds;
        const gap=Math.max(south-north,east-west)*.11;
        svg.setAttribute('viewBox',`${west-gap} ${north-gap} ${east-west+gap*2} ${south-north+gap*2}`);
        const drawing=renderFabric(svg,recipe,'era-'+era),plan=WorldAtlasCities.plan(recipe,{grid:payload.grid});
        return {era,ms:performance.now()-start,blocks:plan.blocks.length,buildings:plan.buildings.length,
          streets:plan.streets.length,plazas:plan.plazas.length,innerWalls:plan.innerWalls.length,outerWalls:plan.walls.length,
          towers:plan.towers.length,farmland:plan.farmland.length,nodes:drawing.querySelectorAll('*').length,
          renderedBuildings:Array.from(drawing.querySelectorAll('[data-building-count]'),node=>Number(node.dataset.buildingCount)).reduce((a,b)=>a+b,0)};
      });
    });
    for(const city of report.cities){
      assert.ok(city.buildings>0,`${city.era} must render building geometry`);
      assert.equal(city.renderedBuildings,city.buildings);
    }
    assert.ok(report.cities.find(city=>city.era==='ancient').innerWalls>0,'Classical comparison includes its inner defensive boundary');
    assert.equal(report.cities.find(city=>city.era==='contemporary').outerWalls,0,'Contemporary comparison removes historic city walls');
    await page.screenshot({path:path.join(output,'city-fabric.png')});
    report.fortress=await page.evaluate(()=>{
      const base=fabricRecipes[1],recipe={...base,siteType:'fortress',landmarks:[...base.landmarks,'citadel']};
      const plan=WorldAtlasCities.plan(recipe,{grid:cityFixture.payload.grid});
      if(!plan.fortress)throw new Error('Classical fortress is missing');
      const svg=document.getElementById('city-1'),fort=plan.fortress;
      const rows=fort.points.map(point=>point.row),columns=fort.points.map(point=>point.column);
      const north=Math.min(...rows),south=Math.max(...rows),west=Math.min(...columns),east=Math.max(...columns);
      const gap=Math.max(south-north,east-west)*1.05;
      svg.setAttribute('viewBox',`${west-gap} ${north-gap} ${east-west+gap*2} ${south-north+gap*2}`);
      const drawing=renderFabric(svg,recipe,'ancient-fortress');
      const rendered=drawing.querySelector('.city-fortress');
      if(!rendered)throw new Error('The fortress recipe must actually be rendered before its screenshot');
      return {era:recipe.era,siteType:recipe.siteType,towers:fort.towers.length,
        kind:fort.kind,interiors:fort.interiors.length,accessPoints:fort.access.length,renderedPaths:rendered.querySelectorAll('path').length,
        innerWalls:plan.innerWalls.length,
        renderedInnerWalls:Array.from(drawing.querySelectorAll('.city-inner-walls')).filter(node=>node.getAttribute('d')).length,
        nodes:drawing.querySelectorAll('*').length};
    });
    assert.equal(report.fortress.towers,4);assert.ok(report.fortress.renderedPaths>=4);
    assert.ok(['garrison','palace-citadel'].includes(report.fortress.kind));assert.ok(report.fortress.interiors>=5);
    assert.ok(report.fortress.innerWalls>0&&report.fortress.renderedInnerWalls>0,
      'The fortress screenshot renders the classical inner-city defensive perimeter as well');
    await page.locator('article').nth(1).screenshot({path:path.join(output,'ancient-fortress.png')});
    assert.deepEqual(report.errors,[]);report.status='ok';
  }catch(error){report.failure=String(error.stack||error);throw error;}
  finally{
    await fs.writeFile(path.join(output,'browser-check.json'),JSON.stringify(report,null,2));
    process.stdout.write(JSON.stringify(report)+'\n');if(browser)await browser.close();
  }
})().catch(error=>{console.error(error);process.exitCode=1;});
