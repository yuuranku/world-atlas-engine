const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const review=path.resolve(process.argv[2]),output=path.resolve(process.argv[3]);
const context={window:{}};
for(const file of ['city-character.js','city-site.js','city-detail.js'])vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../src/world_atlas/core/web',file),'utf8'),context);
const api=context.window,report={cities:[],palaces:[],lakes:[],islands:[]},geometry=[];
for(const id of ['0024','0059','0373','0646','0019','0004','0041','0721']){
  const payload=JSON.parse(fs.readFileSync(path.join(review,'city-maps','settlement-'+id+'.json'),'utf8'));
  for(const seed of [20261002,20261003]){
    const start=performance.now(),r=api.WorldAtlasCitySite.generate(payload.recipe,payload.siteContext,{seed}),p=api.WorldAtlasCities.plan(r,{grid:payload.grid});
    assert.ok(p.buildings.length>0);
    assert.ok(p.countryside.every(h=>h.entrance&&p.countryLanes.some(l=>Math.hypot(l[0].column-h.entrance.column,l[0].row-h.entrance.row)<1e-9)),'each farmstead has an entrance path');
    assert.ok(p.hamlets.every(h=>!h.name.includes('undefined')&&!h.name.startsWith(r.name)),'hamlets have independent names');
    assert.ok(p.landmarks.every(l=>l.foundation.reliefMetres<=4),'civic buildings have coherent foundations');
    for(const lake of r.localSite.lakes){
      assert.ok(lake.waterLevelMetres<=lake.spillElevationMetres&&lake.maximumDepthMetres>2);
      assert.ok(api.WorldAtlasCitySite.elevationAt(r.localSite.surface,lake.anchor)<lake.waterLevelMetres);
      report.lakes.push({id,seed,area:lake.areaKm2,level:lake.waterLevelMetres,spill:lake.spillElevationMetres});
    }
    for(const c of r.localSite.confluences){const joined=r.localSite.channels[c.tributaryChannel].points.at(-1);
      assert.ok(Math.hypot(joined.column-c.anchor.column,joined.row-c.anchor.row)<1e-9,'tributaries share the main-channel node');}
    report.islands.push({id,seed,shapes:r.localSite.coastal.islands.map(i=>({shape:i.shape,seed:i.shapeSeed,area:i.areaKm2}))});
    report.cities.push({id,seed,ms:Math.round(performance.now()-start),buildings:p.buildings.length,colors:new Set(p.buildings.map(b=>b.fill)).size,
      palaces:p.plazas.filter(x=>x.precinct).map(x=>x.design),harbor:r.harbor?.kind,lakes:r.localSite.lakes.length,confluences:r.localSite.confluences.length});
    geometry.push({id,seed,metric:r.urban.gridCellKilometres,buildings:p.buildings.map(b=>b.points),landmarks:p.landmarks,
      plazas:p.plazas,streets:p.streets,walls:p.walls.concat(p.innerWalls),harbor:r.harbor});
  }
}
const base=JSON.parse(fs.readFileSync(path.join(review,'city-maps/settlement-0004.json'),'utf8'));
const designs=new Set();
for(const style of ['courtyard','arcaded','timber-frame','stone-masonry','canal-side']){
  const r={...base.recipe,seed:8123,localSite:{form:{kind:'plain'}},culture:{...base.recipe.culture,style}};
  const profile=api.WorldAtlasCityCharacter.profile(r),parts=api.WorldAtlasCityCharacter.template('royal-palace',style,300000,profile.palace);
  designs.add(JSON.stringify(parts));assert.ok(parts.length>=14,'large palace includes halls and administrative wings');
  report.palaces.push({style,family:profile.palace.family,parts:parts.length,material:profile.material.name});
}
assert.equal(designs.size,5,'cultural palace families have distinct geometry');
const n=33,bounds={north:-.5,west:-.5,south:.5,east:.5},
  bowl=Array.from({length:n*n},(_,i)=>{const x=(i%n/(n-1)-.5)*10,y=(Math.floor(i/n)/(n-1)-.5)*10;
    return 500-y*4-55*Math.exp(-((x+.26)**2+(y-1.7)**2)/.4);}),
  basinContext={schema:'city-site-context-v1',bounds,rows:n,columns:n,elevationMetres:bowl,water:Array(n*n).fill(0),radiusKm:5,
    gridCellKilometres:{row:10,column:10},source:{kind:'river',location:{row:.4,column:0},referenceWidthMetres:60},nameRoots:base.siteContext.nameRoots},
  basinRecipe={id:'basin-fixture',location:{column:0,row:0},population:{estimate:12000},urban:{radiusKm:1,streetWidthsMetres:{arterial:8}},
    siteType:'river-city',transport:{corridors:[]}},
  basin=api.WorldAtlasCitySite.generate(basinRecipe,basinContext,{seed:3});
assert.ok(basin.localSite.lakes.length>0,'a closed river-side depression accumulates a lake');
assert.ok(basin.localSite.confluences.length>0&&basin.harbor?.kind==='river','the navigable confluence has an inland port');
for(const lake of basin.localSite.lakes){assert.ok(lake.waterLevelMetres<lake.spillElevationMetres);
  assert.ok(!api.WorldAtlasCitySite.isDry(basin.localSite.surface,lake.anchor),'lake geometry and land eligibility share the same water surface');}
report.basin={lakes:basin.localSite.lakes.length,confluences:basin.localSite.confluences.length,harbor:basin.harbor.kind};
fs.mkdirSync(output,{recursive:true});fs.writeFileSync(path.join(output,'refinement-geometry.json'),JSON.stringify(geometry));
fs.writeFileSync(path.join(output,'refinement-check.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report));
