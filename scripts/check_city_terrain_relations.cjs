const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const scope={window:{}};
for(const name of ['city-character.js','city-site.js','city-detail.js'])vm.runInNewContext(fs.readFileSync('src/world_atlas/core/web/'+name,'utf8'),scope);
const site=scope.window.WorldAtlasCitySite,cities=scope.window.WorldAtlasCities;
const root='D:/2/fictional-world-v97-rebuilt/review-city-panel/city-maps/';
const load=id=>JSON.parse(fs.readFileSync(root+'settlement-'+id+'.json','utf8'));
const distance=(a,b,m)=>Math.hypot((a.column-b.column)*m.column,(a.row-b.row)*m.row);
const report={cities:[],synthetic:{}};
const intersection=(a,b,c,d)=>{
 if(Math.max(a.column,b.column)<Math.min(c.column,d.column)||Math.min(a.column,b.column)>Math.max(c.column,d.column)
   ||Math.max(a.row,b.row)<Math.min(c.row,d.row)||Math.min(a.row,b.row)>Math.max(c.row,d.row))return false;
 const cross=(a,b,c)=>(b.column-a.column)*(c.row-a.row)-(b.row-a.row)*(c.column-a.column);
 return cross(a,b,c)*cross(a,b,d)<0&&cross(c,d,a)*cross(c,d,b)<0;
};
for(const id of ['0581','0721','0301','0109','0627','0041']){
 const payload=load(id),r=site.generate(payload.recipe,payload.siteContext,{seed:20261023}),s=r.localSite,m=r.urban.gridCellKilometres;
 const roads=[...r.transport.corridors,...s.ruralRoads].filter(p=>p.kind==='road');
 const roadEdges=roads.flatMap(road=>road.points.slice(1).map((b,i)=>[road.points[i],b]));
 for(const field of s.farmland){
   assert.ok(field.areaKm2>=.004,'discard unusable fragments');
   for(let i=0;i<field.points.length;i++)for(const [a,b] of roadEdges)
     assert.ok(!intersection(field.points[i],field.points[(i+1)%field.points.length],a,b),id+' fields leave roads clear');
   if(field.kind==='terrace')for(const line of field.lines){const z=line.map(p=>site.elevationAt(s.surface,p));
     assert.ok(Math.max(...z)-Math.min(...z)<.15,id+' terrace rows follow the actual contour');}
 }
 let mouthCount=0,widthRatio=0;
 for(const channel of s.channels){
   const widths=channel.points.map(p=>p.widthMetres);widthRatio=Math.max(widthRatio,Math.max(...widths)/Math.min(...widths));
   for(let i=1;i<channel.points.length;i++)for(let k=0;k<=4;k++){
     const a=channel.points[i-1],b=channel.points[i],t=k/4,p={column:a.column+(b.column-a.column)*t,row:a.row+(b.row-a.row)*t};
     assert.ok(site.elevationAt(s.surface,p)>-.02,id+' river ends at the rendered coast');
   }
   if(channel.mouth){mouthCount++;assert.ok(Math.abs(site.elevationAt(s.surface,channel.points.at(-1)))<.02);}
 }
 if(s.channels.length)assert.ok(widthRatio>1.05,'channel width varies in ground metres');
 if(r.harbor?.sailing?.length)for(let i=1;i<r.harbor.sailing.length;i++)for(let k=0;k<=64;k++){
   const a=r.harbor.sailing[i-1],z=r.harbor.sailing[i],t=k/64,p={column:a.column+(z.column-a.column)*t,row:a.row+(z.row-a.row)*t},b=s.bounds;
   if(p.column>=b.west&&p.column<=b.east&&p.row>=b.north&&p.row<=b.south)assert.ok(site.elevationAt(s.surface,p)<=.02,'the harbor entrance stays in water');
 }
 const names=s.annotations.filter(a=>/^(supply-river|local-peak|local-wood|local-island)/.test(a.id));
 for(const feature of names){assert.ok(!feature.name.includes('穿城河'));assert.ok(payload.siteContext.nameRoots.some(root=>feature.name.startsWith(root)));}
 report.cities.push({id,fields:s.farmland.length,terraces:s.farmland.filter(f=>f.kind==='terrace').length,mouths:mouthCount,widthRatio,form:s.form.kind});
}
const payload=load('0004'),base={...payload.recipe,transport:{corridors:[]}},b=payload.siteContext.bounds,m=payload.recipe.urban.gridCellKilometres;
const terrain={...payload.siteContext,source:{kind:'groundwater',location:base.location},elevationMetres:payload.siteContext.elevationMetres.map(()=>800)};
const palaceArea=population=>{
 const r=site.generate({...base,population:{...base.population,estimate:population}},terrain,{seed:20261023}),p=cities.plan(r,{grid:payload.grid});
 const palace=p.landmarks.find(l=>l.kind==='royal-palace'||l.kind==='imperial-palace');assert.ok(palace);
 const points=palace.compoundPoints,area=Math.abs(points.reduce((s,a,i)=>{const z=points[(i+1)%points.length];return s+(a.column-base.location.column)*(z.row-base.location.row)-(z.column-base.location.column)*(a.row-base.location.row);},0))*m.column*m.row/2;
 return {area,parts:p.landmarks.filter(l=>l.compoundId===palace.compoundId).length,precinct:p.plazas.some(p=>p.precinct)};
};
const small=palaceArea(20000),large=palaceArea(300000);
assert.ok(large.area>small.area*3,'metropolitan palace reserves more than one ordinary block');assert.ok(large.precinct);assert.ok(large.parts>=15);
report.synthetic.palaces={small,large};
const hillRecipe={...base,siteType:'market',population:{estimate:12000},urban:{...base.urban,radiusKm:1},culture:{...base.culture,isCapital:false},transport:{corridors:[]}};
const hillContext={...terrain,radiusKm:5,bounds:{north:base.location.row-5/m.row,south:base.location.row+5/m.row,west:base.location.column-5/m.column,east:base.location.column+5/m.column},
 elevationMetres:Array.from({length:289},(_,i)=>800+(i%17-8)/8*5*70)};
const hillside=site.generate(hillRecipe,hillContext,{seed:20261023});
const terraces=hillside.localSite.farmland.filter(f=>f.kind==='terrace');assert.ok(terraces.length>0,'arable hillsides have actual contour terraces');
for(const field of terraces)for(const line of field.lines){const heights=line.map(p=>site.elevationAt(hillside.localSite.surface,p));assert.ok(Math.max(...heights)-Math.min(...heights)<.15);}
const steep=site.generate(hillRecipe,{...hillContext,elevationMetres:Array.from({length:289},(_,i)=>1200+(i%17-8)/8*5*200)},{seed:20261023});
let maxGrade=0,uphillRoutes=0;
for(const road of steep.localSite.ruralRoads){
 const heights=road.points.map(p=>site.elevationAt(steep.localSite.surface,p));
 if(Math.max(...heights)-Math.min(...heights)>80)uphillRoutes++;
 for(let i=1;i<road.points.length;i++){const length=distance(road.points[i-1],road.points[i],m);if(length>.001)maxGrade=Math.max(maxGrade,Math.abs(heights[i]-heights[i-1])/(length*1000));}
}
assert.ok(uphillRoutes>0,'switchbacks reach rising terrain instead of discarding every uphill approach');assert.ok(maxGrade<=.1201,'mountain road grade stays under twelve percent');
report.synthetic.hillside={terraces:terraces.length,uphillRoutes,maxGrade};
fs.writeFileSync('D:/2/fictional-world-v97-rebuilt/review-city-panel/city-terrain-relations-check.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report));
