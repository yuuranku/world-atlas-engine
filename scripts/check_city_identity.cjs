const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const scope={window:{},Math,performance,console};
for(const file of ['city-character.js','city-site.js','city-detail.js'])vm.runInNewContext(fs.readFileSync(path.join('src/world_atlas/core/web',file),'utf8'),scope);
const {WorldAtlasCitySite:site,WorldAtlasCities:cities}=scope.window;
const payload=JSON.parse(fs.readFileSync('D:/2/fictional-world-v97-rebuilt/review-city-panel/city-maps/settlement-0005.json','utf8'));
const base={...payload.recipe,siteType:'fortress',culture:{...payload.recipe.culture,isCapital:true,government:'bureaucratic-monarchy',stateName:'测试王朝'},transport:{corridors:[]}};
const ground={...payload.siteContext,elevationMetres:payload.siteContext.elevationMetres.map(()=>500),water:payload.siteContext.water.map(()=>0),source:{kind:'groundwater',location:base.location}};
const report={styles:[],holy:[]};
for(const style of ['courtyard','timber-frame','arcaded']){
 const r=site.generate({...base,culture:{...base.culture,style}},ground,{seed:20261011}),p=cities.plan(r,{grid:payload.grid});
 const names=p.annotations.filter(a=>a.kind==='road').map(a=>a.name);
 assert.ok(p.landmarks.some(l=>l.kind==='imperial-palace'),'capital has a reserved palace parcel');
 assert.equal(new Set(names).size,names.length);
 report.styles.push({style,plan:p.character.plan,buildings:p.buildings.length,moats:p.moats.length,names:names.slice(0,9)});
}
for(const tradition of ['ancestral-rite','river-mysteries','sky-law','mountain-vow','tide-covenant','pilgrim-way']){
 const r=site.generate({...base,culture:{...base.culture,holyReligion:{name:'测试教',tradition}}},ground,{seed:20261011}),p=cities.plan(r,{grid:payload.grid});
 const special=p.landmarks.filter(l=>['holy-tomb','water-sanctuary','observatory-sanctuary','pilgrimage-monastery','harbor-shrine','great-sanctuary'].includes(l.kind));
 assert.equal(special.length,1,tradition+' has one distinct pilgrimage core');report.holy.push({tradition,kind:special[0].kind});
}
const riverGround={...ground,source:{kind:'river',location:{row:base.location.row+1,column:base.location.column}}};
let river;
for(let seed=1;seed<8;seed++){
 const r=site.generate({...base,siteType:'river-city'},riverGround,{seed});if(r.localSite.riverMode==='through'){river=r;break;}
}
assert.ok(river);const p=cities.plan(river,{grid:payload.grid});
const channel=river.localSite.channels[0],a=channel.points[0],z=channel.points.at(-1),dx=z.column-a.column,dy=z.row-a.row;
const signs=new Set(p.buildings.map(b=>Math.sign(dx*(b.points[0].row-a.row)-dy*(b.points[0].column-a.column))));
assert.equal(signs.size,2,'verified bridge supports buildings on both river banks');
for(const route of [...river.transport.corridors,...river.localSite.ruralRoads])for(let i=1;i<route.points.length;i++)for(const ch of river.localSite.channels)for(let j=1;j<ch.points.length;j++){
 const a=route.points[i-1],b=route.points[i],c=ch.points[j-1],d=ch.points[j],dx=b.column-a.column,dy=b.row-a.row,sx=d.column-c.column,sy=d.row-c.row,den=dx*sy-dy*sx;
 if(Math.abs(den)<1e-12)continue;const t=((c.column-a.column)*sy-(c.row-a.row)*sx)/den,u=((c.column-a.column)*dy-(c.row-a.row)*dx)/den;
 if(t<0||t>1||u<0||u>1)continue;const point={column:a.column+dx*t,row:a.row+dy*t},m=river.urban.gridCellKilometres;
 assert.ok(river.localSite.bridges.some(bridge=>Math.hypot((bridge.center.column-point.column)*m.column,(bridge.center.row-point.row)*m.row)<.09),'every road crossing uses a verified bridge');
}
report.river={bridges:river.localSite.bridges.length,buildings:p.buildings.length,banks:signs.size,roads:river.transport.corridors.length};
console.log(JSON.stringify(report));
