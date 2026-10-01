const assert=require('node:assert/strict');
const {create}=require('../src/world_atlas/core/web/atlas-navigation-worker.js');
const fs=require('node:fs');
function fixture(){
 const nodes=[[10,10],[15,10],[15,15],[20,10]],edge=(a,b,grade,hours)=>({kind:'road',a,b,points:[nodes[a],nodes[b]],maximumGrade:grade,distanceKm:1,ascentM:0,descentM:0,segments:[[1,0,0,grade,hours,hours,hours,hours]]});
 return {schema:'world-atlas-navigation-v3',shape:[32,64],radiusKm:100,dayHours:26,nodes,edges:[edge(0,1,0,1),edge(1,3,.2,1),edge(1,2,0,2),edge(2,3,0,2)],
  cities:[{id:'A',name:'A',native:nodes[0],node:0,portNode:null},{id:'B',name:'B',native:nodes[3],node:3,portNode:null}],
  profile:{assumptions:'test',modes:[{id:'walk',kind:'road',maximumGrade:.45,dailyHours:8},{id:'carriage',kind:'road',maximumGrade:.1,dailyHours:8},
    {id:'boat',kind:'ship',speedKmh:8,maximumGrade:.45,dailyHours:16},
    {id:'magic-flight',kind:'flight',speedKmh:40,dailyHours:3}]}};
}
const network=create(fixture()),query=mode=>network.route({start:{cityId:'A'},end:{cityId:'B'},mode});
assert.equal(query('walk').parts.length,2);assert.equal(query('carriage').parts.length,3);
assert.ok(query('carriage').distanceKm>query('walk').distanceKm);
assert.equal(query('magic-flight').parts.length,1);
assert.throws(()=>query('teleport'),/没有这种/);
const partial=network.route({start:{native:[11,10]},end:{native:[14,10]},mode:'walk'});
assert.ok(Math.abs(partial.distanceKm-network.route({start:{native:[14,10]},end:{native:[11,10]},mode:'walk'}).distanceKm)<1e-10);
assert.equal(partial.parts.length,1);
assert.ok(partial.parts[0].every(p=>p[0]>=11&&p[0]<=14));
const disconnected=fixture();disconnected.edges=[];assert.throws(()=>create(disconnected).route({start:{cityId:'A'},end:{cityId:'B'},mode:'walk'}),/没有.*路线/);
// Flight has no settlement mesh or ground-network dependency. Range limits time.
disconnected.nodes=[];disconnected.cities.forEach(c=>c.node=null);disconnected.radiusKm=6400;
const direct=create(disconnected).route({start:{cityId:'A'},end:{cityId:'B'},mode:'magic-flight'});
assert.ok(direct.restCount>1);assert.equal(direct.distanceKm,direct.directKm);
assert.ok(Math.abs(direct.elapsedHours-(direct.restCount*26+direct.hours-direct.restCount*3))<1e-9);
const picked=create(disconnected).route({start:{native:[63,16]},end:{native:[1,16]},mode:'magic-flight'});
assert.deepEqual(picked.start.native,[63,16]);assert.deepEqual(picked.end.native,[1,16]);
assert.equal(picked.parts.length,2,'dateline flight never draws across the world');
assert.ok(picked.parts.every(part=>part.every((p,i)=>!i||Math.abs(p[0]-part[i-1][0])<32)));
// Walking to port, boarding, sailing, disembarking and walking. No ghost transfers.
const maritime=fixture();maritime.edges=[];maritime.nodes=[[8,16],[10,16],[11,16],[21,16],[22,16],[24,16]];
maritime.cities=[{id:'A',name:'A',native:maritime.nodes[0],node:0,portNode:null},
  {id:'B',name:'B',native:maritime.nodes[5],node:5,portNode:null}];
const link=(kind,a,b,hours,km=8)=>({kind,a,b,points:[maritime.nodes[a],maritime.nodes[b]],maximumGrade:0,ascentM:0,descentM:0,
  distanceKm:km,segments:[[km,0,0,0,hours,hours,hours,hours]],...(kind==='port'?{port:{name:a===1?'西港':'东港'}}:{})});
maritime.edges=[link('road',0,1,2),link('port',1,2,0),link('sea',2,3,0,160),link('port',3,4,0),link('road',4,5,2)];
const sea=create(maritime).route({start:{cityId:'A'},end:{cityId:'B'},mode:'boat'});
assert.equal(sea.hours,26);assert.equal(sea.elapsedHours,54);
assert.deepEqual(sea.ports,['西港','东港']);assert.ok(sea.seaDistanceKm>0);
assert.throws(()=>create(maritime).route({start:{cityId:'A'},end:{cityId:'B'},mode:'walk'}),/没有.*路线/);
const oceanPicked=create(maritime).route({start:{native:[12,16]},end:{native:[20,16]},mode:'boat'});
assert.deepEqual(oceanPicked.ports,[]);assert.equal(oceanPicked.parts.length,1);assert.ok(oceanPicked.seaDistanceKm>0);
maritime.cities[1]={id:'B',name:'island port',native:maritime.nodes[4],node:null,portNode:4};
assert.ok(create(maritime).route({start:{cityId:'A'},end:{cityId:'B'},mode:'boat'}).seaDistanceKm>0);
assert.throws(()=>create(maritime).route({start:{cityId:'A'},end:{cityId:'B'},mode:'walk'}),/尚未接入道路/);
maritime.edges=maritime.edges.filter(e=>e.kind!=='port');
assert.throws(()=>create(maritime).route({start:{cityId:'A'},end:{cityId:'B'},mode:'boat'}),/没有.*路线/);
if(process.argv[2]){
 const data=JSON.parse(fs.readFileSync(process.argv[2],'utf8')),engine=create(data);
 const a=data.cities.find(c=>c.node!==null),b=data.cities.find(c=>c.node!==null&&c.id!==a.id&&data.edges.some(e=>e.a===a.node&&e.b===c.node||e.b===a.node&&e.a===c.node));
 if(b){const result=engine.route({start:{cityId:a.id},end:{cityId:b.id},mode:'walk'});assert.ok(result.distanceKm>=result.directKm-1e-7);}
 console.log(JSON.stringify({status:'ok',actualNodes:data.nodes.length,actualEdges:data.edges.length,modes:data.profile.modes.map(m=>m.id)}));
}else console.log(JSON.stringify({status:'ok',checks:['mode-specific roads','partial road queries','disconnected roads','direct unrestricted flight',
 'flight recovery','flight date line','port transfers','mixed daily schedules','sea map points','isolated island port','no ghost transfers']}));
