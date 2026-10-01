/* Query the delivered network away from camera and input handlers. */
(function(root){
  'use strict';
  let ruler;
  if(typeof module!=='undefined'&&module.exports) ruler=require('./atlas-ruler.js');
  else {importScripts('atlas-ruler.js');ruler=root.WorldAtlasRuler;}
  class Heap {
    constructor(){this.values=[];}
    push(item){const a=this.values;let i=a.length;a.push(item);while(i){const p=(i-1)>>1;if(a[p][0]<=item[0])break;a[i]=a[p];i=p;}a[i]=item;}
    pop(){const a=this.values,result=a[0],last=a.pop();if(a.length){let i=0;while(i*2+1<a.length){let c=i*2+1;if(c+1<a.length&&a[c+1][0]<a[c][0])c++;if(a[c][0]>=last[0])break;a[i]=a[c];i=c;}a[i]=last;}return result;}
    get length(){return this.values.length;}
  }
  function shortest(adjacency,start,goal){
    const costs=new Float64Array(adjacency.length).fill(Infinity),parents=new Array(adjacency.length),heap=new Heap();
    costs[start]=0;heap.push([0,start]);
    while(heap.length){const [cost,node]=heap.pop();if(cost!==costs[node])continue;if(node===goal)break;
      for(const arc of adjacency[node]){const next=arc.arriveAt?arc.arriveAt(cost):cost+(arc.costHours??arc.hours);if(next>=costs[arc.to])continue;
        costs[arc.to]=next;parents[arc.to]={from:node,arc};heap.push([next,arc.to]);}}
    if(!Number.isFinite(costs[goal]))throw Error('这两个地点之间没有该交通方式可通行的路线。');
    const result=[];for(let node=goal;node!==start;){const step=parents[node];result.push(step.arc);node=step.from;}return result.reverse();
  }
  function create(data){
    if(data.schema!=='world-atlas-navigation-v3'||!Array.isArray(data.nodes)||!Array.isArray(data.edges))throw Error('导航网络无效');
    const geo=([x,y])=>[x/data.shape[1]*360-180,90-y/data.shape[0]*180];
    const dist=(a,b)=>ruler.greatCircleDistanceKm(geo(a),geo(b),data.radiusKm);
    const cities=new Map(data.cities.map(c=>[c.id,c]));
    const modes=new Map(data.profile.modes.map(m=>[m.id,m]));
    function permitted(edge,mode){return edge.kind==='sea'?mode.kind==='ship':
      edge.maximumGrade<=mode.maximumGrade&&(edge.kind==='road'||mode.kind==='ship');}
    function arrival(elapsed,hours,dailyHours){
      if(hours<=0)return elapsed;
      let phase=elapsed%data.dayHours;
      if(phase>=dailyHours){elapsed+=data.dayHours-phase;phase=0;}
      const available=dailyHours-phase;
      if(hours<=available)return elapsed+hours;
      hours-=available;elapsed+=data.dayHours-phase;
      const days=Math.max(0,Math.ceil(hours/dailyHours)-1);
      return elapsed+days*data.dayHours+hours-days*dailyHours;
    }
    function snap(request,mode){
      if(request.cityId){const city=cities.get(request.cityId);if(!city)throw Error('未找到这个聚落。');
        if(mode.kind==='road'&&city.node===null)throw Error(`${city.name} 尚未接入道路网络。`);
        const node=mode.kind==='ship'?(city.node??city.portNode):city.node;
        if(mode.kind==='ship'&&node===null)throw Error(`${city.name} 尚未接入道路或港口网络。`);
        return {native:city.native,node,name:city.name,cityId:city.id,snapKm:0};}
      const point=request.native;
      if(!Array.isArray(point)||point.length!==2||!point.every(Number.isFinite)||point[0]<0||point[0]>data.shape[1]||point[1]<0||point[1]>data.shape[0])throw Error('请选择地图内的地点。');
      if(mode.kind==='flight')return {native:point,name:'地图选点',snapKm:0};
      let best=null;
      const sx=Math.max(.01,Math.cos(geo(point)[1]*Math.PI/180));
      data.edges.forEach((edge,index)=>{if(!permitted(edge,mode)||edge.kind==='port')return;
        for(let i=1;i<edge.points.length;i++){
          const a=edge.points[i-1],b=edge.points[i],dx=(b[0]-a[0])*sx,dy=b[1]-a[1];
          const t=Math.max(0,Math.min(1,((point[0]-a[0])*sx*dx+(point[1]-a[1])*dy)/(dx*dx+dy*dy||1)));
          const native=[a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1])],km=dist(point,native);
          if(!best||km<best.snapKm)best={native,edge:index,segment:i,t,snapKm:km,name:edge.kind==='sea'?'海路位置':'路网位置'};
        }});
      if(!best)throw Error('没有适合该交通方式的通行网络。');
      return best;
    }
    function edgeArc(edge,to,reverse,points=edge.points){
      const path=reverse?points.slice().reverse():points;
      return {to,kind:edge.kind,port:edge.port,points:path,distanceKm:path.slice(1).reduce((sum,p,i)=>sum+dist(path[i],p),0),
        hours:0,ascentM:reverse?edge.descentM:edge.ascentM,descentM:reverse?edge.ascentM:edge.descentM};
    }
    function road(start,end,mode){
      const nodes=data.nodes.slice(),adj=nodes.map(()=>[]);
      const measure=(edge,station)=>{
        const result=Array(8).fill(0);
        for(let i=0;i<station.segment;i++){
          const part=edge.segments[i],fraction=i===station.segment-1?station.t:1;
          for(let j=0;j<8;j++)result[j]+=part[j]*fraction;
        }return result;
      };
      const add=(edge,a,b,points,from,to)=>{
        const f=edgeArc(edge,b,false,points),r=edgeArc(edge,a,true,points);
        const before=measure(edge,from),after=measure(edge,to),metrics=after.map((v,i)=>Math.max(0,v-before[i]));
        if(edge.kind==='sea')f.hours=r.hours=metrics[0]/mode.speedKmh;
        else {
          const walking=mode.id==='walk'||mode.kind==='ship';
          f.hours=metrics[walking?4:mode.id==='horse'?6:7];r.hours=metrics[walking?5:mode.id==='horse'?6:7];
          if(edge.kind==='port'){f.hours+=1;r.hours+=1;}
        }
        f.dailyHours=r.dailyHours=edge.kind==='sea'?mode.dailyHours:8;
        for(const arc of [f,r])arc.arriveAt=elapsed=>arrival(elapsed,arc.hours,arc.dailyHours);
        f.ascentM=r.descentM=metrics[1];f.descentM=r.ascentM=metrics[2];
        adj[a].push(f);adj[b].push(r);
      };
      const split=new Map();
      for(const station of [start,end])if(station.edge!==undefined){
        station.node=nodes.length;nodes.push(station.native);adj.push([]);
        if(!split.has(station.edge))split.set(station.edge,[]);split.get(station.edge).push(station);
      }
      data.edges.forEach((edge,index)=>{
        if(!permitted(edge,mode))return;
        const cuts=(split.get(index)||[]).sort((a,b)=>a.segment-b.segment||a.t-b.t);
        let a=edge.a,previous=0,first=edge.points[0],station={segment:1,t:0};
        for(const cut of cuts){const p=[first,...edge.points.slice(previous+1,cut.segment),cut.native];
          if(p.length>=2)add(edge,a,cut.node,p,station,cut);a=cut.node;previous=cut.segment-1;first=cut.native;station=cut;}
        const p=[first,...edge.points.slice(previous+1)];if(p.length>=2)add(edge,a,edge.b,p,station,{segment:edge.points.length-1,t:1});
      });
      return shortest(adj,start.node,end.node);
    }
    function flight(start,end,mode){
      const distanceKm=dist(start.native,end.native);
      const points=ruler.greatCircleArc(geo(start.native),geo(end.native),{maxStepDegrees:.25});
      return [{distanceKm,hours:distanceKm/mode.speedKmh,ascentM:0,descentM:0,
        parts:ruler.splitArc(points).map(part=>part.map(([lon,lat])=>[(lon+180)/360*data.shape[1],(90-lat)/180*data.shape[0]]))}];
    }
    function route(request){
      const mode=modes.get(request.mode);if(!mode)throw Error('该世界没有这种交通方式。');
      const start=snap(request.start,mode),end=snap(request.end,mode);
      const arcs=mode.kind==='flight'?flight(start,end,mode):road(start,end,mode);
      const distanceKm=arcs.reduce((sum,a)=>sum+a.distanceKm,0),hours=arcs.reduce((sum,a)=>sum+a.hours,0);
      const restCount=mode.kind==='flight'?Math.max(0,Math.ceil(hours/mode.dailyHours)-1):0;
      const elapsedHours=mode.kind==='flight'?arrival(0,hours,mode.dailyHours)
        :arcs.reduce((elapsed,arc)=>arrival(elapsed,arc.hours,arc.dailyHours),0);
      const travelDays=elapsedHours/data.dayHours;
      return {mode:mode.id,kind:mode.kind,label:mode.label,start,end,distanceKm,directKm:dist(start.native,end.native),hours,elapsedHours,travelDays,restCount,
        ascentM:arcs.reduce((s,a)=>s+a.ascentM,0),descentM:arcs.reduce((s,a)=>s+a.descentM,0),
        seaDistanceKm:arcs.filter(a=>a.kind==='sea').reduce((s,a)=>s+a.distanceKm,0),
        ports:arcs.filter(a=>a.kind==='port').map(a=>a.port.name),
        parts:arcs.flatMap(a=>a.parts||[a.points]),assumptions:data.profile.assumptions,
        dailyHours:mode.dailyHours,dailyRangeKm:mode.speedKmh*mode.dailyHours};
    }
    return {route,summary:()=>({cities:data.cities.map(({id,name,native,node,portNode})=>({id,name,native,node,portNode})),profile:data.profile,
      nodes:data.nodes.length,edges:data.edges.length})};
  }
  if(typeof module!=='undefined'&&module.exports)module.exports={create,shortest};
  else {let network=null;root.onmessage=async({data})=>{try{
    if(data.action==='load'){const r=await fetch('navigation-network.json');if(!r.ok)throw Error(`导航数据读取失败 (${r.status})`);network=create(await r.json());
      root.postMessage({id:data.id,result:network.summary()});}
    else {if(!network)throw Error('导航网络尚未加载');root.postMessage({id:data.id,result:network.route(data.request)});}
  }catch(error){root.postMessage({id:data.id,error:error.message});}};}
})(typeof self!=='undefined'?self:globalThis);
