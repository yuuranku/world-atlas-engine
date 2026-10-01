/* Generate one city's hills, drainage and fields from its saved parent terrain. */
(() => {
  'use strict';
  const M = Math;
  const colours = ['#e0e8cf','#d3dfbe','#c3d0ae','#c3c5a1','#cfbf9e','#d4b79b','#bda48e'];
  const clamp = (value, low, high) => M.max(low, M.min(high, value));
  const round = value => M.round(value * 1e7) / 1e7;
  const random = (seed, salt) => {
    let value = (seed ^ M.imul(salt + 1, 0x9e3779b9)) >>> 0;
    value = M.imul(value ^ value >>> 16, 0x21f0aaad);
    value = M.imul(value ^ value >>> 15, 0x735a2d97);
    return ((value ^ value >>> 15) >>> 0) / 4294967296;
  };

  function sample(values, rows, columns, bounds, row, column) {
    const x = clamp((column - bounds.west) / (bounds.east - bounds.west) * (columns - 1), 0, columns - 1);
    const y = clamp((row - bounds.north) / (bounds.south - bounds.north) * (rows - 1), 0, rows - 1);
    const x0 = M.floor(x), y0 = M.floor(y), x1 = M.min(columns - 1, x0 + 1), y1 = M.min(rows - 1, y0 + 1);
    const fx = x - x0, fy = y - y0;
    return (values[y0 * columns + x0] * (1 - fx) + values[y0 * columns + x1] * fx) * (1 - fy)
      + (values[y1 * columns + x0] * (1 - fx) + values[y1 * columns + x1] * fx) * fy;
  }

  function nearest(values, rows, columns, bounds, point) {
    const x = clamp(M.round((point.column - bounds.west) / (bounds.east - bounds.west) * (columns - 1)), 0, columns - 1);
    const y = clamp(M.round((point.row - bounds.north) / (bounds.south - bounds.north) * (rows - 1)), 0, rows - 1);
    return values[y * columns + x];
  }

  const lineCache=new WeakMap();
  function distanceToLine(point, points, metric) {
    let lines=lineCache.get(points);
    if(!lines){lines=points.slice(1).map((b,i)=>{const a=points[i],dx=(b.column-a.column)*metric.column,dy=(b.row-a.row)*metric.row;
      return {a,b,dx,dy,length:dx*dx+dy*dy||1};});lineCache.set(points,lines);}
    let best=Infinity,height=0,widthMetres=0;
    for(const {a,b,dx,dy,length} of lines){
      const px=(point.column-a.column)*metric.column,py=(point.row-a.row)*metric.row;
      const bx=M.max(0,M.min(0,dx)-px,px-M.max(0,dx)),by=M.max(0,M.min(0,dy)-py,py-M.max(0,dy));
      if(bx*bx+by*by>=best)continue;
      const t=clamp((px*dx+py*dy)/length,0,1),x=px-dx*t,y=py-dy*t,d=x*x+y*y;
      if(d<best){best=d;height=(a.elevationMetres||0)*(1-t)+(b.elevationMetres||0)*t;
        widthMetres=(a.widthMetres||0)*(1-t)+(b.widthMetres||0)*t;}
    }return {distance:M.sqrt(best),elevation:height,widthMetres};
  }

  // This is the same diagonal used by the rendered terrain triangles.
  function elevationAt(surface,point,values=surface.elevationMetres){
    const b=surface.bounds,n=surface.columns,m=surface.rows,
      x=clamp((point.column-b.west)/(b.east-b.west)*(n-1),0,n-1),
      y=clamp((point.row-b.north)/(b.south-b.north)*(m-1),0,m-1),
      i=M.min(n-2,M.floor(x)),j=M.min(m-2,M.floor(y)),u=x-i,v=y-j,k=j*n+i,z=values;
    return u>=v?z[k]*(1-u)+z[k+1]*(u-v)+z[k+n+1]*v:z[k]*(1-v)+z[k+n]*(v-u)+z[k+n+1]*u;
  }
  function isDry(surface,point){return elevationAt(surface,point)>0&&(!surface.lakeDepths||elevationAt(surface,point,surface.lakeDepths)<0);}

  function generate(recipe, context, {seed} = {}) {
    if (!context || context.schema !== 'city-site-context-v1' || !Number.isSafeInteger(seed)) {
      throw new TypeError('City terrain needs its saved metre context and a generation seed');
    }
    const count = context.rows * context.columns;
    if (!Array.isArray(context.elevationMetres) || context.elevationMetres.length !== count
        || !context.elevationMetres.every(Number.isFinite) || !Array.isArray(context.water) || context.water.length !== count) {
      throw new TypeError('Invalid saved city terrain samples');
    }
    seed >>>= 0;
    const SIZE=recipe.harbor||context.source.kind==='river'&&context.source.referenceWidthMetres>=18||context.radiusKm/M.max(.15,recipe.urban.radiusKm)>10?129:65;
    const bounds = {...context.bounds}, metric = context.gridCellKilometres;
    const center = recipe.location, radiusKm = context.radiusKm, cityRadius = recipe.urban.radiusKm;
    const form=landform(recipe,context);
    const widthKm = (bounds.east - bounds.west) * metric.column;
    const heightKm = (bounds.south - bounds.north) * metric.row;
    const stepKm = M.max(widthKm, heightKm) / (SIZE - 1);
    const elevation = [], parent = [], land = [], coordinates = [];
    const phase = random(seed, 17) * M.PI * 2;
    const parentRange = M.max(...context.elevationMetres) - M.min(...context.elevationMetres);
    const amplitude = clamp(8 + parentRange * .12, 8, 90);
    for (let row = 0; row < SIZE; row++) for (let column = 0; column < SIZE; column++) {
      const u = column / (SIZE - 1), v = row / (SIZE - 1);
      const point = {row: bounds.north + v * (bounds.south - bounds.north), column: bounds.west + u * (bounds.east - bounds.west)};
      const original = sample(context.elevationMetres, context.rows, context.columns, bounds, point.row, point.column);
      const dry = original > 0;
      const envelope = M.sin(M.PI * u) ** 2 * M.sin(M.PI * v) ** 2;
      let detail = amplitude * envelope * (
        M.sin(u * 7.2 + phase) * M.cos(v * 5.4 - phase) * .65
        + M.sin(u * 13.1 + v * 9.5 + phase * 2) * .22
        + M.sin(u * 26.3 - phase) * M.cos(v * 22.7 + phase) * .09
        + M.sin(u * 42.4 + v * 33.5 - phase) * .04);
      const x=(point.column-center.column)*metric.column,y=(point.row-center.row)*metric.row,
        across=-M.sin(form.axisRadians)*x+M.cos(form.axisRadians)*y;
      if(form.kind==='valley')detail+=envelope*M.min(90,parentRange*.22)*(1-M.exp(-((across/M.max(.25,cityRadius*.38))**2)));
      if(form.kind==='ridge')detail+=envelope*M.min(60,parentRange*.12)*M.exp(-((across/M.max(.3,cityRadius*.42))**2));
      // Only interior dry ground is refined. Zero-height coastline and the
      // patch boundary retain the exact parent field; water is never displaced.
      elevation.push(dry ? M.max(.01, original + detail*M.min(1,original/35)) : original);
      parent.push(original); land.push(dry); coordinates.push(point);
    }
    const coastal=coastalDetail(recipe,context,elevation,coordinates,metric,seed,stepKm);
    for(let i=0;i<land.length;i++)land[i]=elevation[i]>0;

    const source = context.source;
    const groundwater = source.kind === 'groundwater';
    const riverMode=!groundwater&&recipe.siteType==='river-city'&&recipe.population?.estimate>=3000
      &&['plain','hills','valley'].includes(form.kind)&&random(seed,403)>.32?'through':'edge';
    const channels = [],confluences=[];
    let lakes=[],waterMask=new Array(SIZE*SIZE).fill(0),lakeDepths=new Array(SIZE*SIZE).fill(-1000);
    if (!groundwater) {
      let dx = (source.location.column - center.column) * metric.column;
      let dy = (source.location.row - center.row) * metric.row;
      if (M.hypot(dx, dy) < .001) {
        dx = sample(elevation, SIZE, SIZE, bounds, center.row, bounds.west) - sample(elevation, SIZE, SIZE, bounds, center.row, bounds.east);
        dy = sample(elevation, SIZE, SIZE, bounds, bounds.north, center.column) - sample(elevation, SIZE, SIZE, bounds, bounds.south, center.column);
      }
      const length = M.hypot(dx, dy) || 1;
      if (M.hypot(dx, dy) < .001) {dx = M.cos(phase); dy = M.sin(phase);}
      else {dx /= length; dy /= length;}
      if(form.kind==='valley'){dx=M.cos(form.axisRadians);dy=M.sin(form.axisRadians);}
      const points = [];
      const extent = M.hypot(widthKm,heightKm);
      // Put the valley on the side opposite the main road approaches. Without
      // a verified bridge, a supply creek must not divide the settlement.
      const roadSide=(recipe.transport.corridors||[]).filter(route=>route.kind==='road').reduce((sum,route)=>sum+route.points.reduce((total,p)=>
        total+(p.column-center.column)*metric.column*dy-(p.row-center.row)*metric.row*dx,0),0);
      const bank=roadSide>0?-1:1;
      for (let index = 0; index < 129; index++) {
        const t = index / 128, along = (t * 2 - 1) * extent;
        const meander=cityRadius*.28*M.sin(along/M.max(.6,cityRadius*1.3)+phase)
          +cityRadius*.10*M.sin(along/M.max(.4,cityRadius*.7)-phase);
        const edge=form.kind==='plain'?recipe.harbor?1.42:.80:M.max(.36,form.crossScale*1.35);
        const side = riverMode==='through'?bank*cityRadius*.26+meander*.32:
          bank*(cityRadius*edge+meander*.28);
        const point = {row: center.row + (dy * along + dx * side) / metric.row,
          column: center.column + (dx * along - dy * side) / metric.column};
        point.elevationMetres = sample(elevation, SIZE, SIZE, bounds, point.row, point.column);
        points.push(point);
      }
      const terrain={rows:SIZE,columns:SIZE,bounds,elevationMetres:elevation};
      const runs=landRuns(points,terrain,metric,stepKm);
      // An island supply stream starts on high ground; two ocean endpoints
      // cannot force a channel through the whole island at sea level.
      for(let run of runs){
        if(elevationAt(terrain,run[0])<.05&&elevationAt(terrain,run.at(-1))<.05){
          const summit=run.reduce((best,p,i)=>p.elevationMetres>run[best].elevationMetres?i:best,0);
          const left=run.slice(0,summit+1).reverse(),right=run.slice(summit);
          run=distanceToLine(center,left,metric).distance<distanceToLine(center,right,metric).distance?left:right;
        }else if(run[0].elevationMetres<run.at(-1).elevationMetres)run.reverse();
        if(run.length<2)continue;
        let previous=Infinity;
        const reference=M.max(2,source.referenceWidthMetres||.5*M.sqrt(radiusKm*radiusKm*.6));
        for(let i=0;i<run.length;i++){
          const p=run[i],t=i/(run.length-1),z=elevationAt(terrain,p),
            ahead=run[M.min(run.length-1,i+1)],behind=run[M.max(0,i-1)],
            reach=M.hypot((ahead.column-behind.column)*metric.column,(ahead.row-behind.row)*metric.row),
            grade=M.abs(ahead.elevationMetres-behind.elevationMetres)/M.max(1,reach*1000),
            confinement=clamp(1-grade*2.5,.55,1.12),mouth=run.at(-1).elevationMetres<.05?1+M.exp(-(((1-t)/.10)**2))*.6:1;
          p.widthMetres=reference*(.8+.24*t)*confinement*mouth*(1+.10*M.sin(t*12+phase)+.035*M.sin(t*29-phase));
          p.elevationMetres=z<=0?0:M.max(.02,M.min(z-M.min(4,z*.12),previous));previous=p.elevationMetres;
        }
        channels.push({points:run,widthMetres:M.max(...run.map(p=>p.widthMetres)),kind:source.kind==='river'?'river':'stream',
          widthModel:'catchment-reference-with-local-confinement',mouth:run.at(-1).elevationMetres===0});
      }
      if(source.kind==='river'&&recipe.population?.estimate>=3000&&random(seed,2318)>.42){
        const junction=tributary(channels,elevation,SIZE,bounds,metric,center,cityRadius,seed);
        if(junction)confluences.push(junction);
      }
      const reservoirs=basinLakes(elevation,SIZE,bounds,metric,channels,center,cityRadius),
        reservoirSurface={rows:SIZE,columns:SIZE,bounds,elevationMetres:elevation,lakeDepths:reservoirs.lakeDepths};
      for(const channel of channels){
        for(const lake of reservoirs.lakes)for(const p of channel.points)if(elevationAt(reservoirSurface,p,reservoirs.lakeDepths)>0
          &&M.hypot((p.column-lake.anchor.column)*metric.column,(p.row-lake.anchor.row)*metric.row)<M.sqrt(lake.areaKm2)*2)
          p.elevationMetres=M.max(p.elevationMetres,lake.spillElevationMetres);
        // A reservoir raises its upstream water surface; it cannot create an
        // uphill reach. Carve the outlet at the spill sill, then recompute the
        // actual lake shore against the resulting terrain.
        for(let i=channel.points.length-2;i>=0;i--)channel.points[i].elevationMetres=M.max(channel.points[i].elevationMetres,channel.points[i+1].elevationMetres);
      }
      const drainage=channels.flatMap(ch=>ch.points);
      const valleyWidthKm = M.max(.09, cityRadius * .18);
      for (let index = 0; index < elevation.length; index++) {
        if (!land[index]||waterMask[index]||!drainage.length) continue;
        const point = coordinates[index], near = channels.map(ch=>distanceToLine(point,ch.points,metric)).reduce((a,b)=>a.distance<b.distance?a:b);
        const u = (point.column - bounds.west) / (bounds.east - bounds.west), v = (point.row - bounds.north) / (bounds.south - bounds.north);
        const envelope = M.sin(M.PI * u) ** 2 * M.sin(M.PI * v) ** 2;
        const valley = M.exp(-((near.distance / valleyWidthKm) ** 2)) * M.min(1, envelope * 4);
        elevation[index] = M.max(.01, elevation[index] - M.max(0, elevation[index] - near.elevation) * valley);
      }
      // Reclip against the final terrain after valley carving and rounding.
      terrain.elevationMetres=elevation.map(z=>M.round(z*100)/100);
      const clipped=channels.flatMap(ch=>landRuns(ch.points,terrain,metric,stepKm).map(run=>({...ch,points:run,mouth:run.at(-1).elevationMetres===0})));
      channels.splice(0,channels.length,...clipped);
      ({lakes,waterMask,lakeDepths}=basinLakes(terrain.elevationMetres,SIZE,bounds,metric,channels,center,cityRadius));
    }
    const buildable = [], slopes = [], roadBuildable=[];
    for (let row = 0; row < SIZE; row++) for (let column = 0; column < SIZE; column++) {
      const index = row * SIZE + column;
      const gradientX = (elevation[row * SIZE + M.min(SIZE - 1, column + 1)] - elevation[row * SIZE + M.max(0, column - 1)]) / (widthKm / (SIZE - 1) * 2000);
      const gradientY = (elevation[M.min(SIZE - 1, row + 1) * SIZE + column] - elevation[M.max(0, row - 1) * SIZE + column]) / (heightKm / (SIZE - 1) * 2000);
      const slope = M.hypot(gradientX, gradientY);
      const floodplain = channels.some(channel => {const near=distanceToLine(coordinates[index],channel.points,metric);
        return near.distance<M.max(near.widthMetres/1000*1.5,stepKm*.72);});
      land[index]=land[index]&&!waterMask[index];
      slopes.push(slope); buildable.push(Number(land[index] && slope < .20 && !floodplain));
      roadBuildable.push(Number(land[index]&&slope<.65));
    }
    const surface = {rows: SIZE, columns: SIZE, bounds, elevationMetres: elevation.map(value => M.round(value * 100) / 100), slopes,buildable,roadBuildable,waterMask,lakeDepths};
    if(channels.length&&['hillside','valley','ridge'].includes(form.kind)){
      const ch=channels.reduce((a,b)=>a.points.length>b.points.length?a:b),a=ch.points[0],z=ch.points.at(-1);
      form.kind='valley';form.axisRadians=M.atan2((z.row-a.row)*metric.row,(z.column-a.column)*metric.column);
      form.crossScale=.27;form.alongScale=1.75;form.label='沿河谷与河岸台地';
    }else if(channels.length&&!recipe.harbor&&!['fortress','pass','oasis'].includes(recipe.siteType)){
      const ch=channels.reduce((a,b)=>a.points.length>b.points.length?a:b),a=ch.points[0],z=ch.points.at(-1);
      form.kind='riverbank';form.axisRadians=M.atan2((z.row-a.row)*metric.row,(z.column-a.column)*metric.column);
      form.crossScale=.62;form.alongScale=1.6;form.label='河岸缓坡台地';
    }
    if(recipe.harbor){
      const saved=recipe.harbor,n=saved.outward,at=d=>({column:saved.shore.column+n.x*d/metric.column,row:saved.shore.row+n.y*d/metric.row}),
        z=d=>elevationAt(surface,at(d));
      let low=-M.max(.6,stepKm*2,saved.distanceKm*.8),high=M.max(.6,stepKm*2);
      if(z(low)>0&&z(high)<=0){
        for(let k=0;k<24;k++){const mid=(low+high)/2;if(z(mid)>0)low=mid;else high=mid;}
        const shift=(low+high)/2,shore=at(shift),landPoint=at(shift-.07),seaPoint=at(shift+.12),harbor={...saved,shore,landPoint,seaPoint,
          setting:coastal.setting,access:[center,landPoint]};
        if(saved.sailing?.length){
          const a=seaPoint,z=saved.sailing[0],wet=Array.from({length:33},(_,i)=>interpolate(a,z,i/32)).every(p=>elevationAt(surface,p)<=0);
          harbor.sailing=wet?[seaPoint,...saved.sailing]:[];
        }
        recipe={...recipe,harbor,transport:{...recipe.transport,corridors:recipe.transport.corridors.map(route=>route.id===recipe.id+'-harbor-access'?{...route,points:harbor.access}:route)}};
      }
    }
    if(!recipe.harbor&&source.kind==='river'&&recipe.population?.estimate>=3000){
      const harbor=inlandHarbor(recipe,surface,channels,confluences,metric,context.nameRoots,seed);
      if(harbor)recipe={...recipe,harbor,transport:{...recipe.transport,corridors:[...recipe.transport.corridors,
        {id:recipe.id+'-harbor-access',kind:'road',points:harbor.access}]}};
    }
    if(recipe.harbor){
      const h=recipe.harbor,n=h.outward,t={x:-n.y,y:n.x},factor=clamp(M.sqrt(recipe.population.estimate/22000),.34,1.8),
        at=(x,y)=>({column:h.shore.column+(t.x*x+n.x*y)*factor/metric.column,row:h.shore.row+(t.y*x+n.y*y)*factor/metric.row});
      recipe={...recipe,harbor:{...h,reservedLand:[[-.42,-.17],[-.42,.012],[.42,.012],[.42,-.17]].map(([x,y])=>at(x,y))}};
    }
    const relief = contourPaths(coordinates, elevation, land, metric);
    const waterSource = groundwater
      ? {kind: 'groundwater', label: recipe.siteType === 'oasis' ? '绿洲井泉' : '山泉与井水', location: {...center}}
      : {kind: source.kind === 'lake' ? 'lake' : source.kind === 'river' ? 'tributary' : 'spring-fed-stream',
          label: source.kind === 'lake' ? '湖泊与溪流供水' : source.kind === 'river' ? '河谷支流供水' : '山泉溪流供水',
          location: channels.length ? channels[0].points[M.floor(channels[0].points.length / 2)] : {...center}};
    if(riverMode==='through'&&channels.length){
      const line=channels[0].points,a=line[0],b=line.at(-1),dx=(b.column-a.column)*metric.column,dy=(b.row-a.row)*metric.row,len=M.hypot(dx,dy),
        across={x:-dy/len,y:dx/len};
      const points=[-1,1].map(sign=>({column:center.column+across.x*cityRadius*1.4*sign/metric.column,
        row:center.row+across.y*cityRadius*1.4*sign/metric.row}));
      recipe={...recipe,transport:{...recipe.transport,corridors:[...recipe.transport.corridors,{id:'urban-river-spine',kind:'road',points}]}};
    }
    const bridges=crossings(recipe,surface,context,channels);
    const network=ruralNetwork(recipe,surface,slopes,channels,metric,seed,bridges),ruralRoads=network.ruralRoads;
    recipe={...recipe,transport:{...recipe.transport,corridors:network.corridors}};
    if(recipe.harbor){const route=network.corridors.find(r=>r.id===recipe.id+'-harbor-access');
      recipe={...recipe,harbor:{...recipe.harbor,access:route?.points||[]}};}
    const roadRecipe={...recipe,localSite:{form},transport:{...recipe.transport,corridors:[...network.corridors,...ruralRoads]}};
    const farmland=fields(roadRecipe,surface,slopes,metric,channels,seed);
    const localSite = {schema: 'city-local-site-v1', bounds, radiusKm, form,riverMode,...relief,coastal,channels,lakes,confluences,waterSource, surface,
      nameRoots:context.nameRoots,latitudeDegrees:context.latitudeDegrees,
      farmland, ruralRoads, bridges,...landscape(roadRecipe,surface,slopes,coordinates,land,seed,farmland,channels),
      model: {kind: 'generated-local-refinement', parent: 'saved-world-metre-terrain', boundaryBlended: true, coastlinePreserved: !coastal.refined},
    };
    localSite.annotations=geographicNames(recipe,context,localSite,metric);
    return {...recipe, seed, localSite};
  }

  function interpolate(a,b,t){return {row:a.row+(b.row-a.row)*t,column:a.column+(b.column-a.column)*t,
    elevationMetres:(a.elevationMetres||0)+(b.elevationMetres-a.elevationMetres||0)*t,
    widthMetres:(a.widthMetres||0)+((b.widthMetres||0)-(a.widthMetres||0))*t};}

  function inlandHarbor(recipe,surface,channels,confluences,metric,roots,seed){
    const distance=(a,b)=>M.hypot((a.column-b.column)*metric.column,(a.row-b.row)*metric.row),
      target=confluences[0]?.anchor||recipe.location;
    const candidates=channels.flatMap(channel=>channel.points.slice(1,-1).map((p,i)=>({p,i:i+1,channel})))
      .filter(c=>c.p.widthMetres>=25).sort((a,b)=>distance(a.p,target)-distance(b.p,target));
    for(const {p,i,channel} of candidates.slice(0,24)){
      if(distance(p,recipe.location)>recipe.urban.radiusKm*1.7)continue;
      const a=channel.points[i-1],z=channel.points[i+1],dx=(z.column-a.column)*metric.column,dy=(z.row-a.row)*metric.row,len=M.hypot(dx,dy),half=p.widthMetres/2000;
      if(!len)continue;
      for(const sign of [-1,1]){
        const outward={x:dy/len*sign,y:-dx/len*sign},at=d=>({column:p.column-outward.x*d/metric.column,row:p.row-outward.y*d/metric.row}),
          shore=at(half),landPoint=at(half+.035),seaPoint=at(half*.25);
        if(!isDry(surface,landPoint)||channels.some(ch=>{const q=distanceToLine(landPoint,ch.points,metric);return q.distance<q.widthMetres/2000+.015;}))continue;
        return {kind:'river',name:roots[M.floor(random(seed,9021)*roots.length)]+'埠',shore,landPoint,seaPoint,outward,
          setting:confluences.length?'confluence':'riverbank',shelter:.8,channelWidthMetres:p.widthMetres,
          distanceKm:distance(shore,recipe.location),access:[recipe.location,landPoint],sailing:[]};
      }
    }return null;
  }

  function priorityQueue(){
    const heap=[];
    return {get length(){return heap.length;},put(index,cost){let i=heap.length;heap.push([index,cost]);while(i){const p=(i-1)>>1;
      if(heap[p][1]<=cost)break;heap[i]=heap[p];i=p;}heap[i]=[index,cost];},take(){const first=heap[0],last=heap.pop();
      if(heap.length){let i=0;while(i*2+1<heap.length){let j=i*2+1;if(j+1<heap.length&&heap[j+1][1]<heap[j][1])j++;
        if(heap[j][1]>=last[1])break;heap[i]=heap[j];i=j;}heap[i]=last;}return first;}};
  }

  function basinLakes(elevation,n,bounds,metric,channels,center,radius){
    const fill=new Float64Array(n*n).fill(Infinity),queue=priorityQueue(),visited=new Uint8Array(n*n),waterMask=new Array(n*n).fill(0),lakeDepths=new Array(n*n).fill(-1000),lakes=[];
    const dx=(bounds.east-bounds.west)*metric.column/(n-1),dy=(bounds.south-bounds.north)*metric.row/(n-1),
      at=i=>({column:bounds.west+i%n/(n-1)*(bounds.east-bounds.west),row:bounds.north+M.floor(i/n)/(n-1)*(bounds.south-bounds.north)});
    for(let i=0;i<n*n;i++)if(i%n===0||i%n===n-1||i<n||i>=n*(n-1)||elevation[i]<=0){fill[i]=elevation[i];queue.put(i,fill[i]);}
    while(queue.length){const [i,h]=queue.take();if(h!==fill[i])continue;const x=i%n,y=M.floor(i/n);
      for(const [sx,sy] of [[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,1],[1,-1],[-1,-1]]){
        if(x+sx<0||x+sx>=n||y+sy<0||y+sy>=n)continue;const j=i+sy*n+sx,next=M.max(h,elevation[j]);
        if(next<fill[j]){fill[j]=next;queue.put(j,next);}
      }
    }
    const candidates=Array.from({length:n*n},(_,i)=>i).filter(i=>fill[i]-elevation[i]>2.5&&elevation[i]>0)
      .sort((a,b)=>(fill[b]-elevation[b])-(fill[a]-elevation[a]));
    for(const seed of candidates){
      if(visited[seed]||lakes.length>=3)continue;const level=fill[seed]-.05,component=[seed];visited[seed]=1;
      for(let k=0;k<component.length;k++){const i=component[k],x=i%n,y=M.floor(i/n);
        for(const [sx,sy] of [[1,0],[-1,0],[0,1],[0,-1]]){if(x+sx<=0||x+sx>=n-1||y+sy<=0||y+sy>=n-1)continue;
          const j=i+sy*n+sx;if(!visited[j]&&elevation[j]>0&&elevation[j]<level){visited[j]=1;component.push(j);}}
      }
      const area=component.length*dx*dy,anchor=at(seed),distance=M.hypot((anchor.column-center.column)*metric.column,(anchor.row-center.row)*metric.row);
      if(area<.02||area>4||distance<radius*.80||distance>radius*4.5
        ||!channels.some(ch=>distanceToLine(anchor,ch.points,metric).distance<M.max(dx,dy)*2+ch.widthMetres/1000))continue;
      const cells=new Set(component),patches=[];
      for(const i of component)waterMask[i]=1;
      const tiles=new Set(component.flatMap(i=>[i,i-1,i-n,i-n-1]));
      for(const i of tiles){if(i<0||i>=n*(n-1)||i%n===n-1)continue;
        for(const tri of [[i,i+1,i+n+1],[i,i+n+1,i+n]]){
          const vertices=tri.map(j=>({...at(j),value:cells.has(j)?M.max(.01,level-elevation[j]):-M.max(.01,elevation[j]-level)})),poly=[];
          for(let k=0;k<tri.length;k++)lakeDepths[tri[k]]=M.max(lakeDepths[tri[k]],vertices[k].value);
          for(let k=0;k<3;k++){const a=vertices[k],b=vertices[(k+1)%3];if(a.value>0)poly.push(a);
            if((a.value>0)!==(b.value>0))poly.push(interpolate(a,b,a.value/(a.value-b.value)));}
          if(poly.length>=3)patches.push(poly);
        }
      }
      const spill=component.flatMap(i=>{const x=i%n,y=M.floor(i/n);return [[1,0],[-1,0],[0,1],[0,-1]]
        .filter(([sx,sy])=>x+sx>=0&&x+sx<n&&y+sy>=0&&y+sy<n).map(([sx,sy])=>i+sy*n+sx);})
        .filter(i=>!cells.has(i)).sort((a,b)=>elevation[a]-elevation[b])[0];
      lakes.push({anchor,patches,waterLevelMetres:M.round(level*100)/100,spillElevationMetres:fill[seed],
        outlet:at(spill),areaKm2:area,maximumDepthMetres:level-elevation[seed],kind:'basin-lake'});
    }
    return {lakes,waterMask,lakeDepths};
  }

  function tributary(channels,elevation,n,bounds,metric,center,radius,seed){
    if(!channels.length)return null;const main=channels.reduce((a,b)=>a.points.length>b.points.length?a:b),
      distance=(a,b)=>M.hypot((a.column-b.column)*metric.column,(a.row-b.row)*metric.row),
      join=main.points.slice(2,-2).sort((a,b)=>distance(a,center)-distance(b,center))[0];
    if(!join||join.widthMetres<18)return null;
    const at=i=>({column:bounds.west+i%n/(n-1)*(bounds.east-bounds.west),row:bounds.north+M.floor(i/n)/(n-1)*(bounds.south-bounds.north)}),
      index=p=>clamp(M.round((p.row-bounds.north)/(bounds.south-bounds.north)*(n-1)),0,n-1)*n+clamp(M.round((p.column-bounds.west)/(bounds.east-bounds.west)*(n-1)),0,n-1),
      start=index(join),cost=new Float64Array(n*n).fill(Infinity),parent=new Int32Array(n*n).fill(-1),queue=priorityQueue();
    cost[start]=0;queue.put(start,0);
    while(queue.length){const [i,d]=queue.take();if(d!==cost[i])continue;const x=i%n,y=M.floor(i/n),p=at(i);
      if(distance(p,join)>radius*3.6)continue;
      for(const [sx,sy] of [[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,1],[1,-1],[-1,-1]]){
        if(x+sx<0||x+sx>=n||y+sy<0||y+sy>=n)continue;const j=i+sy*n+sx,q=at(j),length=distance(p,q);
        if(elevation[j]<=0||distance(q,join)>.25&&distanceToLine(q,main.points,metric).distance<join.widthMetres/2000+.05)continue;
        const descent=M.max(0,elevation[i]-elevation[j]),next=d+length*(1+descent/8+(M.abs(elevation[j]-elevation[i])/(length*1000))**2*5);
        if(next<cost[j]){cost[j]=next;parent[j]=i;queue.put(j,next);}
      }
    }
    let best=-1,score=-Infinity;
    for(let i=0;i<n*n;i++){const p=at(i),reach=distance(p,join);if(!Number.isFinite(cost[i])||reach<radius*1.4||reach>radius*3.4||elevation[i]<elevation[start]+5)continue;
      const value=(elevation[i]-elevation[start])/(cost[i]+.5)*(1+random(seed,i+2870)*.1);if(value>score){score=value;best=i;}}
    if(best<0)return null;
    const points=[];for(let i=best;i>=0&&i!==start;i=parent[i])points.push({...at(i),elevationMetres:elevation[i]});points.push({...join});
    if(points.length<4)return null;
    let previous=Infinity;
    for(let i=0;i<points.length;i++){points[i].elevationMetres=M.min(previous,points[i].elevationMetres);previous=points[i].elevationMetres;
      points[i].widthMetres=M.max(3,join.widthMetres*(.12+.25*i/(points.length-1)));}
    channels.push({points,widthMetres:points.at(-1).widthMetres,kind:'river',widthModel:'tributary-catchment',mouth:false});
    return {anchor:{...join},mainChannel:0,tributaryChannel:channels.length-1};
  }

  function landRuns(points,surface,metric,stepKm){
    const runs=[];let run=[];
    const flush=()=>{if(run.length>1)runs.push(run);run=[];};
    for(let i=1;i<points.length;i++){
      const edge=clipEdge(points[i-1],points[i],surface.bounds);if(!edge){flush();continue;}
      const [a,b]=edge,len=M.hypot((b.column-a.column)*metric.column,(b.row-a.row)*metric.row),count=M.max(1,M.ceil(len/(stepKm*.4)));
      for(let j=1;j<=count;j++){
        const p=interpolate(a,b,(j-1)/count),q=interpolate(a,b,j/count),zp=elevationAt(surface,p),zq=elevationAt(surface,q);
        if(zp<=0&&zq<=0){flush();continue;}
        if((zp>0)!==(zq>0)){
          let low=0,high=1;for(let k=0;k<22;k++){const t=(low+high)/2;if((elevationAt(surface,interpolate(p,q,t))>0)===(zp>0))low=t;else high=t;}
          const mouth=interpolate(p,q,(low+high)/2);mouth.elevationMetres=0;
          if(zp>0){if(!run.length)run.push(p);run.push(mouth);flush();}else{flush();run.push(mouth,q);}
        }else{if(!run.length)run.push(p);run.push(q);}
      }
    }flush();return runs;
  }

  function coastalDetail(recipe,context,elevation,coordinates,metric,seed,stepKm){
    const harbor=recipe.harbor,result={refined:false,setting:harbor?.setting||'open-coast',islands:[]};
    if(!harbor||harbor.kind!=='sea')return result;
    const b=context.bounds,c=recipe.location,n=harbor.outward,t={x:-n.y,y:n.x},r=recipe.urban.radiusKm,
      relief=M.max(...context.elevationMetres)-M.min(...context.elevationMetres),
      rugged=relief>180&&['steep','escarpment','rolling'].includes(recipe.terrain?.slopeClass),
      inlet=rugged&&(harbor.shelter>.58||random(seed,2101)<.40),
      setback=M.hypot((harbor.shore.column-c.column)*metric.column,(harbor.shore.row-c.row)*metric.row),
      length=M.min(setback*.48,M.max(stepKm*2,r*.52)),halfWidth=M.max(stepKm*1.2,M.min(r*.24,.30));
    if(inlet&&length>stepKm){
      for(let i=0;i<coordinates.length;i++){
        const p=coordinates[i],x=(p.column-harbor.shore.column)*metric.column,y=(p.row-harbor.shore.row)*metric.row,
          along=x*n.x+y*n.y,across=x*t.x+y*t.y;
        if(along< -length||along>stepKm*2||M.abs(across)>halfWidth*2||M.hypot((p.column-c.column)*metric.column,(p.row-c.row)*metric.row)<r*.24)continue;
        const taper=clamp((along+length)/(length+stepKm),0,1),w=halfWidth*M.sqrt(taper),
          edge=M.abs(across)-w,depth=clamp(edge/halfWidth*24,-24,24);
        elevation[i]=M.min(elevation[i],depth);
      }
      result.refined=true;result.setting=context.latitudeDegrees>45&&relief>300?'fjord':'sheltered-inlet';
    }
    // Skerries are refinements of shallow coastal seabed on rugged coasts.
    // The berth, entrance and saved sailing corridor remain open water.
    if(!rugged&&recipe.siteType!=='island-port')return result;
    const count=1+M.floor(random(seed,2110)*6);
    for(let k=0;k<count;k++){
      const salt=2111+k*71,across=(random(seed,salt)*2-1)*M.max(1.4,r*3.8),along=M.max(.3,r*.4)*(1+random(seed,salt+1)*3.8),
        p={column:harbor.shore.column+(n.x*along+t.x*across)/metric.column,row:harbor.shore.row+(n.y*along+t.y*across)/metric.row},
        depth=sample(context.elevationMetres,context.rows,context.columns,b,p.row,p.column),
        rx=M.max(stepKm*2,M.min(.85,r*(.12+random(seed,salt+2)*.48))),ry=rx*(.25+random(seed,salt+3)*.68),
        heading=random(seed,salt+4)*M.PI*2,cos=M.cos(heading),sin=M.sin(heading),family=M.floor(random(seed,salt+5)*4),
        phase=random(seed,salt+6)*M.PI*2;
      if(depth>=-2||depth< -100||p.column<b.west+rx/metric.column||p.column>b.east-rx/metric.column||p.row<b.north+rx/metric.row||p.row>b.south-rx/metric.row
        ||result.islands.some(island=>M.hypot((island.anchor.column-p.column)*metric.column,(island.anchor.row-p.row)*metric.row)<rx+island.radiusKm)
        ||harbor.sailing&&distanceToLine(p,harbor.sailing,metric).distance<rx+.15)continue;
      const peak=7+random(seed,salt+7)*57;let dryCount=0;
      for(let i=0;i<coordinates.length;i++){
        const q=coordinates[i],dx=(q.column-p.column)*metric.column,dy=(q.row-p.row)*metric.row,u=(dx*cos+dy*sin)/rx,v=(-dx*sin+dy*cos)/ry,
          angle=M.atan2(v,u),lobes=1+.18*M.sin(angle*(2+family)+phase)+.09*M.sin(angle*7-phase*1.7),
          rho=M.hypot(u,v)/lobes;
        let z=peak*(1-rho*rho);
        if(family===1)z=M.min(z,peak*(M.hypot(u-.43,v-.28)/.78-1));
        if(family===2)z=M.max(z,peak*.68*(1-((u+.72)**2/.55+(v-.36)**2/.66)));
        if(family===3)z=peak*(1-M.abs(v)/lobes-.63*M.abs(u)**1.7);
        if(z>0&&elevation[i]<=0&&(!harbor.sailing?.length||distanceToLine(q,harbor.sailing,metric).distance>.10)){
          elevation[i]=z;dryCount++;
        }
      }
      if(dryCount){
        const summit=coordinates.map((q,i)=>({q,i,d:M.hypot((q.column-p.column)*metric.column,(q.row-p.row)*metric.row)}))
          .filter(v=>v.d<rx*1.3&&elevation[v.i]>0).sort((a,z)=>elevation[z.i]-elevation[a.i])[0];
        result.islands.push({anchor:summit.q,heightMetres:M.round(peak),radiusKm:rx,shape:['skerry','crescent','paired-ridge','long-ridge'][family],shapeSeed:salt,
          areaKm2:dryCount*stepKm*stepKm});result.refined=true;
      }
    }return result;
  }

  function clipEdge(a,b,bounds){
    let low=0,high=1;const dx=b.column-a.column,dy=b.row-a.row;
    for(const [p,q] of [[-dx,a.column-bounds.west],[dx,bounds.east-a.column],[-dy,a.row-bounds.north],[dy,bounds.south-a.row]]){
      if(M.abs(p)<1e-12){if(q<0)return null;continue;}
      const t=q/p;if(p<0)low=M.max(low,t);else high=M.min(high,t);if(low>high)return null;
    }return [interpolate(a,b,low),interpolate(a,b,high)];
  }

  function ruralNetwork(recipe,surface,slopes,channels,metric,seed,bridges){
    const b=surface.bounds,n=surface.columns,c=recipe.location,r=recipe.urban.radiusKm;
    const index=p=>clamp(M.round((p.row-b.north)/(b.south-b.north)*(n-1)),0,n-1)*n+clamp(M.round((p.column-b.west)/(b.east-b.west)*(n-1)),0,n-1);
    const nodes=Array.from({length:n*n},(_,i)=>({row:b.north+M.floor(i/n)/(n-1)*(b.south-b.north),column:b.west+(i%n)/(n-1)*(b.east-b.west)}));
    const point=i=>nodes[i];
    const distance=(a,b)=>M.hypot((a.column-b.column)*metric.column,(a.row-b.row)*metric.row);
    const start=index(c),cost=new Float64Array(n*n).fill(Infinity),parent=new Int32Array(n*n).fill(-1),heap=[];
    const put=(i,d)=>{heap.push([i,d]);let k=heap.length-1;while(k){const p=(k-1)>>1;if(heap[p][1]<=d)break;heap[k]=heap[p];k=p;}heap[k]=[i,d];};
    const take=()=>{const first=heap[0],last=heap.pop();if(heap.length){let i=0;while(i*2+1<heap.length){let j=i*2+1;if(j+1<heap.length&&heap[j+1][1]<heap[j][1])j++;if(heap[j][1]>=last[1])break;heap[i]=heap[j];i=j;}heap[i]=last;}return first;};
    const allowed=Array.from({length:n*n},(_,i)=>surface.roadBuildable[i]===1&&!channels.some(ch=>{const near=distanceToLine(point(i),ch.points,metric);return near.distance<near.widthMetres/2000+.025;}));
    const links=new Map(),bridgeEdges=new Map();
    const waterEdges=channels.flatMap(ch=>ch.points.slice(1).map((z,i)=>({a:ch.points[i],z})));
    const waterBins=new Map(),binX=p=>clamp(M.floor((p.column-b.west)/(b.east-b.west)*(n-1)),0,n-1),
      binY=p=>clamp(M.floor((p.row-b.north)/(b.south-b.north)*(n-1)),0,n-1);
    const bins=(a,z)=>{const keys=[];for(let y=M.min(binY(a),binY(z));y<=M.max(binY(a),binY(z));y++)for(let x=M.min(binX(a),binX(z));x<=M.max(binX(a),binX(z));x++)keys.push(y*n+x);return keys;};
    for(const edge of waterEdges)for(const key of bins(edge.a,edge.z)){if(!waterBins.has(key))waterBins.set(key,[]);waterBins.get(key).push(edge);}
    const crossesWater=(a,z)=>bins(a,z).some(key=>(waterBins.get(key)||[]).some(({a:c,z:d})=>{
      if(M.max(a.column,z.column)<M.min(c.column,d.column)||M.min(a.column,z.column)>M.max(c.column,d.column)
        ||M.max(a.row,z.row)<M.min(c.row,d.row)||M.min(a.row,z.row)>M.max(c.row,d.row))return false;
      const dx=z.column-a.column,dy=z.row-a.row,sx=d.column-c.column,sy=d.row-c.row,den=dx*sy-dy*sx;
      if(M.abs(den)<1e-14)return false;const t=((c.column-a.column)*sy-(c.row-a.row)*sx)/den,u=((c.column-a.column)*dy-(c.row-a.row)*dx)/den;
      return t>=0&&t<=1&&u>=0&&u<=1;
    }));
    const bankNode=(p,opposite)=>{let best=-1,d=Infinity;for(let i=0;i<n*n;i++)if(allowed[i]){const q=point(i);
      if(opposite){const mx=(p.column+opposite.column)/2,my=(p.row+opposite.row)/2;
        if((q.column-mx)*(p.column-mx)*metric.column**2+(q.row-my)*(p.row-my)*metric.row**2<=0)continue;}
      const v=distance(q,p);if(v<d){best=i;d=v;}}return best;};
    for(const bridge of bridges){
      const a=bankNode(bridge.points[0],bridge.points[1]),z=bankNode(bridge.points[1],bridge.points[0]);if(a<0||z<0||a===z)continue;
      const safe=(from,to)=>{if(crossesWater(from,to))return false;for(let j=0;j<=8;j++){const p=interpolate(from,to,j/8);
        if(!isDry(surface,p)||sample(surface.slopes,n,n,b,p.row,p.column)>.25)return false;}return true;};
      if(!safe(point(a),bridge.points[0])||!safe(point(z),bridge.points[1]))continue;
      if(!links.has(a))links.set(a,[]);if(!links.has(z))links.set(z,[]);links.get(a).push(z);links.get(z).push(a);
      bridgeEdges.set(a+':'+z,bridge.points);bridgeEdges.set(z+':'+a,bridge.points.slice().reverse());
    }
    cost[start]=0;put(start,0);
    const dx=(b.east-b.west)*metric.column/(n-1),dy=(b.south-b.north)*metric.row/(n-1);
    while(heap.length){const [i,d]=take();if(d!==cost[i])continue;const x=i%n,y=M.floor(i/n);
      for(const j of links.get(i)||[]){const next=d+distance(point(i),point(j))*1.2;
        if(next<cost[j]){cost[j]=next;parent[j]=i;put(j,next);}}
      for(const [sx,sy] of [[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,1],[1,-1],[-1,-1],
        [1,2],[-1,2],[1,-2],[-1,-2],[2,1],[-2,1],[2,-1],[-2,-1]]){
        if(x+sx<0||x+sx>=n||y+sy<0||y+sy>=n)continue;const j=i+sy*n+sx;
        if(!allowed[j]||sx&&sy&&(!allowed[i+sx]||!allowed[i+sy*n]))continue;
        if(crossesWater(point(i),point(j)))continue;
        const length=M.hypot(sx*dx,sy*dy),grade=M.abs(surface.elevationMetres[i]-surface.elevationMetres[j])/(length*1000);
        if(grade>.12)continue;const next=d+length*(1+(grade/.035)**2*2+random(seed,j+67000)*.18);
        if(next<cost[j]){cost[j]=next;parent[j]=i;put(j,next);}
      }
    }
    const chainTo=i=>{const chain=[];for(;i!==start&&i>=0;i=parent[i])chain.push(i);chain.push(start);chain.reverse();return chain;};
    const pathTo=i=>{const chain=chainTo(i),points=[point(start)];for(let k=1;k<chain.length;k++){
      const bridge=bridgeEdges.get(chain[k-1]+':'+chain[k]);if(bridge)points.push(...bridge);points.push(point(chain[k]));}return points;};
    const corridors=[];
    for(const route of recipe.transport.corridors){
      if(route.kind!=='road'){corridors.push(route);continue;}
      const clipped=route.points.slice(1).flatMap((p,i)=>clipEdge(route.points[i],p,b)||[]);if(clipped.length<2)continue;
      const ends=[clipped[0],clipped.at(-1)],paths=[];
      for(const end of ends){const node=bankNode(end);
        if(node<0||!Number.isFinite(cost[node])||distance(point(node),end)>M.hypot(dx,dy)*1.8)continue;
        const tail=point(node);let safe=true;for(let t=0;t<=12;t++){const p=interpolate(tail,end,t/12);
          if(!isDry(surface,p)||channels.some(ch=>distanceToLine(p,ch.points,metric).distance<ch.widthMetres/2000+.004)){safe=false;break;}}
        const points=pathTo(node);points[0]={...c};if(safe)points.push(end);paths.push(points);
      }
      if(paths.length)corridors.push({...route,points:paths.length===2?paths[0].slice().reverse().concat(paths[1].slice(1)):paths[0]});
    }
    const roads=[],used=new Set();
    for(let k=0;k<10;k++){
      const angle=(k+random(seed,k+67100)*.55)*M.PI/5,reach=r*(2.3+random(seed,k+67200)*1.6),target={row:c.row+M.sin(angle)*reach/metric.row,column:c.column+M.cos(angle)*reach/metric.column};
      let best=-1,score=Infinity;for(let i=0;i<n*n;i++)if(Number.isFinite(cost[i])){const d=distance(point(i),target);if(d<score){score=d;best=i;}}
      if(best<0||score>r*.8||distance(point(best),c)<r*1.4)continue;
      const chain=chainTo(best);
      let run=[];const flush=()=>{if(run.length>1)roads.push({id:'country-road-'+roads.length,kind:'road',local:true,points:run});run=[];};
      for(let i=1;i<chain.length;i++){
        const a=chain[i-1],z=chain[i],key=[M.min(a,z),M.max(a,z)].join(':');
        if(used.has(key)||bridgeEdges.has(a+':'+z)){flush();continue;}used.add(key);if(!run.length)run.push(point(a));run.push(point(z));
      }flush();
    }
    const graph=new Map(),key=p=>`${round(p.column)}:${round(p.row)}`;
    for(const route of [...roads,...corridors].filter(r=>r.kind==='road'))for(let i=1;i<route.points.length;i++){
      const a=key(route.points[i-1]),z=key(route.points[i]);if(!graph.has(a))graph.set(a,new Set());if(!graph.has(z))graph.set(z,new Set());
      graph.get(a).add(z);graph.get(z).add(a);
    }
    const safeCurve=(a,z)=>{if(crossesWater(a,z))return false;const length=distance(a,z);if(length<1e-6)return true;
      if(M.abs(elevationAt(surface,a)-elevationAt(surface,z))/(length*1000)>.12)return false;
      const steps=M.max(4,M.ceil(length/.025));let previous=elevationAt(surface,a);
      for(let k=1;k<=steps;k++){const p=interpolate(a,z,k/steps),height=elevationAt(surface,p);
        if(!isDry(surface,p)||M.abs(height-previous)/(length/steps*1000)>.12||channels.some(ch=>{const near=distanceToLine(p,ch.points,metric);return near.distance<near.widthMetres/2000+.010;}))return false;
        previous=height;
      }return true;};
    const curve=route=>{
      if(route.kind!=='road'||route.points.length<3)return route;
      // Remove grid stair steps only where a continuous terrain check permits
      // the shortcut. Every junction and bridgehead remains a shared vertex.
      const fixed=p=>graph.get(key(p))?.size!==2||bridges.some(bridge=>distanceToLine(p,bridge.points,metric).distance<.035);
      const points=[route.points[0]];
      for(let i=1;i<route.points.length;){
        let far=i;
        while(far+1<route.points.length&&!fixed(route.points[far])&&distance(route.points[i-1],route.points[far+1])<.38
          &&safeCurve(points.at(-1),route.points[far+1]))far++;
        points.push(route.points[far]);i=far+1;
      }
      const out=[points[0]];
      for(let i=1;i<points.length-1;i++){
        const a=points[i-1],p=points[i],z=points[i+1];
        if(graph.get(key(p))?.size!==2||bridges.some(bridge=>distanceToLine(p,bridge.points,metric).distance<.035)){out.push(p);continue;}
        const first=interpolate(p,a,.42),last=interpolate(p,z,.42),arc=[first];
        for(let k=1;k<=5;k++){const t=k/5,q=1-t;arc.push({column:q*q*first.column+2*q*t*p.column+t*t*last.column,row:q*q*first.row+2*q*t*p.row+t*t*last.row});}
        if(safeCurve(out.at(-1),first)&&arc.slice(1).every((q,k)=>safeCurve(arc[k],q)))out.push(...arc);else out.push(p);
      }out.push(points.at(-1));return {...route,points:out};
    };
    return {ruralRoads:roads.map(curve),corridors:corridors.map(curve)};
  }

  function landform(recipe,context) {
    const b=context.bounds,c=recipe.location,m=context.gridCellKilometres,r=M.max(.5,recipe.urban.radiusKm),
      z=(x,y)=>sample(context.elevationMetres,context.rows,context.columns,b,c.row+y/m.row,c.column+x/m.column),h=z(0,0),
      dx=z(r,0)-z(-r,0),dy=z(0,r)-z(0,-r),cx=z(r,0)+z(-r,0)-2*h,cy=z(0,r)+z(0,-r)-2*h;
    const grade=M.hypot(dx,dy)/(r*2000),steep=['steep','escarpment'].includes(recipe.terrain?.slopeClass)||grade>.065,
      rolling=recipe.terrain?.slopeClass==='rolling'||grade>.018;
    const mixed=(z(r,r)+z(-r,-r)-z(r,-r)-z(-r,r))/4,delta=M.hypot(cx-cy,2*mixed),
      high=(cx+cy+delta)/2,low=(cx+cy-delta)/2,acrossAxis=M.atan2(2*mixed,cx-cy)/2;
    const valley=high>M.max(10,M.hypot(dx,dy)*.12)&&low<high*.5;
    const ridge=low< -15;
    const coast=['port','island-port'].includes(recipe.siteType)&&steep;
    const kind=coast?'coastal-slope':valley?'valley':ridge?'ridge':steep?'hillside':rolling?'hills':'plain';
    const axisRadians=kind==='valley'||kind==='ridge'?acrossAxis+M.PI/2:kind==='plain'?(recipe.morphology?.orientationDegrees||0)*M.PI/180:M.atan2(dx,-dy);
    const crossScale={plain:.92,hills:.64,hillside:.30,valley:.22,ridge:.33,'coastal-slope':.26}[kind];
    return {kind,axisRadians,crossScale,alongScale:['valley','hillside','coastal-slope'].includes(kind)?1.32:kind==='ridge'?1.08:.95,
      label:{plain:'缓坡平原',hills:'丘陵与缓坡',hillside:'山坡台地',valley:'狭长河谷',ridge:'山脊与高台','coastal-slope':'沿海陡坡台地'}[kind]};
  }

  function crossings(recipe,surface,context,channels) {
    const result=[],metric=context.gridCellKilometres;
    for(const route of recipe.transport.corridors)if(route.kind==='road')for(let segment=1;segment<route.points.length;segment++){
      const a=route.points[segment-1],b=route.points[segment],dx=(b.column-a.column)*metric.column,dy=(b.row-a.row)*metric.row,length=M.hypot(dx,dy);
      if(length<.001)continue;
      for(const channel of channels)for(let index=1;index<channel.points.length;index++){
        const c=channel.points[index-1],d=channel.points[index],sx=(d.column-c.column)*metric.column,sy=(d.row-c.row)*metric.row,
          qx=(c.column-a.column)*metric.column,qy=(c.row-a.row)*metric.row,denominator=dx*sy-dy*sx;
        if(M.abs(denominator)<1e-9)continue;
        const t=(qx*sy-qy*sx)/denominator,u=(qx*dy-qy*dx)/denominator;
        if(t<0||t>1||u<0||u>1)continue;
        const point={row:a.row+(b.row-a.row)*t,column:a.column+(b.column-a.column)*t};
        if(result.some(bridge=>M.hypot((bridge.center.column-point.column)*metric.column,(bridge.center.row-point.row)*metric.row)<.08))continue;
        const sine=M.abs(denominator)/(length*M.hypot(sx,sy));
        const widthMetres=c.widthMetres+(d.widthMetres-c.widthMetres)*u;
        const half=M.max(.027,(widthMetres/1000*.65+.016)/M.max(.25,sine));
        const points=[-1,1].map(sign=>({row:point.row+dy/length*half*sign/metric.row,column:point.column+dx/length*half*sign/metric.column}));
        if(!points.every(p=>elevationAt(surface,p)>0))continue;
        const heights=points.map(p=>elevationAt(surface,p));
        if(M.abs(heights[1]-heights[0])/(half*2000)>.12)continue;
        result.push({id:'bridge-'+result.length,routeId:route.id,center:point,points,widthMetres:M.max(5,recipe.urban.streetWidthsMetres?.arterial||8),
          name:Array.from(recipe.name||'当地').slice(0,3).join('')+'溪桥'+(result.length?String(result.length+1):''),
          grade:M.abs(heights[1]-heights[0])/(half*2000),bankHeightsMetres:heights});
      }
    }
    return result;
  }

  function contourPaths(points, heights, land, metric) {
    const SIZE=M.round(M.sqrt(points.length));
    const dryHeights = heights.filter((_, index) => land[index]);
    if (!dryHeights.length) return {terrainBands: [], contours: []};
    const minimum = M.min(...dryHeights), maximum = M.max(...dryHeights);
    const span = M.max(1, maximum - minimum);
    const levels = Array.from({length: 7}, (_, index) => minimum + span * index / 6);
    const interval=[5,10,20,25,50,100,200,500].find(v=>span/v<=20)||1000;
    const contourLevels=Array.from({length:M.floor(maximum/interval)-M.ceil(minimum/interval)+1},(_,i)=>(M.ceil(minimum/interval)+i)*interval);
    const bandPaths = levels.map(() => []);
    const lines=contourLevels.map(()=>[]),lineSegments=contourLevels.map(()=>[]);
    const shadePaths=Array.from({length:16},()=>[]),waterPaths=[],dryPaths=[];
    const fmt = point => `${round(point.column)} ${round(point.row)}`;
    const clip = (polygon, level) => {
      const result = [];
      for (let index = 0; index < polygon.length; index++) {
        const a = polygon[index], b = polygon[(index + 1) % polygon.length];
        if (a.height >= level) result.push(a);
        if ((a.height >= level) !== (b.height >= level)) {
          const t = (level - a.height) / (b.height - a.height);
          result.push({row: a.row + (b.row - a.row) * t, column: a.column + (b.column - a.column) * t, height: level});
        }
      }
      return result;
    };
    for (let row = 0; row < SIZE - 1; row++) for (let column = 0; column < SIZE - 1; column++) {
      const a = row * SIZE + column, b = a + 1, c = a + SIZE, d = c + 1;
      if([a,b,c,d].every(i=>land[i])){
        const dx=(heights[b]+heights[d]-heights[a]-heights[c])/((points[b].column-points[a].column)*metric.column*2000),
          dy=(heights[c]+heights[d]-heights[a]-heights[b])/((points[c].row-points[a].row)*metric.row*2000);
        // Northwest light and a modest vertical exaggeration reveal slopes.
        const nx=-dx*5,ny=-dy*5,nz=1,length=M.hypot(nx,ny,nz),light=(nx*-.5+ny*-.5+nz*.707)/length;
        const shade=clamp(M.round((light-.2)/.8*15),0,15);
        shadePaths[shade].push(`M${[a,b,d,c].map(i=>fmt(points[i])).join('L')}Z`);
      }
      for (const indices of [[a,b,d],[a,d,c]]) {
        const triangle = indices.map(index => ({...points[index], height: heights[index]}));
        if(!indices.every(index=>land[index])){
          const wet=clip(triangle.map(p=>({...p,height:-p.height})),0);
          if(wet.length>=3)waterPaths.push(`M${wet.map(fmt).join('L')}Z`);
        }
        const dry=clip(triangle,0);if(dry.length<3)continue;
        const min=M.min(...triangle.map(p=>p.height)),max=M.max(...triangle.map(p=>p.height)),full=`M${dry.map(fmt).join('L')}Z`;
        dryPaths.push(full);
        for (let levelIndex = 0; levelIndex < levels.length; levelIndex++) {
          const level = levels[levelIndex];if(level<=min){bandPaths[levelIndex].push(full);continue;}if(level>max)continue;
          const polygon = clip(dry, level);
          if (polygon.length >= 3) bandPaths[levelIndex].push(`M${polygon.map(fmt).join('L')}Z`);
        }
        for(let levelIndex=0;levelIndex<contourLevels.length;levelIndex++){
          const level=contourLevels[levelIndex];
          if(level<=min||level>=max)continue;
          const intersections = [];
          for (let edge = 0; edge < 3; edge++) {
            const first = triangle[edge], last = triangle[(edge + 1) % 3];
            if ((first.height < level) === (last.height < level)) continue;
            const t = (level - first.height) / (last.height - first.height);
            intersections.push({row: first.row + (last.row - first.row) * t, column: first.column + (last.column - first.column) * t});
          }
          if (intersections.length === 2) {lines[levelIndex].push(`M${fmt(intersections[0])}L${fmt(intersections[1])}`);lineSegments[levelIndex].push(intersections);}
        }
      }
    }
    return {terrainBands: bandPaths.map((paths, index) => ({d: paths.join(''), fill: colours[index], elevation: M.round(levels[index]), opacity: .72})).filter(band => band.d),
      contours: lines.map((paths,index)=>({d:paths.join(''),elevation:contourLevels[index],major:index%4===0,
        anchor:lineSegments[index][M.floor(lineSegments[index].length*.47)]?.[0]})).filter(line=>line.d),
      hillshade:shadePaths.map((paths,index)=>({d:paths.join(''),fill:index<9?'#5a6254':'#fff8e4',opacity:index<9?(9-index)*.055:(index-9)*.034})).filter(p=>p.d),
      elevationRange:{minimum:M.round(minimum),maximum:M.round(maximum),interval},
      waterArea:waterPaths.join(''),dryArea:dryPaths.join('')};
  }

  function fields(recipe, surface, slopes, metric, channels, seed) {
    const SIZE=surface.rows;
    const roads=recipe.transport.corridors.filter(r=>r.kind==='road'&&r.points.length>1),c=recipe.location,r=recipe.urban.radiusKm,b=surface.bounds;
    const local=p=>({x:(p.column-c.column)*metric.column,y:(p.row-c.row)*metric.row});
    const world=p=>({column:c.column+p.x/metric.column,row:c.row+p.y/metric.row});
    const box={west:(b.west-c.column)*metric.column,east:(b.east-c.column)*metric.column,
      north:(b.north-c.row)*metric.row,south:(b.south-c.row)*metric.row};
    const form=recipe.localSite?.form||landform(recipe,{bounds:b,gridCellKilometres:metric,rows:surface.rows,columns:surface.columns,elevationMetres:surface.elevationMetres});
    const check=p=>{
      if(p.column<b.west||p.column>b.east||p.row<b.north||p.row>b.south)return false;
      const z=elevationAt(surface,p),s=sample(slopes,SIZE,SIZE,b,p.row,p.column),q=local(p),
        along=q.x*M.cos(form.axisRadians)+q.y*M.sin(form.axisRadians),across=-q.x*M.sin(form.axisRadians)+q.y*M.cos(form.axisRadians);
      return z>0&&s<.16&&M.hypot(along/(form.alongScale*r),across/(form.crossScale*r))>1.12
        &&!channels.some(ch=>{const near=distanceToLine(p,ch.points,metric);return near.distance<near.widthMetres/2000+.035;})
        &&!roads.some(road=>distanceToLine(p,road.points,metric).distance<.018);
    };
    const clip=(polygon,nx,ny,d)=>{const out=[];for(let i=0;i<polygon.length;i++){
      const a=polygon[i],z=polygon[(i+1)%polygon.length],da=d-a.x*nx-a.y*ny,dz=d-z.x*nx-z.y*ny;
      if(da>=0)out.push(a);if((da>=0)!==(dz>=0)){const t=da/(da-dz);out.push({x:a.x+(z.x-a.x)*t,y:a.y+(z.y-a.y)*t});}}return out;};
    const area=p=>M.abs(p.reduce((s,a,i)=>{const z=p[(i+1)%p.length];return s+a.x*z.y-z.x*a.y;},0))/2;
    const centre=p=>({x:p.reduce((s,a)=>s+a.x,0)/p.length,y:p.reduce((s,a)=>s+a.y,0)/p.length});
    // One landscape-wide partition: holdings share boundaries, with no repeated
    // hexagon or nine-cell stamp. Fertility and access leave woodland and pasture.
    const sites=[],limit=recipe.population?.estimate<1800?52:115;
    for(let trial=0;trial<900&&sites.length<limit;trial++){
      const q={x:box.west+(box.east-box.west)*random(seed,70000+trial*3),y:box.north+(box.south-box.north)*random(seed,70001+trial*3)},p=world(q);
      const spacing=clamp(r*.22,.17,.55)*(0.65+random(seed,70002+trial*3));
      if(sites.some(a=>M.hypot(a.x-q.x,a.y-q.y)<spacing))continue;
      sites.push(q);
    }
    const results=[];
    for(let holding=0;holding<sites.length;holding++){
      const a=sites[holding],p=world(a);
      if(!check(p))continue;
      const roadDistance=M.min(...roads.map(road=>distanceToLine(p,road.points,metric).distance));
      if(roadDistance>M.max(.6,r*.85))continue;
      const fertility=M.sin(a.x/(r*.65+.2)+random(seed,811)*6)*M.cos(a.y/(r*.47+.3)-random(seed,812)*6);
      let polygon=[{x:box.west,y:box.north},{x:box.east,y:box.north},{x:box.east,y:box.south},{x:box.west,y:box.south}];
      for(let j=0;j<sites.length&&polygon.length>=3;j++)if(j!==holding){const z=sites[j];
        polygon=clip(polygon,z.x-a.x,z.y-a.y,(z.x*z.x+z.y*z.y-a.x*a.x-a.y*a.y)/2);}
      if(polygon.length<3)continue;
      const slope=sample(slopes,SIZE,SIZE,b,p.row,p.column),delta=.03,
        gx=sample(surface.elevationMetres,SIZE,SIZE,b,p.row,p.column+delta/metric.column)-sample(surface.elevationMetres,SIZE,SIZE,b,p.row,p.column-delta/metric.column),
        gy=sample(surface.elevationMetres,SIZE,SIZE,b,p.row+delta/metric.row,p.column)-sample(surface.elevationMetres,SIZE,SIZE,b,p.row-delta/metric.row,p.column);
      const target=clamp(r*r*.028,.050,.16)*(0.70+random(seed,holding+75000)*1.0),parcels=[];
      const divide=(poly,depth,salt)=>{
        if(depth>9||area(poly)<target){parcels.push(poly);return;}
        const xs=poly.map(p=>p.x),ys=poly.map(p=>p.y),angle=
          (M.max(...xs)-M.min(...xs)>M.max(...ys)-M.min(...ys)?0:M.PI/2)+(random(seed,salt)-.5)*.7,
          nx=M.cos(angle),ny=M.sin(angle),v=poly.map(p=>p.x*nx+p.y*ny),d=M.min(...v)+(M.max(...v)-M.min(...v))*(.40+random(seed,salt+1)*.20),
          first=clip(poly,nx,ny,d),second=clip(poly,-nx,-ny,-d);
        if(first.length<3||second.length<3){parcels.push(poly);return;}divide(first,depth+1,salt*2+1);divide(second,depth+1,salt*2+2);
      };divide(polygon,0,76000+holding*83);
      const quota=M.max(8,M.floor(1100/sites.length));
      const selected=parcels.map((poly,i)=>({poly,score:random(seed,holding*2048+i+91000)})).sort((a,b)=>a.score-b.score).slice(0,quota).map(p=>p.poly);
      for(let index=0;index<selected.length;index++){
        let poly=selected[index];const mid=centre(poly),use=M.sin(mid.x/(r*.65+.2)+random(seed,811)*6)*M.cos(mid.y/(r*.47+.3)-random(seed,812)*6)
          +M.sin(mid.x/(r*.19+.16)-mid.y/(r*.27+.18))*.32;
        if(use<-.32||random(seed,holding*71+index+74000)<.11)continue;
        const perimeter=poly.reduce((s,a,i)=>s+M.hypot(a.x-poly[(i+1)%poly.length].x,a.y-poly[(i+1)%poly.length].y),0);
        if(area(poly)<.007||area(poly)/(perimeter*perimeter)<.035)continue;
        const midWorld=world(mid),localSlope=sample(slopes,SIZE,SIZE,b,midWorld.row,midWorld.column),
          dx=sample(surface.elevationMetres,SIZE,SIZE,b,midWorld.row,midWorld.column+delta/metric.column)-sample(surface.elevationMetres,SIZE,SIZE,b,midWorld.row,midWorld.column-delta/metric.column),
          dy=sample(surface.elevationMetres,SIZE,SIZE,b,midWorld.row+delta/metric.row,midWorld.column)-sample(surface.elevationMetres,SIZE,SIZE,b,midWorld.row-delta/metric.row,midWorld.column);
        const terrace=localSlope>.028;
        const theta=terrace?M.atan2(dx,-dy):random(seed,holding+index+78000)*M.PI,
          ux=M.cos(theta),uy=M.sin(theta),vx=-uy,vy=ux;
        // Cross-sections follow actual sampled heights, giving curved contour
        // terraces rather than straight strips pasted over rising terrain.
        const contour=level=>{
          const projected=poly.map(q=>({x:q.x*ux+q.y*uy,y:q.x*vx+q.y*vy})),lo=M.min(...projected.map(q=>q.x)),hi=M.max(...projected.map(q=>q.x)),out=[];
          const count=M.max(8,M.ceil((hi-lo)/.025));
          for(let k=1;k<count;k++){
            const x=lo+(hi-lo)*k/count,cuts=[];
            for(let j=0;j<projected.length;j++){const a=projected[j],z=projected[(j+1)%projected.length];if((a.x>x)!==(z.x>x))cuts.push(a.y+(z.y-a.y)*(x-a.x)/(z.x-a.x));}
            if(cuts.length!==2)continue;cuts.sort((a,b)=>a-b);let low=cuts[0],high=cuts[1];
            const at=y=>({x:x*ux+y*vx,y:x*uy+y*vy}),height=y=>elevationAt(surface,world(at(y))),za=height(low),zb=height(high);
            if(level<M.min(za,zb)||level>M.max(za,zb))continue;
            for(let step=0;step<17;step++){const y=(low+high)/2;if((height(y)<level)===(za<zb))low=y;else high=y;}
            out.push(at((low+high)/2));
          }return out;
        };
        let rowLevels=[];
        if(terrace){
          const heights=poly.map(q=>elevationAt(surface,world(q))),lo=M.min(...heights),hi=M.max(...heights),middle=elevationAt(surface,midWorld),
            rise=clamp(localSlope*70,1.8,5),bottom=M.max(lo,middle-rise*.5),top=M.min(hi,middle+rise*.5),
            lower=contour(bottom),upper=contour(top);
          if(lower.length<4||upper.length<4)continue;
          const strip=[...lower,...upper.reverse()];
          if(area(strip)<.004)continue;
          poly=strip;rowLevels=[.25,.5,.75].map(t=>bottom+(top-bottom)*t);
        }
        const border=poly.map(q=>({x:mid.x+(q.x-mid.x)*.985,y:mid.y+(q.y-mid.y)*.985})),outline=[];
        for(let edge=0;edge<border.length;edge++){
          const a=border[edge],z=border[(edge+1)%border.length];outline.push(a);
          for(const fraction of terrace?[]:[.33,.67]){const p={x:a.x+(z.x-a.x)*fraction,y:a.y+(z.y-a.y)*fraction},amount=.001+random(seed,holding*201+index*23+edge)*.003,
            dx=mid.x-p.x,dy=mid.y-p.y,len=M.hypot(dx,dy)||1;outline.push({x:p.x+dx/len*amount,y:p.y+dy/len*amount});}
        }
        const points=outline.map(world);
        if(!points.every(check)||!check(world(mid)))continue;
        let safe=true;for(let i=0;i<points.length&&safe;i++){const a=points[i],z=points[(i+1)%points.length],length=M.hypot((a.column-z.column)*metric.column,(a.row-z.row)*metric.row);
          for(let t=.12;t<length;t+=.12)if(!check(interpolate(a,z,t/length))){safe=false;break;}}
        if(!safe)continue;
        const intersects=(a,z,p,q)=>{const dx=z.column-a.column,dy=z.row-a.row,sx=q.column-p.column,sy=q.row-p.row,den=dx*sy-dy*sx;
          if(M.abs(den)<1e-12)return false;const t=((p.column-a.column)*sy-(p.row-a.row)*sx)/den,u=((p.column-a.column)*dy-(p.row-a.row)*dx)/den;
          return t>=0&&t<=1&&u>=0&&u<=1;};
        if(roads.some(road=>road.points.some(p=>{const q=local(p);return poly.length>=3&&poly.reduce((hit,a,i)=>{const z=poly[(i+1)%poly.length];return (a.y>q.y)!==(z.y>q.y)&&q.x<(z.x-a.x)*(q.y-a.y)/(z.y-a.y)+a.x?!hit:hit;},false);})
          ||road.points.slice(1).some((q,j)=>points.some((p,i)=>intersects(p,points[(i+1)%points.length],road.points[j],q)))))continue;
        const kind=use<-.08?'meadow':terrace?'terrace':random(seed,holding*37+index+77000)<.16?'orchard':'field';
        const projected=poly.map(q=>({x:q.x*ux+q.y*uy,y:q.x*vx+q.y*vy})),low=M.min(...projected.map(q=>q.y)),high=M.max(...projected.map(q=>q.y)),lines=[];
        if(terrace&&kind!=='meadow')for(const level of rowLevels){const line=contour(level);if(line.length>1)lines.push(line.map(world));}
        const count=terrace||kind==='meadow'?0:kind==='orchard'?3:4;
        for(let row=1;row<=count;row++){
          const y=low+(high-low)*row/(count+1),cuts=[];
          for(let i=0;i<projected.length;i++){const a=projected[i],z=projected[(i+1)%projected.length];if((a.y>y)!==(z.y>y))cuts.push(a.x+(z.x-a.x)*(y-a.y)/(z.y-a.y));}
          if(cuts.length===2)lines.push(cuts.sort((a,b)=>a-b).map(x=>world({x:x*ux+y*vx,y:x*uy+y*vy})));
        }
        let access=null,best=Infinity;for(const road of roads)for(let i=1;i<road.points.length;i++){
          const a=local(road.points[i-1]),z=local(road.points[i]),dx=z.x-a.x,dy=z.y-a.y,t=clamp(((mid.x-a.x)*dx+(mid.y-a.y)*dy)/(dx*dx+dy*dy||1),0,1),q={x:a.x+t*dx,y:a.y+t*dy},d=M.hypot(q.x-mid.x,q.y-mid.y);
          if(d<best){best=d;access=world(q);}}
        results.push({points,lines,center:world(mid),access:[world(mid),access],kind,slope:localSlope,areaKm2:area(poly),
          fill:kind==='meadow'?'#c8d5b4':kind==='orchard'?'#c0cba4':['#d5d1aa','#dce0b9','#e3ddba','#d1d9b0'][M.floor(random(seed,holding*71+index+79000)*4)]});
      }
    }return results;
  }

  function landscape(recipe,surface,slopes,coordinates,land,seed,farmland,channels) {
    const SIZE=surface.rows;
    const metric=recipe.urban.gridCellKilometres||{row:10,column:10},center=recipe.location,radius=recipe.urban.radiusKm;
    const trees=[],rocks=[],woods=[];
    const inside=(p,polygon)=>{let result=false;for(let i=0,j=polygon.length-1;i<polygon.length;j=i++){
      const a=polygon[i],b=polygon[j];if((a.row>p.row)!==(b.row>p.row)&&p.column<(b.column-a.column)*(p.row-a.row)/(b.row-a.row)+a.column)result=!result;}return result;};
    const roads=recipe.transport.corridors.filter(r=>r.kind==='road');
    const fieldBoxes=farmland.map(field=>({field,west:M.min(...field.points.map(p=>p.column)),east:M.max(...field.points.map(p=>p.column)),
      north:M.min(...field.points.map(p=>p.row)),south:M.max(...field.points.map(p=>p.row))}));
    const available=p=>nearest(surface.buildable,SIZE,SIZE,surface.bounds,p)===1
      &&M.hypot((p.column-center.column)*metric.column,(p.row-center.row)*metric.row)>radius*1.12
      &&!fieldBoxes.some(b=>p.column>=b.west&&p.column<=b.east&&p.row>=b.north&&p.row<=b.south&&inside(p,b.field.points))
      &&!roads.some(r=>distanceToLine(p,r.points,metric).distance<.07);
    const free=coordinates.map((p,i)=>land[i]&&available(p));
    const noise=(u,v)=>M.sin(u*19+random(seed,90)*6)*M.cos(v*15+random(seed,91)*6)+M.sin(u*31-v*24)*.35;
    for(let row=1;row<SIZE-2;row++)for(let column=1;column<SIZE-2;column++){
      const index=row*SIZE+column,p=coordinates[index],u=column/(SIZE-1),v=row/(SIZE-1);
      if(!free[index])continue;
      const riverside=channels.some(channel=>distanceToLine(p,channel.points,metric).distance<.18);
      const value=noise(u,v)+(slopes[index]>.04?.25:0);
      if(value>.30||riverside){
        if([index,index+1,index+SIZE,index+SIZE+1].every(i=>free[i])){
          for(const triangle of [[index,index+1,index+SIZE+1],[index,index+SIZE+1,index+SIZE]]){
            const vertices=triangle.map(i=>({...coordinates[i],value:noise((i%SIZE)/(SIZE-1),M.floor(i/SIZE)/(SIZE-1))})),polygon=[];
            for(let edge=0;edge<3;edge++){
              const a=vertices[edge],b=vertices[(edge+1)%3];
              if(a.value>.5)polygon.push(a);
              if((a.value>.5)!==(b.value>.5)){const t=(.5-a.value)/(b.value-a.value);
                polygon.push({row:a.row+(b.row-a.row)*t,column:a.column+(b.column-a.column)*t});}
            }
            if(polygon.length>=3)woods.push(polygon);
          }
        }
        const count=value>.5?4:2;
        for(let n=0;n<count;n++){
          const x=column+(random(seed,index*9+n)-.5)*.85,y=row+(random(seed,index*9+n+4)-.5)*.85;
          const point={row:surface.bounds.north+y/(SIZE-1)*(surface.bounds.south-surface.bounds.north),column:surface.bounds.west+x/(SIZE-1)*(surface.bounds.east-surface.bounds.west)};
          if(available(point)&&trees.length<5000)trees.push({point,radiusMetres:9+random(seed,index+n)*11});
        }
      }else if(slopes[index]>.035&&random(seed,index)<.18)rocks.push({row:p.row,column:p.column});
    }
    return {woodland:woods,trees,rocks};
  }

  function geographicNames(recipe,context,site,metric){
    const prefix=Array.from(recipe.name||'当地').slice(0,3).join(''),result=[],c=recipe.location,f=site.surface,b=f.bounds;
    const SIZE=f.rows;
    const roots=context.nameRoots||[],used=new Set();
    const name=(suffix,salt)=>{
      if(!roots.length)return prefix+suffix;
      const start=M.floor(random(recipe.seed||0,salt)*roots.length);
      for(let i=0;i<roots.length;i++){const candidate=roots[(start+i)%roots.length]+suffix;if(!used.has(candidate)){used.add(candidate);return candidate;}}
      throw new RangeError('City landscape name inventory exhausted');
    };
    for(const feature of context.namedFeatures||[])result.push({id:'world-'+feature.id,name:feature.name,
      kind:feature.kind==='river'?'river':'terrain',anchor:feature.location,priority:4,detailLevel:0});
    for(const [i,ch] of site.channels.entries()){
      const anchor=ch.points[M.floor(ch.points.length*.53)],riverName=name(ch.kind==='river'?'河':'溪',5100+i);
      result.push({id:'supply-river-'+i,name:riverName,kind:'river',anchor,priority:2,detailLevel:0});ch.name=riverName;
    }
    for(const [i,lake] of site.lakes.entries()){
      lake.name=name('湖',5119+i);result.push({id:'local-lake-'+i,name:lake.name,kind:'river',anchor:lake.anchor,priority:1,detailLevel:0});
    }
    const candidates=[];
    for(let y=3;y<SIZE-3;y++)for(let x=3;x<SIZE-3;x++){
      const i=y*SIZE+x,z=f.elevationMetres[i];if(z<=site.elevationRange.minimum+30)continue;
      if([-SIZE-1,-SIZE,-SIZE+1,-1,1,SIZE-1,SIZE,SIZE+1].some(d=>f.elevationMetres[i+d]>z))continue;
      const p={row:b.north+y/(SIZE-1)*(b.south-b.north),column:b.west+x/(SIZE-1)*(b.east-b.west)};
      if(M.hypot((p.row-c.row)*metric.row,(p.column-c.column)*metric.column)<recipe.urban.radiusKm*.9)continue;
      candidates.push({p,z});
    }
    const peaks=[];for(const peak of candidates.sort((a,z)=>z.z-a.z)){
      if(peaks.some(p=>M.hypot((p.p.row-peak.p.row)*metric.row,(p.p.column-peak.p.column)*metric.column)<site.radiusKm*.35))continue;
      peaks.push(peak);if(peaks.length===3)break;
    }
    for(const [i,peak] of peaks.entries())result.push({id:'local-peak-'+i,name:name(i===0&&recipe.culture?.holyReligion?.tradition==='mountain-vow'?'圣山':i?'岭':'山',5200+i)+' · '+M.round(peak.z)+' m',
      kind:'terrain',anchor:peak.p,priority:3,detailLevel:0,spot:true});
    const cells=new Map();
    for(const polygon of site.woodland){const p={row:polygon.reduce((s,p)=>s+p.row,0)/polygon.length,column:polygon.reduce((s,p)=>s+p.column,0)/polygon.length},
      x=M.floor((p.column-b.west)/(b.east-b.west)*(SIZE-1)),y=M.floor((p.row-b.north)/(b.south-b.north)*(SIZE-1));cells.set(y*SIZE+x,p);}
    const woods=[];while(cells.size){const first=cells.keys().next().value,queue=[first],points=[];cells.delete(first);
      for(let k=0;k<queue.length;k++){const i=queue[k],x=i%SIZE,y=M.floor(i/SIZE);points.push({row:b.north+(y+.5)/(SIZE-1)*(b.south-b.north),column:b.west+(x+.5)/(SIZE-1)*(b.east-b.west)});
        for(const d of [-1,1,-SIZE,SIZE])if(cells.has(i+d)){cells.delete(i+d);queue.push(i+d);}}
      if(points.length>5)woods.push(points);
    }
    for(const [i,wood] of woods.sort((a,z)=>z.length-a.length).slice(0,4).entries()){
      const mid={row:wood.reduce((s,p)=>s+p.row,0)/wood.length,column:wood.reduce((s,p)=>s+p.column,0)/wood.length},
        anchor=wood.reduce((a,p)=>M.hypot(p.row-mid.row,p.column-mid.column)<M.hypot(a.row-mid.row,a.column-mid.column)?p:a,wood[0]);
      result.push({id:'local-wood-'+i,name:name('林',5300+i),kind:'forest',anchor,priority:5,detailLevel:0});
    }
    for(const [i,island] of (site.coastal.islands||[]).entries())result.push({id:'local-island-'+i,name:name('岛',5400+i),kind:'terrain',anchor:island.anchor,priority:2,detailLevel:0});
    for(const [i,line] of site.contours.entries())if(line.major&&line.anchor)result.push({id:'contour-'+i,name:line.elevation+' m',kind:'elevation',anchor:line.anchor,priority:8,detailLevel:0});
    return result;
  }

  window.WorldAtlasCitySite = {generate,elevationAt,isDry,distanceToChannel:(point,channel,metric)=>distanceToLine(point,channel.points,metric)};
})();
