/* One city, generated on selection in the existing place card. */
(() => {
  'use strict';
  const SVG_NS='http://www.w3.org/2000/svg';
  const svg=(name,attributes={})=>{
    const node=document.createElementNS(SVG_NS,name);
    for(const [key,value] of Object.entries(attributes))node.setAttribute(key,String(value));
    return node;
  };
  const path=points=>points.map((p,i)=>`${i?'L':'M'}${p.column.toFixed(7)} ${p.row.toFixed(7)}`).join(' ');

  function create({viewport,container}) {
    const cameraNode=document.getElementById('city-map-camera'),status=document.getElementById('city-map-status');
    const reliefKey=document.createElement('div');reliefKey.className='city-relief-key';reliefKey.setAttribute('aria-label','海拔与等高线');viewport.append(reliefKey);
    let selected=null,recipe=null,saved=null,controller=null,request=0,generatedCount=0,seed=null,loading=false;
    let scale=1,translateX=0,translateY=0,width=0,height=0,fitScale=1,extent='city';
    let minimumScale=1,maximumScale=1800;
    const notify=message=>{status.textContent=message;};
    function updateCamera() {
      if(recipe&&width&&height){
        const bounds=recipe.localSite.bounds,metric=recipe.urban.gridCellKilometres;
        minimumScale=Math.max(width/((bounds.east-bounds.west)*metric.column),height/((bounds.south-bounds.north)*metric.row))*1.015;
        maximumScale=Math.max(minimumScale,1800);
        scale=Math.max(minimumScale,Math.min(maximumScale,scale));
        translateX=Math.max(width-(bounds.east-recipe.location.column)*metric.column*scale,
          Math.min(-(bounds.west-recipe.location.column)*metric.column*scale,translateX));
        translateY=Math.max(height-(bounds.south-recipe.location.row)*metric.row*scale,
          Math.min(-(bounds.north-recipe.location.row)*metric.row*scale,translateY));
        document.getElementById('city-map-zoom-in').disabled=scale>=maximumScale*.999;
        document.getElementById('city-map-zoom-out').disabled=scale<=minimumScale*1.001;
      }
      cameraNode.setAttribute('transform',`translate(${translateX} ${translateY}) scale(${scale})`);
      if(!recipe)return;
      const metric=recipe.urban.gridCellKilometres;
      const diameter=recipe.urban.radiusKm*2*scale,occupied=[],names=new Set();
      const labels=Array.from(container.querySelectorAll('[data-city-screen-label]')).sort((a,b)=>
        (Number(a.dataset.priority)+(diameter>640&&a.dataset.featureKind==='square'?4:0))
        -(Number(b.dataset.priority)+(diameter>640&&b.dataset.featureKind==='square'?4:0)));
      for(const node of labels){
        const road=node.dataset.featureKind==='road',natural=['terrain','forest','river','elevation'].includes(node.dataset.featureKind),
          font=node.dataset.featureKind==='elevation'?9:road?10.5:11.5,angle=road?Number(node.dataset.angle):0;
        const row=Number(node.dataset.anchorRow),column=Number(node.dataset.anchorColumn);
        const x=translateX+(column-recipe.location.column)*metric.column*scale,y=translateY+(row-recipe.location.row)*metric.row*scale;
        node.setAttribute('font-size',font/(scale*metric.row));node.setAttribute('stroke-width',3/(scale*metric.row));
        node.setAttribute('transform',`translate(${column} ${row}) scale(${metric.row/metric.column} 1) rotate(${angle})`);
        node.setAttribute('font-weight',road?'400':'500');
        const textWidth=Array.from(node.textContent).reduce((sum,c)=>sum+(c.charCodeAt(0)>255?1:.58),0)*font+12;
        const radians=angle*Math.PI/180,w=Math.abs(Math.cos(radians))*textWidth+Math.abs(Math.sin(radians))*font+8,
          h=Math.abs(Math.sin(radians))*textWidth+Math.abs(Math.cos(radians))*font+8;
        const box={left:x-w/2,right:x+w/2,top:y-h/2,bottom:y+h/2};
        const visible=(natural||diameter>=[100,250,610][Number(node.dataset.detailLevel)])
          &&(!road||Number(node.dataset.lengthKm)*scale>textWidth*1.18)&&x>30&&x<width-30&&y>20&&y<height-20
          &&!names.has(node.textContent)&&!occupied.some(b=>box.left<b.right&&box.right>b.left&&box.top<b.bottom&&box.bottom>b.top);
        node.style.display=visible?'':'none';
        if(visible){occupied.push(box);names.add(node.textContent);}
      }
      container.classList.toggle('city-close-detail',diameter>520);
      reliefKey.textContent=`海拔 ${recipe.localSite.elevationRange.minimum}–${recipe.localSite.elevationRange.maximum} m · 等高距 ${recipe.localSite.elevationRange.interval} m`;
      const maximum=100/scale,power=10**Math.floor(Math.log10(maximum));
      const km=[1,2,5].map(v=>v*power).filter(v=>v<=maximum).pop()||power/2;
      document.getElementById('city-scale-label').textContent=km<1?`${Math.round(km*1000)} m`:`${Number(km.toFixed(2))} km`;
      document.getElementById('city-scale-rule').style.width=`${km*scale}px`;
    }
    function reset() {
      width=viewport.clientWidth;height=viewport.clientHeight;
      if(!recipe||!width||!height)return;
      const radius=extent==='region'?recipe.localSite.radiusKm*.94:recipe.urban.radiusKm*1.45;
      fitScale=extent==='region'?Math.max(width,height)/(recipe.localSite.radiusKm*2)*1.04:Math.min(width,height)/(radius*2)*.91;
      scale=fitScale;
      translateX=width/2;translateY=height/2;updateCamera();
    }
    function setExtent(value) {
      if(!['city','region'].includes(value))throw new TypeError('Invalid city extent');
      extent=value;
      for(const key of ['city','region'])document.getElementById('city-map-'+key).setAttribute('aria-pressed',String(key===extent));
      reset();
    }
    function zoomAt(factor,x,y) {
      if(!recipe)return;
      const box=viewport.getBoundingClientRect(),px=x-box.left,py=y-box.top;
      const next=Math.max(minimumScale,Math.min(maximumScale,scale*factor));
      translateX=px-(px-translateX)*next/scale;translateY=py-(py-translateY)*next/scale;scale=next;updateCamera();
    }
    function zoomBy(factor) {const box=viewport.getBoundingClientRect();zoomAt(factor,box.left+box.width/2,box.top+box.height/2);}
    function clear() {
      request++;controller?.abort();controller=null;selected=null;recipe=null;saved=null;loading=false;
      container.replaceChildren();notify('');
    }
    function drawSite(site) {
      const group=svg('g',{class:'city-local-site','data-water-source':site.waterSource.kind});
      const bands=svg('g',{class:'city-local-terrain'});
      for(const band of site.terrainBands)bands.append(svg('path',{d:band.d,fill:band.fill}));
      group.append(bands);
      const shade=svg('g',{class:'city-local-hillshade'});
      const definitions=svg('defs'),filter=svg('filter',{id:'city-terrain-soften',x:'-10%',y:'-10%',width:'120%',height:'120%'});
      const metric=recipe.urban.gridCellKilometres;
      const soften=site.radiusKm*2/(site.surface.rows-1)*.28;
      filter.append(svg('feGaussianBlur',{stdDeviation:`${soften/metric.column} ${soften/metric.row}`}));definitions.append(filter);group.append(definitions);
      shade.setAttribute('filter','url(#city-terrain-soften)');
      for(const patch of site.hillshade||[])shade.append(svg('path',{d:patch.d,fill:patch.fill,opacity:patch.opacity}));
      group.append(shade);
      if(site.waterArea)group.append(svg('path',{d:site.waterArea,fill:'#b5d5df',class:'city-coastal-water'}));
      const contours=svg('g',{class:'city-local-contours',fill:'none',stroke:'#85816b','stroke-opacity':.52});
      for(const line of site.contours)contours.append(svg('path',{d:line.d,'data-height-m':line.elevation,'stroke-width':line.major?.95:.5,'vector-effect':'non-scaling-stroke'}));
      group.append(contours);
      const water=svg('g',{class:'city-local-water'});
      const dryClip=svg('clipPath',{id:'city-river-dry-ground',clipPathUnits:'userSpaceOnUse'});
      dryClip.append(svg('path',{d:site.dryArea}));definitions.append(dryClip);
      water.setAttribute('clip-path','url(#city-river-dry-ground)');
      const woods=svg('g',{class:'city-local-woodland'});
      woods.append(svg('path',{d:(site.woodland||[]).map(points=>path(points)+'Z').join(' '),fill:'#b8caa5',opacity:.55}));
      const treePath=(site.trees||[]).map(tree=>{
        const rx=tree.radiusMetres/1000/metric.column,ry=tree.radiusMetres/1000/metric.row,p=tree.point;
        return `M${p.column-rx} ${p.row}a${rx} ${ry} 0 1 0 ${rx*2} 0a${rx} ${ry} 0 1 0 ${-rx*2} 0`;
      }).join(' ');
      woods.append(svg('path',{d:treePath,class:'city-tree-canopies',fill:'#8fac87',stroke:'#739575','stroke-width':.25,
        'vector-effect':'non-scaling-stroke',opacity:.74,'data-tree-count':(site.trees||[]).length}));
      const rockPath=(site.rocks||[]).map(p=>path([{column:p.column-.015/metric.column,row:p.row},
        {column:p.column,row:p.row-.01/metric.row},{column:p.column+.018/metric.column,row:p.row+.007/metric.row}])).join(' ');
      woods.append(svg('path',{d:rockPath,fill:'none',stroke:'#9b9b87','stroke-width':.6,'vector-effect':'non-scaling-stroke'}));
      group.append(woods);
      for(const channel of site.channels){
        const ribbon=factor=>{
          const banks=[[],[]];
          for(let i=0;i<channel.points.length;i++){
            const p=channel.points[i],a=channel.points[Math.max(0,i-1)],z=channel.points[Math.min(channel.points.length-1,i+1)],
              dx=(z.column-a.column)*metric.column,dy=(z.row-a.row)*metric.row,len=Math.hypot(dx,dy)||1,half=p.widthMetres/2000*factor;
            for(const [bank,sign] of [[0,-1],[1,1]])banks[bank].push({column:p.column-dy/len*half*sign/metric.column,row:p.row+dx/len*half*sign/metric.row});
          }return path(banks[0].concat(banks[1].reverse()))+'Z';
        };
        water.append(svg('path',{d:ribbon(1.65),fill:'#a9c6a7',opacity:.58,class:'city-riverbank'}));
        water.append(svg('path',{d:ribbon(1),fill:'#8ebdcc',stroke:'#6b9aa8','stroke-width':.45,'vector-effect':'non-scaling-stroke',class:'city-supply-river',
          'data-min-width-m':Math.min(...channel.points.map(p=>p.widthMetres)),'data-max-width-m':Math.max(...channel.points.map(p=>p.widthMetres))}));
      }
      const source=site.waterSource;
      if(source.kind==='groundwater')water.append(svg('circle',{cx:source.location.column,cy:source.location.row,r:.028/metric.row,fill:'#9dbcc2',stroke:'#536d75','stroke-width':1,'vector-effect':'non-scaling-stroke',class:'city-water-well'}));
      group.append(water);
      const lakes=svg('g',{class:'city-local-lakes','data-lake-count':site.lakes.length});
      for(const lake of site.lakes)lakes.append(svg('path',{d:lake.patches.map(p=>path(p)+'Z').join(' '),fill:'#8fb8c1',
        'data-water-level-m':lake.waterLevelMetres,'data-area-km2':lake.areaKm2}));
      group.append(lakes);
      const roadWidth=recipe.urban.streetWidthsMetres.arterial/1000/((metric.row+metric.column)/2);
      const roads=svg('g',{class:'city-external-roads',fill:'none','stroke-linecap':'round','stroke-linejoin':'round'}),segments=[];
      const field=site.surface,bounds=field.bounds;
      const nearLine=(p,points)=>{let best=Infinity;for(let i=1;i<points.length;i++){
        const a=points[i-1],b=points[i],dx=(b.column-a.column)*metric.column,dy=(b.row-a.row)*metric.row,
          px=(p.column-a.column)*metric.column,py=(p.row-a.row)*metric.row,t=Math.max(0,Math.min(1,(px*dx+py*dy)/(dx*dx+dy*dy)));
        best=Math.min(best,Math.hypot(px-dx*t,py-dy*t));}return best;};
      const allowed=(point,corridor)=>{
        const u=(point.column-bounds.west)/(bounds.east-bounds.west),v=(point.row-bounds.north)/(bounds.south-bounds.north);
        if(u<0||u>1||v<0||v>1)return false;
        return WorldAtlasCitySite.isDry(field,point)&&!site.channels.some(channel=>{const near=window.WorldAtlasCitySite.distanceToChannel(point,channel,metric);return near.distance<near.widthMetres/2000+.008;})
          ||site.bridges.some(bridge=>nearLine(point,bridge.points)<.02);
      };
      for(const corridor of [...recipe.transport.corridors,...site.ruralRoads].filter(c=>c.kind==='road')){
        let run=[];
        const flush=()=>{if(run.length>1)segments.push(path(run));run=[];};
        for(let i=1;i<corridor.points.length;i++){
          const a=corridor.points[i-1],b=corridor.points[i];
          const count=Math.max(1,Math.ceil(Math.hypot((b.column-a.column)*metric.column,(b.row-a.row)*metric.row)/.06));
          for(let j=0;j<=count;j++){
            const t=j/count,point={row:a.row+(b.row-a.row)*t,column:a.column+(b.column-a.column)*t};
            if(allowed(point,corridor))run.push(point);else flush();
          }
        }flush();
      }
      roads.append(svg('path',{d:segments.join(' '),stroke:'#c7bca7','stroke-width':roadWidth*1.25}));
      roads.append(svg('path',{d:segments.join(' '),stroke:'#fffdf7','stroke-width':roadWidth}));
      group.append(roads);
      const bridges=svg('g',{class:'city-local-bridges','data-bridge-count':site.bridges.length});
      for(const bridge of site.bridges){
        const [a,b]=bridge.points,dx=(b.column-a.column)*metric.column,dy=(b.row-a.row)*metric.row,length=Math.hypot(dx,dy);
        bridges.append(svg('path',{d:path(bridge.points),fill:'none',stroke:'#8b8d7f','stroke-width':roadWidth*1.65}));
        bridges.append(svg('path',{d:path(bridge.points),fill:'none',stroke:'#fff9ea','stroke-width':roadWidth*1.15}));
        for(const sign of [-1,1]){const offset=bridge.widthMetres/2000*sign;
          bridges.append(svg('path',{d:path(bridge.points.map(p=>({row:p.row+dx/length*offset/metric.row,column:p.column-dy/length*offset/metric.column}))),
            fill:'none',stroke:'#727e77','stroke-width':.65,'vector-effect':'non-scaling-stroke'}));}
      }
      group.append(bridges);return group;
    }
    function drawHarbor(harbor){
      if(!harbor)return svg('g');
      const m=recipe.urban.gridCellKilometres,n=harbor.outward,t={x:-n.y,y:n.x},s=harbor.shore,
        factor=Math.max(.34,Math.min(1.8,Math.sqrt(recipe.population.estimate/22000))),
        at=(x,y)=>({column:s.column+(t.x*x+n.x*y)*factor/m.column,row:s.row+(t.y*x+n.y*y)*factor/m.row});
      const group=svg('g',{class:'city-harbor','data-harbor-kind':harbor.kind});
      const field=recipe.localSite.surface,b=field.bounds;
      const inside=p=>p.column>=b.west&&p.column<=b.east&&p.row>=b.north&&p.row<=b.south;
      const inChannel=p=>recipe.localSite.channels.some(channel=>{const q=WorldAtlasCitySite.distanceToChannel(p,channel,m);return q.distance<q.widthMetres/2000;});
      const dry=p=>inside(p)&&WorldAtlasCitySite.isDry(field,p)&&!inChannel(p),wet=p=>inside(p)&&(!WorldAtlasCitySite.isDry(field,p)||inChannel(p));
      const random=salt=>{let x=(recipe.seed^Math.imul(salt,0x9e3779b9))>>>0;x=Math.imul(x^x>>>16,0x21f0aaad);return ((x^x>>>15)>>>0)/4294967296;};
      const small=recipe.population.estimate<1800,capacity=Math.max(1,Math.min(7,Math.round(Math.sqrt(recipe.population.estimate/1600)))),
        span=small?.10:.22+capacity*.016,berths=[];
      const bank=x=>{let low=-.16,high=harbor.kind==='river'?harbor.channelWidthMetres/2000/factor:.32;
        if(!dry(at(x,low))||!wet(at(x,high)))return null;
        for(let i=0;i<19;i++){const middle=(low+high)/2;if(dry(at(x,middle)))low=middle;else high=middle;}return (low+high)/2;};
      const fronts=[];
      for(let i=0;i<=16;i++){const x=(i/16*2-1)*span,y=bank(x);if(y!==null)fronts.push({x,y});}
      // Quay segments follow the actual bank instead of bridging across bays.
      for(let i=1;i<fronts.length;i++){
        const a=fronts[i-1],z=fronts[i];if(z.x-a.x>span*.18)continue;
        const poly=[at(a.x,a.y-.014),at(z.x,z.y-.014),at(z.x,z.y-.002),at(a.x,a.y-.002)];
        if(!poly.every(dry))continue;
        group.append(svg('path',{d:path(poly)+'Z',fill:small?'#a8916b':'#cbc5ac',stroke:'#727e74','stroke-width':.45,'vector-effect':'non-scaling-stroke',class:'city-harbor-quay'}));
      }
      const palette=WorldAtlasCityCharacter.profile(recipe).roof;
      for(let i=0;i<capacity;i++){
        const x=capacity===1?0:(i/(capacity-1)*2-1)*span*.80,y=bank(x);if(y===null)continue;
        const length=harbor.kind==='river'?Math.min(.11,harbor.channelWidthMetres/1000/factor*.32):.08+random(i+303)*.15,
          half=small?.004:.006+random(i+309)*.003,
          pier=[at(x-half,y-.006),at(x+half,y-.006),at(x+half,y+length),at(x-half,y+length)];
        if(!Array.from({length:9},(_,k)=>at(x,y+.01+(length-.01)*k/8)).every(wet))continue;
        berths.push({x,y,length});
        group.append(svg('path',{d:path(pier)+'Z',fill:'#a78d64',stroke:'#58655e','stroke-width':.6,'vector-effect':'non-scaling-stroke',class:'city-harbor-pier'}));
        for(let k=1;k<8;k++)group.append(svg('path',{d:path([at(x-half,y+length*k/8),at(x+half,y+length*k/8)]),
          stroke:'#ddc8a0','stroke-width':.45,'vector-effect':'non-scaling-stroke',class:'city-pier-planks'}));
        const shipWidth=Math.min(.015,length*.17),shipLength=Math.min(.050,length*.45),shipX=x+half+shipWidth*1.4,shipY=y+length*.55;
        const ship=[at(shipX,shipY-shipLength*.5),at(shipX+shipWidth,shipY-shipLength*.22),at(shipX+shipWidth*.8,shipY+shipLength*.40),
          at(shipX,shipY+shipLength*.5),at(shipX-shipWidth*.8,shipY+shipLength*.40),at(shipX-shipWidth,shipY-shipLength*.22)];
        if(ship.every(wet)){
          group.append(svg('path',{d:path(ship)+'Z',fill:'#e6d4ab',stroke:'#435d65','stroke-width':.7,'vector-effect':'non-scaling-stroke',class:'city-harbor-vessel'}));
          group.append(svg('path',{d:path([at(shipX,shipY-shipLength*.35),at(shipX,shipY+shipLength*.34)]),stroke:'#645e4b','stroke-width':.6,'vector-effect':'non-scaling-stroke'}));
          group.append(svg('path',{d:path([at(shipX,shipY-shipLength*.25),at(shipX+shipWidth*1.7,shipY+.003),at(shipX,shipY+shipLength*.25)])+'Z',
            fill:'#fff2d4',stroke:'#827d65','stroke-width':.35,'vector-effect':'non-scaling-stroke',class:'city-ship-sails'}));
        }
        const h=.025+random(i+330)*.01,w=.023+random(i+329)*.013,
          house=[at(x-w,y-.039),at(x+w,y-.039),at(x+w,y-.039-h),at(x-w,y-.039-h)];
        const heights=house.map(p=>WorldAtlasCitySite.elevationAt(field,p));
        if(house.every(dry)&&Math.max(...heights)-Math.min(...heights)<4){
          group.append(svg('path',{d:path(house)+'Z',fill:palette[i%palette.length],stroke:'#4e5a4b','stroke-width':.6,'vector-effect':'non-scaling-stroke',class:'city-harbor-warehouse'}));
          group.append(svg('path',{d:path([at(x,y-.041),at(x,y-.037-h)]),stroke:'#e4d8b8','stroke-width':.55,'vector-effect':'non-scaling-stroke'}));
          group.append(svg('path',{d:path([at(x,y-.039),at(x,y-.014)]),stroke:'#e8dfc8','stroke-width':.0035/factor,'stroke-linecap':'round'}));
        }
      }
      if(harbor.kind==='sea'&&!small&&harbor.shelter<.78&&berths.length>1){
        const end=berths.at(-1),wall=[at(end.x+.025,end.y+.008),at(end.x+.04,end.y+.11),at(end.x-.04,end.y+.19)];
        const inWater=wall.slice(1).every((z,i)=>Array.from({length:17},(_,j)=>({column:wall[i].column+(z.column-wall[i].column)*j/16,
          row:wall[i].row+(z.row-wall[i].row)*j/16})).every(wet));
        if(inWater){
          group.append(svg('path',{d:path(wall),fill:'none',stroke:'#62746d','stroke-width':.012/((m.row+m.column)/2),'stroke-linecap':'round','stroke-linejoin':'round',class:'city-harbor-breakwater'}));
          group.append(svg('path',{d:path(wall),fill:'none',stroke:'#d7cfae','stroke-width':.007/((m.row+m.column)/2),'stroke-linecap':'round','stroke-linejoin':'round'}));
          const tip=wall.at(-1);group.append(svg('circle',{cx:tip.column,cy:tip.row,r:.008/m.row,fill:'#ede0ba',stroke:'#56695f','stroke-width':.7,'vector-effect':'non-scaling-stroke',class:'city-harbor-beacon'}));
        }
      }
      group.dataset.berthCount=String(berths.length);
      const access=harbor.access;
      group.append(svg('path',{d:path(access),fill:'none',stroke:'#bdb8a3','stroke-width':recipe.urban.streetWidthsMetres.arterial/1000/((m.row+m.column)/2)*1.3,'stroke-linejoin':'round'}));
      group.append(svg('path',{d:path(access),fill:'none',stroke:'#fff6e2','stroke-width':recipe.urban.streetWidthsMetres.arterial/1000/((m.row+m.column)/2),'stroke-linejoin':'round'}));
      if(harbor.sailing)group.append(svg('path',{d:path(harbor.sailing),fill:'none',stroke:'#4c8ca8','stroke-width':1.3,'stroke-dasharray':'4 4','vector-effect':'non-scaling-stroke'}));
      const label=svg('text',{'data-anchor-column':s.column,'data-anchor-row':s.row,'data-city-screen-label':harbor.name,
        'data-feature-kind':'harbor','data-priority':0,'data-detail-level':0,'data-angle':0,'paint-order':'stroke fill',
        'text-anchor':'middle',fill:'#2f6984',stroke:'#f5faf6'});label.textContent='⚓ '+harbor.name;group.append(label);
      return group;
    }
    async function open(place) {
      if(place.kind!=='city'){clear();return null;}
      const same=saved?.recipe.id===place.sourceId,cached=same?saved:null,reuseCamera=same&&recipe&&width&&height;
      request++;controller?.abort();controller=new AbortController();const current=request,signal=controller.signal;
      selected=place;recipe=null;container.replaceChildren();loading=true;notify(`正在生成${place.name}…`);
      try {
        if(!cached){
          const response=await fetch(`city-maps/${encodeURIComponent(place.sourceId)}.json`,{signal});
          if(!response.ok)throw new Error('城市地理数据未能载入');
          const payload=await response.json();if(current!==request)return null;
          if(payload.schema!=='world-atlas-city-map-v1'||payload.recipe?.id!==place.sourceId||payload.gridDigest!==document.querySelector('meta[name="grid-digest"]').content)throw new Error('城市数据与当前世界不一致');
          saved=payload;
        }
        await new Promise(resolve=>requestAnimationFrame(resolve));if(current!==request)return null;
        const value=crypto.getRandomValues(new Uint32Array(1))[0];seed=value===seed?(value+1)>>>0:value;
        recipe=WorldAtlasCitySite.generate({...saved.recipe,seed},saved.siteContext,{seed});
        const city=WorldAtlasCities.render(recipe,{grid:saved.grid,clipPrefix:'city-panel'});
        city.dataset.seed=String(seed);city.dataset.population=String(recipe.population.estimate);
        city.dataset.nativeRow=String(recipe.location.row);city.dataset.nativeColumn=String(recipe.location.column);
        const metric=recipe.urban.gridCellKilometres;
        container.setAttribute('transform',`scale(${metric.column} ${metric.row}) translate(${-recipe.location.column} ${-recipe.location.row})`);
        container.replaceChildren(drawSite(recipe.localSite),city,drawHarbor(recipe.harbor));generatedCount++;loading=false;notify('');
        if(!reuseCamera)setExtent('city');else updateCamera();return recipe;
      }catch(error){
        if(signal.aborted||current!==request)return null;
        loading=false;recipe=null;notify(error.message+'，点击“重新生成”重试。');return null;
      }
    }
    const interaction=WorldAtlasInteraction.attach({viewport,
      getCamera:()=>({scale,translateX,translateY,minimumScale,maximumScale}),
      setCamera:next=>{scale=next.scale;translateX=next.translateX;translateY=next.translateY;updateCamera();},
      zoomAt,zoomBy,reset,onTap:()=>{},onCommit:()=>updateCamera()});
    const listeners=[];
    for(const [id,action] of [['zoom-in',()=>zoomBy(1.6)],['zoom-out',()=>zoomBy(1/1.6)],['reset',reset],['city',()=>setExtent('city')],['region',()=>setExtent('region')]]){
      const node=document.getElementById('city-map-'+id);node.addEventListener('click',action);listeners.push(()=>node.removeEventListener('click',action));
    }
    const observer=new ResizeObserver(()=>{
      if(!recipe)return;
      const nextWidth=viewport.clientWidth,nextHeight=viewport.clientHeight;
      if(!nextWidth||!nextHeight)return;
      if(!width||!height)reset();else{translateX+=(nextWidth-width)/2;translateY+=(nextHeight-height)/2;width=nextWidth;height=nextHeight;updateCamera();}
    });observer.observe(viewport);
    return {open,clear,zoomBy,reset,setExtent,get recipe(){return recipe;},stats:()=>({selected:selected?.id||null,recipeId:recipe?.id||null,
      seed,generatedCount,loading,active:recipe?1:0,anchor:recipe?{...recipe.location}:null,population:recipe?.population||null,
      buildings:Array.from(container.querySelectorAll('[data-building-count]'),node=>Number(node.dataset.buildingCount)).reduce((a,b)=>a+b,0),
      innerCity:container.querySelectorAll('.city-block-old-town,.city-block-civic').length,
      innerWalls:Array.from(container.querySelectorAll('.city-inner-walls')).filter(node=>node.getAttribute('d')).length,
      gates:Number(container.querySelector('[data-gate-count]')?.dataset.gateCount||0),
      bridges:recipe?.localSite.bridges.length||0,
      barbicans:Number(container.querySelector('[data-barbican-count]')?.dataset.barbicanCount||0),
      moats:Number(container.querySelector('[data-moat-count]')?.dataset.moatCount||0),riverMode:recipe?.localSite.riverMode,
      harbor:recipe?.harbor||null,
      character:recipe?WorldAtlasCityCharacter.profile(recipe).label:null,
      labels:container.querySelectorAll('[data-city-screen-label]').length,
      visibleLabels:Array.from(container.querySelectorAll('[data-city-screen-label]')).filter(node=>node.style.display!=='none').map(node=>node.textContent),
      outerCity:container.querySelectorAll('.city-blocks .city-block').length,
      farmland:Array.from(container.querySelectorAll('[data-farmland-count]'),node=>Number(node.dataset.farmlandCount)).reduce((a,b)=>a+b,0),
      waterSource:recipe?.localSite.waterSource||null,camera:{scale,translateX,translateY,width,height,extent,minimumScale,maximumScale}}),
      destroy(){clear();interaction.destroy();observer.disconnect();listeners.forEach(remove=>remove());}};
  }
  window.WorldAtlasCityMap={create};
})();
