/* Navigation panel, network worker and one transient route overlay. */
(() => {
  'use strict';
  const SVG='http://www.w3.org/2000/svg';
  function create({overlay,onFit,onPicking}){
    const $=id=>document.getElementById(id),panel=$('navigation-panel');
    let worker=null,ready=null,serial=0,revision=0,picking=null,result=null,cities=[],profile=null;
    const pending=new Map(),selected={start:null,end:null};
    const group=document.createElementNS(SVG,'g');group.dataset.atlasNavigation='';group.setAttribute('pointer-events','none');overlay.appendChild(group);
    const status=message=>{$('navigation-status').textContent=message;};
    function ask(action,request){return new Promise((resolve,reject)=>{const id=++serial;pending.set(id,{resolve,reject});worker.postMessage({id,action,request});});}
    async function load(){
      if(!ready){worker=new Worker('atlas-navigation-worker.js');worker.onmessage=({data})=>{
        const promise=pending.get(data.id);if(!promise)return;pending.delete(data.id);data.error?promise.reject(Error(data.error)):promise.resolve(data.result);
      };worker.onerror=()=>{for(const promise of pending.values())promise.reject(Error('导航服务无法启动'));pending.clear();};
        ready=ask('load').then(data=>{cities=data.cities;profile=data.profile;
          const list=$('navigation-cities');list.replaceChildren(...cities.map(city=>{const o=document.createElement('option');o.value=city.name;o.label=city.id;return o;}));
          $('navigation-mode').replaceChildren(...profile.modes.map(mode=>{const o=document.createElement('option');o.value=mode.id;o.textContent=mode.label+(mode.rarity==='rare'?' · 稀有':'');return o;}));
          $('navigation-rules').textContent=profile.assumptions;return data;});}
      return ready;
    }
    function pick(which){picking=which;onPicking(true);status(`请在地图上选择${which==='start'?'起点':'终点'}；飞行使用所选地点，地面和船只使用实际通行网络。`);}
    function finishPicking(){picking=null;onPicking(false);}
    function endpoint(which){
      const value=$(`navigation-${which}`).value.trim();
      if(selected[which]&&selected[which].label===value)return selected[which].request;
      const matches=cities.filter(city=>city.name===value);
      if(matches.length===1)return {cityId:matches[0].id};
      if(matches.length>1)throw Error('有同名聚落，请在地图上选择。');
      throw Error(`请从列表选择${which==='start'?'起点':'终点'}聚落，或在地图上选点。`);
    }
    function draw(route){
      group.replaceChildren();
      const d=route.parts.map(points=>{
        if(route.kind!=='flight')return points.map((p,i)=>`${i?'L':'M'}${p.join(' ')}`).join(' ');
        // Airline-style presentation; distance/time still use the spherical arc.
        const a=points[0],b=points.at(-1),dx=b[0]-a[0],dy=b[1]-a[1],side=dx<0?-1:1;
        const c=[(a[0]+b[0])/2+dy*.08*side,(a[1]+b[1])/2-dx*.08*side];
        return `M${a.join(' ')} Q${c.join(' ')} ${b.join(' ')}`;
      }).join(' ');
      for(const [color,width] of [['#fff',7],['#126fe6',4]]){
        const path=document.createElementNS(SVG,'path');for(const [key,value] of Object.entries({d,fill:'none',stroke:color,'stroke-width':width,'stroke-linecap':'round','stroke-linejoin':'round','vector-effect':'non-scaling-stroke'}))path.setAttribute(key,value);
        if(route.kind==='flight')path.dataset.navigationFlight='';
        group.appendChild(path);}
      for(const [station,label,color] of [[route.start,'起','#087952'],[route.end,'终','#b44b3b']]){
        const marker=document.createElementNS(SVG,'circle');marker.setAttribute('cx',station.native[0]);marker.setAttribute('cy',station.native[1]);marker.setAttribute('r','2');marker.dataset.navigationMarker='';marker.setAttribute('fill',color);marker.setAttribute('stroke','#fff');marker.setAttribute('stroke-width','2');marker.setAttribute('vector-effect','non-scaling-stroke');
        const title=document.createElementNS(SVG,'title');title.textContent=`${label}点：${station.name}`;marker.appendChild(title);group.appendChild(marker);}
      group.setAttribute('aria-label',`${route.label}路线 ${route.distanceKm.toFixed(1)} 公里`);
    }
    async function calculate(){
      const request=++revision;finishPicking();$('navigation-go').disabled=true;group.replaceChildren();$('navigation-result').hidden=true;result=null;status('正在计算路线…');
      try{await load();const value=await ask('route',{start:endpoint('start'),end:endpoint('end'),mode:$('navigation-mode').value});if(request!==revision)return;
        result=value;draw(value);$('navigation-result').hidden=false;
        $('navigation-distance').textContent=`${value.distanceKm.toFixed(1)} km`;
        $('navigation-time').textContent=`预计 ${value.travelDays<1?(value.elapsedHours.toFixed(1)+' 小时'):(value.travelDays.toFixed(1)+' 天')}`
          +(value.kind==='ship'?` · 陆路每日 8 小时，航海每日 ${value.dailyHours} 小时`:` · 每日 ${value.dailyHours} 小时`);
        $('navigation-detail').textContent=`两点球面距离 ${value.directKm.toFixed(1)} km · 累计爬升 ${Math.round(value.ascentM).toLocaleString('zh-CN')} m · 下降 ${Math.round(value.descentM).toLocaleString('zh-CN')} m`
          +(value.mode==='boat'?` · 海路 ${value.seaDistanceKm.toFixed(1)} km`:'');
        const notes=[];for(const [station,label] of [[value.start,'起点'],[value.end,'终点']])if(station.snapKm>.001)notes.push(`${label}已选用最近的通行网络位置，距所选点 ${station.snapKm.toFixed(1)} km`);
        if(value.kind==='flight')notes.push(`直接飞行 · 每日航程 ${value.dailyRangeKm} km`+(value.restCount?` · 途中休息 ${value.restCount} 次`:''));
        if(value.ports.length)notes.push(`上下船港口：${value.ports.join(' → ')}`);
        if(value.mode==='boat'&&!value.seaDistanceKm)notes.push('这条路线全程步行，无需乘船。');
        status(notes.join('。')||`${value.start.name} → ${value.end.name}`);onFit(value.parts.flat());
      }catch(error){if(request===revision)status(error.message);}finally{if(request===revision)$('navigation-go').disabled=false;}
    }
    $('navigation-toggle').addEventListener('click',async()=>{
      panel.hidden=!panel.hidden;$('navigation-toggle').setAttribute('aria-expanded',String(!panel.hidden));
      if(!panel.hidden){status('正在加载路网…');try{await load();status('选择两个聚落，或在地图上选点。');}catch(e){status(e.message);}}
      else finishPicking();
    });
    $('navigation-close').addEventListener('click',()=>{panel.hidden=true;$('navigation-toggle').setAttribute('aria-expanded','false');finishPicking();});
    document.addEventListener('keydown',event=>{if(event.key==='Escape'&&picking){finishPicking();status('已取消地图选点。');event.preventDefault();}},true);
    for(const which of ['start','end']){
      $(`navigation-pick-${which}`).addEventListener('click',()=>pick(which));
      $(`navigation-${which}`).addEventListener('input',()=>{selected[which]=null;});
    }
    $('navigation-form').addEventListener('submit',event=>{event.preventDefault();calculate();});
    $('navigation-swap').addEventListener('click',()=>{const a=$('navigation-start').value;$('navigation-start').value=$('navigation-end').value;$('navigation-end').value=a;
      [selected.start,selected.end]=[selected.end,selected.start];if(result)calculate();});
    $('navigation-mode').addEventListener('change',()=>{if(result)calculate();});
    $('navigation-clear').addEventListener('click',()=>{revision++;group.replaceChildren();result=null;$('navigation-result').hidden=true;status('');$('navigation-go').disabled=false;});
    return {get picking(){return !!picking;},get result(){return result;},cancelPicking:finishPicking,
      addPoint(native,cityId){const which=picking;if(!which)return;const city=cityId?cities.find(c=>c.id===cityId):null;
        const label=city?city.name:`地图选点 (${native[0].toFixed(2)}, ${native[1].toFixed(2)})`;
        selected[which]={label,request:city?{cityId:city.id}:{native}};$(`navigation-${which}`).value=label;finishPicking();status('地点已选择，可计算路线。');},
      updateScale(scale){for(const marker of group.querySelectorAll('[data-navigation-marker]'))marker.setAttribute('r',String(6/scale));}};
  }
  window.WorldAtlasNavigation={create};
})();
