/* Place search and map panels. Geometry, camera and generation stay external. */
(() => {
  'use strict';
  const themeLabels = {none:'地形与水系',climate:'柯本气候',biome:'生物群系',watershed:'水文流域',potential:'农业潜力',
    habitability:'宜居度',vegetation:'植被覆盖',population:'人口分布',civilizations:'文明区',languages:'语言分布',
    religions:'宗教分布',political:'国家政区',provinces:'省份政区',monsoon:'季风',tectonic:'板块'};
  const kinds = {city:0,state:1,province:2,geography:3};
  const isTheme=key=>Object.hasOwn(themeLabels,key)&&!['monsoon','tectonic'].includes(key);
  const population = value => value ? (value[0]===value[1] ? value[0].toLocaleString('zh-CN')
    : `${value[0].toLocaleString('zh-CN')}–${value[1].toLocaleString('zh-CN')}`)+' 人' : '';
  const distance = km => km < 1 ? `${Number((km*1000).toFixed(1))} m` : `${Number(km.toFixed(2))} km`;

  function create({places,onSelectPlace,onRegenerateCity,onClosePlace,onViewChange,onThemeChange,onSeasonChange,onLayerChange,onZoom,onRulerAction}) {
    const callbacks={onSelectPlace,onRegenerateCity,onClosePlace,onViewChange,onThemeChange,onSeasonChange,onLayerChange,onZoom,onRulerAction};
    if (Object.values(callbacks).some(callback=>typeof callback!=='function')) throw new TypeError('Atlas UI needs its map action callbacks');
    if (!Array.isArray(places) && typeof places!=='function') throw new TypeError('Atlas UI needs saved places or a lazy place loader');
    const node = id => {
      const value=document.getElementById(id);
      if (!value) throw new Error(`Missing map interface element: ${id}`);
      return value;
    };
    const search=node('city-search'), results=node('city-results'), suggestions=node('search-suggestions');
    const card=node('place-card'), controls=node('atlas-controls'), info=node('map-info');
    const abort=new AbortController(), signal=abort.signal;
    let loaded=null, loading=null, selected=null, matches=[], activeResult=-1, destroyed=false, selectionRequest=0;
    let state={view:'physical',theme:'none',season:'vernal'}, rulerState={active:false,stations:[],segments:[],totalKm:0};
    const listen=(target,event,handler)=>target.addEventListener(event,handler,{signal});
    const setStatus=message=>{node('map-status').textContent=String(message || '');};
    const safely=operation=>Promise.resolve().then(operation).catch(error=>{if(!destroyed)setStatus(error.message);});
    const lower=value=>value.toLocaleLowerCase('zh-CN');
    const owners=place=>[place.province?.name,place.state?.name].filter(Boolean).join(' · ');
    const subtitle=place=>[place.typeLabel,owners(place),place.frontier ? '无常设政权区' : ''].filter(Boolean).join(' · ');
    function loadPlaces() {
      if (loaded) return Promise.resolve(loaded);
      if (!loading) loading=Promise.resolve().then(()=>typeof places==='function' ? places() : places).then(values=>{
        if (!Array.isArray(values)) throw new TypeError('Saved place index must be an array');
        const identifiers=new Set();
        for (const place of values) {
          if (!place || typeof place.id!=='string' || identifiers.has(place.id) || !Object.hasOwn(kinds,place.kind)
              || typeof place.name!=='string' || !place.name || typeof place.typeLabel!=='string'
              || !Array.isArray(place.native) || place.native.length!==2 || !place.native.every(Number.isFinite)
              || !Number.isFinite(place.longitude) || !Number.isFinite(place.latitude) || Math.abs(place.latitude)>90
              || place.population && (!Array.isArray(place.population) || place.population.length!==2
                || !place.population.every(value=>Number.isSafeInteger(value)&&value>=0) || place.population[1]<place.population[0])) {
            throw new TypeError('Invalid saved place record');
          }
          identifiers.add(place.id);
        }
        loaded=values.slice();return loaded;
      }).catch(error=>{loading=null;throw error;});
      return loading;
    }
    function hideSuggestions() {
      suggestions.hidden=true;search.setAttribute('aria-expanded','false');search.removeAttribute('aria-activedescendant');activeResult=-1;
    }
    function activate(index) {
      activeResult=index;
      Array.from(results.children).forEach((option,i)=>option.setAttribute('aria-selected',String(i===index)));
      if(index>=0){search.setAttribute('aria-activedescendant',results.children[index].id);results.children[index].scrollIntoView({block:'nearest'});}
      else search.removeAttribute('aria-activedescendant');
    }
    async function showResults() {
      const values=await loadPlaces();
      if(destroyed || document.activeElement!==search) return;
      const term=lower(search.value.trim()), score=place=>{
        const name=lower(place.name);
        return name===term ? 0 : name.startsWith(term) ? 1 : name.includes(term) ? 2 : 3;
      };
      const all=values.filter(place=>!term || lower(`${place.name} ${owners(place)} ${place.typeLabel}`).includes(term));
      all.sort((a,b)=>score(a)-score(b)||kinds[a.kind]-kinds[b.kind]||Number(!!b.isCapital)-Number(!!a.isCapital)
        ||(b.population?.[1] || 0)-(a.population?.[1] || 0)||a.name.localeCompare(b.name,'zh-CN'));
      matches=all.slice(0,term ? 40 : 12);activeResult=-1;
      const fragment=document.createDocumentFragment();
      matches.forEach((place,index)=>{
        const option=document.createElement('li');option.id=`place-option-${index}`;option.role='option';
        option.setAttribute('aria-selected','false');option.dataset.placeId=place.id;
        const name=document.createElement('strong');name.textContent=place.name;
        const detail=document.createElement('span');detail.textContent=subtitle(place);option.append(name,detail);fragment.appendChild(option);
      });
      results.replaceChildren(fragment);
      node('search-summary').textContent=term ? all.length ? `${all.length} 个结果${all.length>40 ? ' · 显示前 40 项' : ''}` : '没有匹配的地点' : '探索当前世界';
      suggestions.hidden=false;search.setAttribute('aria-expanded','true');search.removeAttribute('aria-activedescendant');
      node('search-clear').hidden=!search.value;setStatus('');
    }
    function showCard(place) {
      selected=place;node('place-name').textContent=place.name;
      card.dataset.kind=place.kind;
      node('place-panel').classList.toggle('has-city',place.kind==='city');
      node('place-recenter').textContent='在地图中定位';
      node('place-regenerate').hidden=place.kind!=='city';
      node('city-map-section').hidden=place.kind!=='city';
      node('place-extra').hidden=place.kind!=='city';node('place-extra').open=false;
      node('place-kind').textContent=[place.typeLabel,place.siteLabel,place.isCapital ? '首都' : ''].filter(Boolean).join(' · ');
      const facts=[];
      if(place.population) facts.push(['人口区间',population(place.population)]);
      if(owners(place)) facts.push([place.ownerLabel || '行政归属',owners(place)]);
      if(place.frontier) facts.push(['治理', '无常设政权区']);
      if(place.eraLabel) facts.push(['时代', place.eraLabel]);
      if(place.administrationLabel && !place.frontier) facts.push(['治理', place.administrationLabel]);
      if(Number.isFinite(place.centralTravelDays)) facts.push(['中央通达', `${place.centralTravelDays.toFixed(1)} 天（模拟）`]);
      if(place.nominalRealm) facts.push(['名义臣属',place.nominalRealm.name+'势力范围']);
      if(place.capitalName) facts.push([place.kind==='state' ? '首都' : '首府',place.capitalName]);
      if(place.holyReligion) facts.push(['圣城',place.holyReligion]);
      facts.push(['经纬度',`${Math.abs(place.latitude).toFixed(3)}°${place.latitude<0?'S':'N'} · ${Math.abs(place.longitude).toFixed(3)}°${place.longitude<0?'W':'E'}`]);
      node('place-facts').replaceChildren();node('place-extra-facts').replaceChildren();
      for(const [label,value] of facts) appendFact(place.kind==='city'&&!['人口区间','时代'].includes(label)?node('place-extra-facts'):node('place-facts'),label,value);
      card.hidden=false;hideSuggestions();setStatus('');
    }
    function appendFact(list,label,value,generated=false) {
      const item=document.createElement('div'),term=document.createElement('dt'),description=document.createElement('dd');
      if(generated)item.dataset.generatedFact='';term.textContent=label;description.textContent=value;item.append(term,description);list.append(item);
    }
    function setCityDetails(recipe) {
      if(!recipe || selected?.sourceId!==recipe.id || selected.kind!=='city')return;
      node('place-facts').querySelectorAll('[data-generated-fact]').forEach(node=>node.remove());
      node('place-extra-facts').querySelectorAll('[data-generated-fact]').forEach(node=>node.remove());
      const character=WorldAtlasCityCharacter.profile(recipe);
      const facts=[['聚落',character.label],['供水',recipe.localSite.waterSource.label],['地形',recipe.localSite.form.label]];
      if(recipe.harbor)facts.push(['港口','⚓ '+recipe.harbor.name]);
      for(const [label,value] of facts)appendFact(node('place-facts'),label,value,true);
      appendFact(node('place-extra-facts'),'规划面积',`${recipe.urban.targetFootprintKm2.toLocaleString('zh-CN',{maximumFractionDigits:2})} km²`,true);
      if(character.defence.layers)appendFact(node('place-extra-facts'),'城防',character.defence.layers>1?'内城、外城'+(character.defence.barbicans?'与瓮城':'与城门'):'城墙与城门',true);
    }
    function writePlace(id) {
      const url=new URL(location.href);
      if((url.searchParams.get('place') || null)===id) return;
      if(id) url.searchParams.set('place',id);else url.searchParams.delete('place');
      history.pushState(history.state,'',url.pathname+url.search+url.hash);
    }
    async function selectPlace(id,{restore=false,history:saveHistory=true,center=true}={}) {
      const request=++selectionRequest;
      const values=await loadPlaces();
      if(destroyed || request!==selectionRequest)return false;
      const place=values.find(value=>value.id===id);
      if(!place){setStatus('未找到存档中的地点');return false;}
      if(saveHistory)writePlace(place.id);
      showCard(place);
      await onSelectPlace(place,{restore,center});return true;
    }
    function closePlace({history:saveHistory=true,restore=false}={}) {
      selectionRequest++;
      if(!selected)return false;
      selected=null;card.hidden=true;node('place-panel').classList.remove('has-city');
      if(saveHistory)writePlace(null);
      onClosePlace({restore});return true;
    }
    function setPanel(panel,toggle,open) {
      panel.hidden=!open;panel.inert=!open;toggle.setAttribute('aria-expanded',String(open));
    }
    function setControls(open) {
      setPanel(controls,node('toggle-controls'),open);
      if(open)setPanel(info,node('map-info-toggle'),false);
    }
    function sync(next) {
      const updated={...state,...next};
      if(!['physical','monsoon','tectonic'].includes(updated.view) || !isTheme(updated.theme)
          || !['vernal','june','autumnal','december'].includes(updated.season)) throw new RangeError('Invalid current map UI state');
      state=updated;
      node('physical-controls').hidden=state.view!=='physical';node('monsoon-controls').hidden=state.view!=='monsoon';node('tectonic-controls').hidden=state.view!=='tectonic';
      document.querySelectorAll('[data-theme-button]').forEach(button=>button.setAttribute('aria-pressed',String(state.view==='physical'&&button.dataset.themeButton===state.theme)));
      document.querySelectorAll('[data-view-button]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.viewButton===state.view)));
      document.querySelectorAll('[data-season-button]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.seasonButton===state.season)));
      if(next.layers)controls.querySelectorAll('input[data-layer]').forEach(input=>{if(input.dataset.layer in next.layers)input.checked=!!next.layers[input.dataset.layer];});
      const key=state.view==='physical' ? state.theme : state.view;
      let hasLegend=false;
      node('map-legend-body').querySelectorAll('[data-legend-key]').forEach(legend=>{legend.hidden=legend.dataset.legendKey!==key;if(!legend.hidden)hasLegend=true;});
      node('map-legend').hidden=!hasLegend;node('legend-title').textContent=themeLabels[key]+'图例';
    }
    function setRulerState(value) {
      if(!value || !Number.isFinite(value.totalKm) || value.totalKm<0 || !Array.isArray(value.stations) || !Array.isArray(value.segments)) throw new TypeError('Invalid ruler summary');
      rulerState=value;node('ruler-toggle').setAttribute('aria-pressed',String(value.active));
      node('ruler-card').hidden=!value.active&&!value.stations.length;
      node('ruler-summary').textContent=distance(value.totalKm);
      node('ruler-hint').textContent=`${value.stations.length} 个站点 · ${value.active ? '点击地图添加测距点；拖动和缩放不会添加站点。' : '测量已完成'}`;
      const fragment=document.createDocumentFragment();
      value.segments.forEach((segment,index)=>{const item=document.createElement('li');item.textContent=`第 ${index+1} 段：${distance(segment.distanceKm)}`;fragment.appendChild(item);});
      node('ruler-segments').replaceChildren(fragment);
      node('ruler-undo').disabled=!value.stations.length;node('ruler-clear').disabled=!value.stations.length;node('ruler-finish').disabled=!value.active;
    }

    listen(search,'focus',()=>safely(showResults));listen(search,'input',()=>safely(showResults));
    listen(search,'keydown',event=>{
      if(event.key==='ArrowDown' || event.key==='ArrowUp'){
        event.preventDefault();if(suggestions.hidden){safely(showResults);return;}
        if(matches.length){const step=event.key==='ArrowDown'?1:-1;activate(activeResult<0 ? step>0 ? 0 : matches.length-1 : (activeResult+step+matches.length)%matches.length);}
      }else if(event.key==='Enter' && !suggestions.hidden && matches.length){event.preventDefault();safely(()=>selectPlace(matches[Math.max(0,activeResult)].id));}
    });
    listen(results,'pointerdown',event=>event.preventDefault());
    listen(results,'click',event=>{const result=event.target.closest('[data-place-id]');if(result)safely(()=>selectPlace(result.dataset.placeId));});
    listen(node('search-clear'),'click',()=>{search.value='';search.focus();safely(showResults);});
    listen(node('place-close'),'click',()=>closePlace());
    listen(node('place-recenter'),'click',()=>{if(selected)safely(()=>onSelectPlace(selected,{restore:false,center:true,generate:false}));});
    listen(node('place-regenerate'),'click',()=>{if(selected?.kind==='city')safely(()=>onRegenerateCity(selected));});
    listen(node('toggle-controls'),'click',()=>setControls(controls.hidden));listen(node('layers-close'),'click',()=>{setControls(false);node('toggle-controls').focus();});
    listen(node('map-info-toggle'),'click',()=>{const open=info.hidden;setPanel(info,node('map-info-toggle'),open);if(open)setControls(false);});
    listen(node('map-info-close'),'click',()=>{setPanel(info,node('map-info-toggle'),false);node('map-info-toggle').focus();});
    listen(node('legend-toggle'),'click',()=>{const body=node('map-legend-body');body.hidden=!body.hidden;node('legend-toggle').setAttribute('aria-expanded',String(!body.hidden));});
    controls.querySelectorAll('[data-theme-button]').forEach(button=>listen(button,'click',()=>safely(()=>{const action=onThemeChange(button.dataset.themeButton);if(matchMedia('(max-width:700px)').matches)setControls(false);return action;})));
    controls.querySelectorAll('[data-view-button]').forEach(button=>listen(button,'click',()=>safely(()=>onViewChange(button.dataset.viewButton))));
    controls.querySelectorAll('[data-season-button]').forEach(button=>listen(button,'click',()=>safely(()=>onSeasonChange(button.dataset.seasonButton))));
    controls.querySelectorAll('input[data-layer]').forEach(input=>listen(input,'change',()=>safely(()=>onLayerChange(input.dataset.layer,input.checked))));
    for(const [identifier,action] of [['zoom-in','in'],['zoom-out','out'],['zoom-reset','reset']])listen(node(identifier),'click',()=>safely(()=>onZoom(action)));
    listen(node('ruler-toggle'),'click',()=>safely(()=>onRulerAction(rulerState.active?'finish':'start')));
    for(const action of ['undo','clear','finish'])listen(node(`ruler-${action}`),'click',()=>safely(()=>onRulerAction(action)));
    listen(document,'pointerdown',event=>{if(!node('place-panel').contains(event.target))hideSuggestions();});
    listen(document,'keydown',event=>{
      if(event.key!=='Escape' || event.defaultPrevented)return;
      if(!suggestions.hidden){hideSuggestions();event.preventDefault();}
      else if(!controls.hidden){setControls(false);node('toggle-controls').focus();event.preventDefault();}
      else if(!info.hidden){setPanel(info,node('map-info-toggle'),false);node('map-info-toggle').focus();event.preventDefault();}
      else if(selected){closePlace();search.focus();event.preventDefault();}
    });
    listen(window,'popstate',()=>{
      const identifier=new URLSearchParams(location.search).get('place');
      if(identifier)safely(()=>selectPlace(identifier,{restore:true,history:false}));
      else closePlace({history:false,restore:true});
    });
    const query=new URLSearchParams(location.search);
    sync({view:['monsoon','tectonic'].includes(query.get('view'))?query.get('view'):'physical',
      theme:isTheme(query.get('theme'))?query.get('theme'):'none',
      season:['vernal','june','autumnal','december'].includes(query.get('season'))?query.get('season'):'vernal'});
    if(query.get('place'))safely(()=>selectPlace(query.get('place'),{restore:true,history:false}));
    return {selectPlace,closePlace,sync,setRulerState,setStatus,setCityDetails,
      get selectedPlaceId(){return selected?.id || null;},
      destroy(){destroyed=true;abort.abort();hideSuggestions();},
    };
  }
  window.WorldAtlasUI={create};
})();
