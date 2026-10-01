/* Mount only the detailed geometry currently inside the native map viewport. */
(() => {
  'use strict';
  const SVG_NS = 'http://www.w3.org/2000/svg';
  const MAX_REQUESTS = 4;
  const CACHE_LIMIT = 64;

  function create({container, onMounted = () => {}, loadError = () => {}}) {
    if (!(container instanceof SVGElement)) throw new TypeError('Atlas tiles need a native SVG container');
    const active = new Map();
    const cache = new Map();
    const pending = new Map();
    const failed = new Set();
    let manifest = null;
    let current = null;
    let desired = new Set();
    let queue = [];
    let generation = 0;
    let notificationFrame = 0;
    let reconciliationKey = null;

    function tileAddress(key) {
      const [level, position] = key.split('/');
      const [row, column] = position.split('-').map(Number);
      return {level, position, row, column};
    }

    function notify() {
      if (!notificationFrame) notificationFrame = requestAnimationFrame(() => {
        notificationFrame = 0;
        onMounted();
      });
    }

    function remember(key, tile) {
      cache.delete(key);
      cache.set(key, tile);
      while (cache.size > CACHE_LIMIT) cache.delete(cache.keys().next().value);
    }

    function detach(key) {
      const tile = active.get(key);
      if (!tile) return;
      if (!key.startsWith('city-detail/')) {
        const fineKey='city-detail/'+tileAddress(key).position;
        if(active.has(fineKey))detach(fineKey);
      }
      if(tile.svg)tile.svg.remove();
      for(const node of tile.nodes||[])node.remove();
      active.delete(key);
      remember(key, tile);
      notify();
    }

    function present(tile) {
      if (tile.presentation === current.presentation) return;
      if (!tile.overlay && tile.theme !== current.theme) {
        tile.themeGroup.innerHTML = tile.payload.themes[current.theme];
        tile.theme = current.theme;
      }
      const roots=tile.nodes||[tile.svg];
      for (const node of roots.flatMap(root=>Array.from(root.querySelectorAll('[data-tile-layer], [data-season]')))) {
        const visible = current.layerVisibility[node.dataset.tileLayer] !== false
          && (!node.dataset.season || node.dataset.season === current.season);
        node.toggleAttribute('hidden', !visible);
        node.style.display = visible ? '' : 'none';
        if(node.dataset.tileLayer==='geographic-textures')node.setAttribute('opacity',
          ['potential','habitability','vegetation','population'].includes(current.theme)?'.28'
          :['political','provinces'].includes(current.theme)?'.6':current.theme!=='none'?'.75':'1');
      }
      tile.presentation = current.presentation;
      notify();
    }

    function scalePresentation(tile) {
      const roots=tile.nodes||[tile.svg];
      if(!tile.scaleNodes){
        tile.scaleNodes=roots.flatMap(root=>Array.from(root.querySelectorAll('[data-min-scale], [data-max-scale]')));
        tile.patterns=roots.flatMap(root=>Array.from(root.querySelectorAll('pattern[data-landform-pattern]')));
      }
      // A cartographic repeat changes only at an octave boundary. Ordinary
      // wheel frames transform cached paint, rather than rebuild textures.
      const bucket=2**Math.floor(Math.log2(current.scale));
      if(tile.patternScale!==bucket){
        for(const pattern of tile.patterns)pattern.setAttribute('patternTransform',`scale(${1/bucket})`);
        tile.patternScale=bucket;
      }
      for(const node of tile.scaleNodes){
        const visible=current.scale>=Number(node.dataset.minScale||0)
          &&current.scale<Number(node.dataset.maxScale||Infinity);
        if(node._atlasScaleVisible!==visible){
          node.toggleAttribute('hidden',!visible);node.style.display=visible?'':'none';
          node._atlasScaleVisible=visible;
        }
      }
    }

    function mount(key, tile) {
      if (!desired.has(key) || !current || !current.level) {
        remember(key, tile);
        return;
      }
      if (active.has(key)) return;
      const address = tileAddress(key);
      if(address.level==='city-detail') {
        const base=active.get('detail/'+address.position);
        if(!base){remember(key,tile);return;}
        cache.delete(key);
        if(!tile.nodes) {
          tile.overlay=true;
          tile.nodes=['surface','ink'].map(name=>{
            const group=document.createElementNS(SVG_NS,'g');
            group.dataset.atlasRefinement='city-detail';
            group.dataset.atlasTile=address.position;
            group.innerHTML=tile.payload[name];
            return group;
          });
        }
        active.set(key,tile);
        present(tile);
        scalePresentation(tile);
        base.svg.querySelector(':scope > [data-atlas-layer="surface"]').appendChild(tile.nodes[0]);
        const ink=base.svg.querySelector(':scope > [data-atlas-layer="ink"]');
        ink.prepend(tile.nodes[1]);
        notify();
        return;
      }
      cache.delete(key);
      // Immutable blocks retain their already parsed SVG while detached.
      // Crossing a tile edge repeatedly must not rebuild the same path tree.
      // Both detached markup and DOM share the existing 64-block LRU bound.
      if (!tile.svg) {
        const svg = document.createElementNS(SVG_NS, 'svg');
        const [x, y, width, height] = tile.payload.bounds;
        for (const [name, value] of Object.entries({x, y, width, height,
          viewBox: tile.payload.bounds.join(' '), preserveAspectRatio: 'none', overflow: 'hidden'})) {
          svg.setAttribute(name, value);
        }
        svg.dataset.atlasTile = address.position;
        svg.dataset.atlasLod = address.level;
        const parts = {};
        for (const name of ['surface', 'theme', 'ink']) {
          const group = document.createElementNS(SVG_NS, 'g');
          group.dataset.atlasLayer = name;
          group.innerHTML = name === 'theme' ? '' : tile.payload[name];
          parts[name] = group;
          svg.appendChild(group);
        }
        Object.assign(tile,{svg,themeGroup:parts.theme,theme:null,presentation:null});
      }
      active.set(key, tile);
      present(tile);
      scalePresentation(tile);
      container.appendChild(tile.svg);
      const fineKey='city-detail/'+address.position;
      if(desired.has(fineKey)&&cache.has(fineKey))mount(fineKey,cache.get(fineKey));
      notify();
    }

    function validatePayload(payload, key) {
      const {level,row, column} = tileAddress(key);
      const x = column * manifest.tileSize, y = row * manifest.tileSize;
      const bounds = [x, y, Math.min(manifest.tileSize, manifest.width - x),
        Math.min(manifest.tileSize, manifest.height - y)];
      if (!payload || !Array.isArray(payload.bounds) || payload.bounds.length !== 4
          || payload.bounds.some((value, index) => value !== bounds[index])
          || typeof payload.surface !== 'string' || typeof payload.ink !== 'string'
          || (level==='city-detail' ? payload.themes!==undefined
            : !payload.themes || manifest.themes.some(theme => typeof payload.themes[theme] !== 'string'))) {
        throw new Error('Invalid published atlas tile');
      }
      return payload;
    }

    function pump() {
      while (pending.size < MAX_REQUESTS && queue.length) {
        const key = queue.shift();
        if (!desired.has(key) || active.has(key) || pending.has(key) || failed.has(key)) continue;
        const request = {controller: new AbortController(), generation};
        pending.set(key, request);
        fetch(`tiles/${key}.json`, {signal: request.controller.signal, cache: 'no-store'})
          .then(response => {
            if (!response.ok) throw new Error(`Atlas tile HTTP ${response.status}`);
            return response.json();
          })
          .then(payload => {
            if (request.controller.signal.aborted) return;
            payload = validatePayload(payload, key);
            // Geometry is immutable. A still-visible old camera request is
            // reusable, but it always paints the latest theme and controls.
            if (request.generation !== generation && !desired.has(key)) remember(key, {payload});
            else mount(key, {payload});
          })
          .catch(error => {
            if (request.controller.signal.aborted || error.name === 'AbortError') return;
            if (desired.has(key)) {
              failed.add(key);
              loadError('部分地图暂未载入，已载入的区域仍可查看。');
            }
          })
          .finally(() => {
            if (pending.get(key) === request) pending.delete(key);
            // An aborted old request can finish after the same block becomes
            // visible again. Requeue it rather than leaving a permanent hole.
            if (desired.has(key) && !active.has(key) && !cache.has(key) && !failed.has(key) && !queue.includes(key)) queue.push(key);
            pump();
          });
      }
    }

    function reconcile() {
      if (!manifest || !current) return;
      if (!manifest.themes.includes(current.theme)) {
        loadError('当前地图专题未发布。');
        return;
      }
      const [west, north, east, south] = current.worldBounds;
      current.level = null;
      for (const level of manifest.levels.slice(0,3)) {
        if (current.scale >= level.minScale) current.level = level.id;
      }
      const firstColumn = Math.max(0, Math.floor(west / manifest.tileSize));
      const lastColumn = Math.min(manifest.columns - 1, Math.ceil(east / manifest.tileSize) - 1);
      const firstRow = Math.max(0, Math.floor(north / manifest.tileSize));
      const lastRow = Math.min(manifest.rows - 1, Math.ceil(south / manifest.tileSize) - 1);
      const drawingLevel = current.level && east > west && south > north ? current.level : null;
      const cityEnabled=current.scale>=manifest.levels[3].minScale&&drawingLevel==='detail';
      const signature = JSON.stringify([drawingLevel,
        drawingLevel ? [firstColumn, lastColumn, firstRow, lastRow] : null,cityEnabled,current.presentation]);
      if (signature === reconciliationKey) return;
      reconciliationKey = signature;
      const keys = [];
      if (drawingLevel) {
        for (let row = firstRow; row <= lastRow; row++) {
          for (let column = firstColumn; column <= lastColumn; column++) {
            const position=`${row}-${column}`;
            keys.push(`${current.level}/${position}`);
            if(cityEnabled&&manifest.cityPositions.has(position))keys.push(`city-detail/${position}`);
          }
        }
      }
      const next = new Set(keys);
      // Reserve incoming cache hits before outgoing blocks fill the LRU. An
      // old requested block must not be evicted by the block it replaces.
      const cached = new Map();
      for (const key of keys) if (!active.has(key) && cache.has(key)) {
        cached.set(key, cache.get(key));
        cache.delete(key);
      }
      for (const key of active.keys()) if (!next.has(key)) detach(key);
      for (const key of failed) if (!next.has(key)) failed.delete(key);
      desired = next;
      for (const [key, request] of pending) if (!desired.has(key)) request.controller.abort();
      for (const key of keys) {
        if (active.has(key)) present(active.get(key));
        else if (cached.has(key)) mount(key, cached.get(key));
      }
      const centerX = (west + east) / 2, centerY = (north + south) / 2;
      const distance = key => {
        const {row, column} = tileAddress(key);
        return (column * manifest.tileSize + manifest.tileSize / 2 - centerX) ** 2
          + (row * manifest.tileSize + manifest.tileSize / 2 - centerY) ** 2;
      };
      queue = keys.filter(key => !active.has(key) && !cache.has(key) && !pending.has(key) && !failed.has(key))
        .sort((first, second) => distance(first) - distance(second));
      pump();
    }

    fetch('atlas-manifest.json', {cache: 'no-store'})
      .then(response => {
        if (!response.ok) throw new Error(`Atlas manifest HTTP ${response.status}`);
        return response.json();
      })
      .then(value => {
        if (!value || !['width', 'height', 'tileSize', 'columns', 'rows'].every(key => Number.isInteger(value[key]) && value[key] > 0)
            || !Array.isArray(value.levels) || value.levels.length !== 4
            || value.levels.slice(0,3).some((level, index) => !level
              || level.id !== ['regional', 'local', 'detail'][index]
              || level.minScale !== [8, 32, 64][index])
            || value.levels[3].id!=='city-detail'||value.levels[3].minScale!==128
            || value.levels[3].kind!=='relief-overlay'||!Array.isArray(value.levels[3].publishedTiles)
            || new Set(value.levels[3].publishedTiles).size!==value.levels[3].publishedTiles.length
            || value.levels[3].publishedTiles.some(position=>{
              if(typeof position!=='string'||!/^\d+-\d+$/.test(position))return true;
              const [row,column]=position.split('-').map(Number);
              return row>=value.rows||column>=value.columns||`${row}-${column}`!==position;
            })
            || value.columns !== Math.ceil(value.width / value.tileSize)
            || value.rows !== Math.ceil(value.height / value.tileSize)
            || !Array.isArray(value.themes) || !value.themes.includes('none')
            || value.themes.some(theme => typeof theme !== 'string')
            || new Set(value.themes).size !== value.themes.length) throw new Error('Invalid published atlas manifest');
        manifest = value;
        manifest.cityPositions=new Set(value.levels[3].publishedTiles);
        reconcile();
      })
      .catch(() => loadError('详细地图暂未载入，世界轮廓仍可查看。'));

    return {
      update({scale, worldBounds, theme, season, layerVisibility}) {
        if (!Number.isFinite(scale) || scale <= 0 || !Array.isArray(worldBounds) || worldBounds.length !== 4
            || !worldBounds.every(Number.isFinite) || typeof theme !== 'string' || typeof season !== 'string'
            || !layerVisibility || typeof layerVisibility !== 'object') throw new TypeError('Invalid atlas camera state');
        if (manifest && !manifest.themes.includes(theme)) {
          loadError('当前地图专题未发布。');
          return;
        }
        generation++;
        const visibility = {...layerVisibility};
        current = {scale, worldBounds: worldBounds.slice(), theme, season, layerVisibility: visibility,
          presentation: JSON.stringify([theme, season, Object.entries(visibility).sort(([first], [second]) => first.localeCompare(second))])};
        for(const tile of active.values())scalePresentation(tile);
        reconcile();
      },
      clear() {
        generation++;
        current = null;
        reconciliationKey = null;
        desired = new Set();
        queue = [];
        failed.clear();
        for (const request of pending.values()) request.controller.abort();
        for (const key of Array.from(active.keys())) detach(key);
        cache.clear();
        notify();
      },
      stats() {
        const fineTiles=Array.from(active.keys()).filter(key=>key.startsWith('city-detail/')).length;
        return {activeTiles: active.size,activeBaseTiles:active.size-fineTiles,activeCityTiles:fineTiles,
          baseTilesReady:!!current?.level && desired.size>0
            && Array.from(desired).every(key=>key.startsWith('city-detail/') || active.has(key)),
          cachedTiles: cache.size, pendingRequests: pending.size,
          queuedTiles: queue.length, failedTiles: failed.size, generation, manifestLoaded: manifest !== null,
          activeLod: fineTiles ? 'city-detail' : current?.level || null};
      },
    };
  }

  window.WorldAtlasTiles = {create};
})();
