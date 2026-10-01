/* Cache the world preview and sharpen only the visible region at screen density. */
(() => {
  'use strict';
  const SVG_NS = 'http://www.w3.org/2000/svg';
  const CACHE_LIMIT = 4;

  function create({sourceURL, container, width, height, rasterScale = 2, onReady = () => {}, loadError = () => {}}) {
    if (typeof sourceURL !== 'string' || !sourceURL || !(container instanceof SVGElement)
        || !Number.isInteger(width) || width < 1 || !Number.isInteger(height) || height < 1 || rasterScale !== 2) {
      throw new TypeError('Atlas overview needs native SVG geometry and dimensions');
    }
    const image = document.createElementNS(SVG_NS, 'image');
    for (const [name, value] of Object.entries({x:0, y:0, width, height, preserveAspectRatio:'none'})) image.setAttribute(name, value);
    image.dataset.atlasOverviewRaster = '';
    const refinement = document.createElementNS(SVG_NS, 'image');
    refinement.setAttribute('preserveAspectRatio', 'none');
    refinement.dataset.atlasOverviewRefinement = '';
    container.replaceChildren(image, refinement);
    const cache = new Map();
    const themeCache = new Map();
    const physicalPaint = new Map();
    const themes = new Map();
    let source = null;
    let wanted = null;
    let queued = null;
    let active = null;
    let currentKey = null;
    let rasterizations = 0;
    let camera = null;
    let refined = null;
    let refinementTimer = 0;
    let refinementJob = null;
    let queuedRefinement = null;
    let failedRefinement = null;
    let viewportRasterizations = 0;

    function covers(frame, view) {
      return frame && view && frame.key === currentKey
        && frame.density >= view.scale * view.pixelRatio
        && frame.bounds[0] <= view.bounds[0] && frame.bounds[1] <= view.bounds[1]
        && frame.bounds[2] >= view.bounds[2] && frame.bounds[3] >= view.bounds[3];
    }

    function discardRefinement() {
      window.clearTimeout(refinementTimer);
      refinementTimer = 0;
      queuedRefinement = null;
      failedRefinement = null;
      refinement.removeAttribute('href');
      image.style.display = '';
      if (refined) URL.revokeObjectURL(refined.uri);
      refined = null;
    }

    function requestRefinement() {
      image.style.display = covers(refined, camera) ? 'none' : '';
      if (camera && !camera.refine) {
        window.clearTimeout(refinementTimer);
        refinementTimer = 0;
        queuedRefinement = null;
        return;
      }
      if (!camera || camera.scale * camera.pixelRatio <= rasterScale
          || camera.bounds[2] <= camera.bounds[0] || camera.bounds[3] <= camera.bounds[1]) {
        discardRefinement();
        return;
      }
      if (!currentKey || wanted?.key !== currentKey || covers(refined, camera)
          || covers(refinementJob, camera) || covers(queuedRefinement, camera)
          || covers(failedRefinement, camera)) return;
      const [west, north, east, south] = camera.bounds;
      const marginX = (east - west) * .15, marginY = (south - north) * .15;
      queuedRefinement = {
        key:currentKey, scale:camera.scale, pixelRatio:camera.pixelRatio,
        density:2 ** Math.ceil(Math.log2(camera.scale * camera.pixelRatio)),
        bounds:[Math.max(0, Math.floor((west-marginX)/16)*16),
          Math.max(0, Math.floor((north-marginY)/16)*16),
          Math.min(width, Math.ceil((east+marginX)/16)*16),
          Math.min(height, Math.ceil((south+marginY)/16)*16)],
      };
      window.clearTimeout(refinementTimer);
      // Nearby camera frames reuse the same padded request; no whole-world
      // repaint or per-frame decode is needed during a wheel gesture.
      refinementTimer = window.setTimeout(pumpRefinement, 80);
    }

    async function pumpRefinement() {
      refinementTimer = 0;
      if (refinementJob || !queuedRefinement) return;
      const frame = queuedRefinement;
      queuedRefinement = null;
      if (frame.key !== currentKey || !cache.has(frame.key)) return;
      refinementJob = frame;
      let uri = null;
      try {
        const scene = cache.get(frame.key).scene.cloneNode(true);
        const [west, north, east, south] = frame.bounds;
        const nativeWidth = east-west, nativeHeight = south-north;
        const pixelWidth = Math.ceil(nativeWidth*frame.density);
        const pixelHeight = Math.ceil(nativeHeight*frame.density);
        scene.setAttribute('viewBox', `${west} ${north} ${nativeWidth} ${nativeHeight}`);
        scene.setAttribute('width', pixelWidth);
        scene.setAttribute('height', pixelHeight);
        const screenStroke = frame.density/frame.scale;
        for (const node of scene.querySelectorAll('[vector-effect="non-scaling-stroke"]')) {
          node.setAttribute('stroke-width', Number(node.getAttribute('stroke-width') || 1)*screenStroke);
          const dash = node.getAttribute('stroke-dasharray');
          if (dash && dash !== 'none') node.setAttribute('stroke-dasharray',
            dash.trim().split(/[\s,]+/).map(value=>Number(value)*screenStroke).join(' '));
        }
        const painted = await paintVector(new XMLSerializer().serializeToString(scene), pixelWidth, pixelHeight);
        const blob = await new Promise((resolve, reject) => painted.toBlob(
          result => result ? resolve(result) : reject(new Error('Atlas region encoding failed')), 'image/png'));
        uri = URL.createObjectURL(blob);
        // Decode before replacing the previous sharp region to avoid a blank
        // image while the browser uploads the new screen-density buffer.
        const decoded = new Image(); decoded.src = uri; await decoded.decode();
        viewportRasterizations++;
        if (wanted?.key !== frame.key || !covers(frame, camera)) return;
        if (refined) URL.revokeObjectURL(refined.uri);
        refined = {...frame, uri, pixelWidth, pixelHeight};
        for (const [name, value] of Object.entries({x:west, y:north,
          width:nativeWidth, height:nativeHeight, href:uri})) refinement.setAttribute(name, value);
        image.style.display = 'none';
        uri = null;
        onReady();
      } catch (error) {
        failedRefinement = frame;
        if (wanted?.key === frame.key) loadError('当前区域清晰底图暂未更新。');
      } finally {
        if (uri) URL.revokeObjectURL(uri);
        refinementJob = null;
        if (queuedRefinement) pumpRefinement();
        else requestRefinement();
      }
    }

    async function sourceDocument(signal) {
      if (source) return;
      const response = await fetch(sourceURL, {signal, cache:'force-cache'});
      if (!response.ok) throw new Error(`Atlas source HTTP ${response.status}`);
      const document = new DOMParser().parseFromString(await response.text(),'image/svg+xml');
      const root = document.documentElement;
      const candidate = root.querySelector('#overview-source');
      if (root.localName !== 'svg' || root.getAttribute('viewBox') !== `0 0 ${width} ${height}`
          || !candidate || document.querySelector('parsererror')) throw new Error('Invalid published overview source');
      candidate.remove();
      source = candidate;
      for (const node of source.querySelectorAll('[data-theme-map]')) themes.set(node.dataset.themeMap,node.dataset.source);
    }

    function remember(key, entry) {
      cache.delete(key);
      cache.set(key, entry);
      while (cache.size > CACHE_LIMIT) {
        const oldest = Array.from(cache.keys()).find(candidate => candidate !== currentKey);
        URL.revokeObjectURL(cache.get(oldest).uri);
        cache.delete(oldest);
      }
    }

    function show(key, entry) {
      remember(key, entry);
      image.setAttribute('href', entry.uri);
      if (currentKey !== key) discardRefinement();
      currentKey = key;
      requestRefinement();
      onReady();
    }

    async function themeDocument(theme, signal) {
      if (theme === 'none') return null;
      const url = themes.get(theme);
      if (!url) throw new Error('Atlas overview theme was not published');
      if (themeCache.has(url)) {
        const markup = themeCache.get(url);
        themeCache.delete(url); themeCache.set(url, markup);
        return markup;
      }
      const response = await fetch(url, {signal, cache:'force-cache'});
      if (!response.ok) throw new Error(`Atlas overview HTTP ${response.status}`);
      const markup = await response.text();
      themeCache.set(url, markup);
      while (themeCache.size > CACHE_LIMIT) themeCache.delete(themeCache.keys().next().value);
      return markup;
    }

    function compose(state, thematicMarkup) {
      const svg = document.createElementNS(SVG_NS, 'svg');
      svg.setAttribute('xmlns', SVG_NS);
      for (const [name, value] of Object.entries({width:width*rasterScale, height:height*rasterScale, viewBox:`0 0 ${width} ${height}`})) svg.setAttribute(name, value);
      const body = source.cloneNode(true);
      body.removeAttribute('hidden');
      body.style.removeProperty('display');
      body.style.removeProperty('visibility');
      const thematic = state.view === 'physical' && state.theme !== 'none';
      const administrative = thematic && ['political', 'provinces'].includes(state.theme);
      const quantitative = thematic && ['potential', 'habitability', 'vegetation', 'population'].includes(state.theme);
      for (const node of body.querySelectorAll('[data-tile-layer], [data-season]')) {
        const layer = node.dataset.tileLayer;
        const visible = state.layerVisibility[layer] !== false && (!node.dataset.season || node.dataset.season === state.season);
        node.removeAttribute('hidden');
        node.style.display = visible ? '' : 'none';
        if (layer === 'elevation-bands') node.setAttribute('opacity', quantitative ? '.12' : administrative ? '.16' : thematic ? '.34' : '1');
        if (layer === 'bathymetry-bands') node.setAttribute('opacity', administrative ? '.72' : thematic ? '.62' : '1');
        if (layer === 'elevation-contours') node.setAttribute('opacity', quantitative ? '.24' : administrative ? '.18' : thematic ? '.36' : '.60');
        if (layer === 'geographic-textures') node.setAttribute('opacity',quantitative?'.28':administrative?'.6':thematic?'.75':'1');
      }
      const seasonIndex = ['vernal', 'june', 'autumnal', 'december'].indexOf(state.season);
      for (const path of body.querySelectorAll('[data-tile-layer="rivers"] path')) {
        const strengths = (path.dataset.seasonalStrengths || '0,0,0,0').split(',').map(Number);
        const strength = strengths[seasonIndex] || 0;
        path.style.display = state.view === 'physical' || strength > 0 ? '' : 'none';
        path.setAttribute('opacity', state.view === 'physical' ? '1' : strength >= 48 ? '.96' : '.68');
        const navigation = state.view === 'physical' && state.layerVisibility['transport-network'] !== false && path.dataset.navigable === 'true';
        const color = navigation ? path.dataset.navigationColor : path.dataset.riverColor;
        if (color) path.setAttribute(path.classList.contains('river-channel') ? 'fill' : 'stroke', color);
        if (path.classList.contains('river-readable-line')) {
          path.setAttribute('stroke-width', '1.10');
          path.setAttribute('stroke-dasharray', strengths.some(value => value <= 0) ? '3 2' : 'none');
        }
      }
      for (const path of body.querySelectorAll('[data-route-importance]')) path.style.display = '';
      const themeImages = Array.from(body.querySelectorAll('[data-theme-map]'));
      const selectedImage = themeImages.find(node => node.dataset.themeMap === state.theme);
      for (const node of themeImages) if (node !== selectedImage) node.remove();
      if (selectedImage) {
        if (thematic && thematicMarkup) {
          const document = new DOMParser().parseFromString(thematicMarkup, 'image/svg+xml');
          if (document.querySelector('parsererror') || document.documentElement.localName !== 'svg') throw new Error('Invalid published overview SVG');
          const theme = document.documentElement;
          // The base owns the common shore. Theme assets contain paint only
          // and use that one authoritative clip after composition.
          if (theme.querySelector('defs')) throw new Error('Overview theme must use the common base definitions');
          for (const [name, value] of Object.entries({x:0, y:0, width, height, preserveAspectRatio:'none'})) theme.setAttribute(name, value);
          selectedImage.replaceWith(theme);
        } else selectedImage.remove();
      }
      svg.appendChild(body);
      return svg;
    }

    async function paintVector(markup, pixelWidth = width*rasterScale, pixelHeight = height*rasterScale) {
      const url = URL.createObjectURL(new Blob([markup], {type:'image/svg+xml'}));
      try {
        const decoded = new Image();
        decoded.src = url;
        await decoded.decode();
        const canvas = document.createElement('canvas');
        canvas.width = pixelWidth; canvas.height = pixelHeight;
        const context = canvas.getContext('2d', {alpha:true});
        context.drawImage(decoded, 0, 0, canvas.width, canvas.height);
        return canvas;
      } finally {
        URL.revokeObjectURL(url);
      }
    }

    async function rasterize(document) {
      const body = document.firstElementChild;
      const surface = body.querySelector('#physical-surface');
      const definitions = Array.from(surface?.children || []).filter(node => node.localName === 'defs');
      const nodes = Array.from(body.children).flatMap(node => node === surface
        ? Array.from(node.children).filter(child => child.localName !== 'defs') : [node]);
      const canvas = window.document.createElement('canvas');
      canvas.width = width*rasterScale; canvas.height = height*rasterScale;
      const context = canvas.getContext('2d', {alpha:true});
      const serializer = new XMLSerializer();
      const heavyLayers = new Set(['elevation-bands', 'bathymetry-bands', 'geographic-textures']);
      let light = [];
      const markup = content => {
        const svg = document.cloneNode(false);
        for (const definition of definitions) svg.appendChild(definition.cloneNode(true));
        const group = body.cloneNode(false);
        for (const node of content) group.appendChild(node.cloneNode(true));
        svg.appendChild(group);
        return serializer.serializeToString(svg);
      };
      const flush = async () => {
        if (!light.length) return;
        const painted = await paintVector(markup(light));
        context.drawImage(painted, 0, 0);
        light = [];
      };
      for (const node of nodes) {
        const layer = node.dataset.tileLayer;
        if (!heavyLayers.has(layer)) { light.push(node); continue; }
        await flush();
        if (node.style.display === 'none') continue;
        let bitmap = physicalPaint.get(layer);
        if (!bitmap) {
          const source = node.cloneNode(true);
          source.removeAttribute('opacity');
          const painted = await paintVector(markup([source]));
          bitmap = await createImageBitmap(painted);
          physicalPaint.set(layer, bitmap);
        }
        context.globalAlpha = Number(node.getAttribute('opacity') || 1);
        context.drawImage(bitmap, 0, 0);
        context.globalAlpha = 1;
      }
      await flush();
      const encoded = await new Promise((resolve, reject) => {
        canvas.toBlob(blob => blob ? resolve(blob) : reject(new Error('Atlas overview encoding failed')), 'image/png');
      });
      const uri = URL.createObjectURL(encoded);
      rasterizations++;
      return {uri, scene:document};
    }

    async function pump() {
      if (active || !queued) return;
      const job = queued;
      queued = null;
      active = {key:job.key, controller:new AbortController()};
      const running = active;
      try {
        await sourceDocument(running.controller.signal);
        const markup = await themeDocument(job.state.view === 'physical' ? job.state.theme : 'none', running.controller.signal);
        if (running.controller.signal.aborted || wanted?.key !== job.key) return;
        const entry = await rasterize(compose(job.state, markup));
        if (running.controller.signal.aborted) {
          URL.revokeObjectURL(entry.uri);
          return;
        }
        remember(job.key, entry);
        if (wanted?.key === job.key) show(job.key, entry);
      } catch (error) {
        if (!running.controller.signal.aborted && wanted?.key === job.key) loadError('世界轮廓暂未更新，已显示的地图仍可查看。');
      } finally {
        if (active === running) active = null;
        pump();
      }
    }

    return {
      update({theme, season, view, layerVisibility, scale, worldBounds, pixelRatio, refine}) {
        if (typeof theme !== 'string' || !['physical', 'monsoon'].includes(view)
            || !['vernal', 'june', 'autumnal', 'december'].includes(season)
            || !layerVisibility || typeof layerVisibility !== 'object'
            || !Number.isFinite(scale) || scale <= 0 || !Number.isFinite(pixelRatio) || pixelRatio <= 0
            || !Array.isArray(worldBounds) || worldBounds.length !== 4 || !worldBounds.every(Number.isFinite)
            || typeof refine !== 'boolean') throw new TypeError('Invalid atlas overview presentation');
        camera = {scale, pixelRatio, refine,
          bounds:[Math.max(0,worldBounds[0]),Math.max(0,worldBounds[1]),
            Math.min(width,worldBounds[2]),Math.min(height,worldBounds[3])]};
        if (source && theme !== 'none' && !themes.has(theme)) throw new TypeError('Atlas overview theme was not published');
        const state = {theme, season, view, layerVisibility:{...layerVisibility}};
        const key = JSON.stringify([view, theme, season, Object.entries(state.layerVisibility).sort(([first], [second]) => first.localeCompare(second))]);
        if (wanted?.key === key) {requestRefinement(); return;}
        wanted = {key, state};
        queued = null;
        if (active && active.key !== key) active.controller.abort();
        if (cache.has(key)) show(key, cache.get(key));
        else {queued = wanted; pump();}
      },
      clear() {
        wanted = null; queued = null; currentKey = null;
        camera = null; discardRefinement();
        active?.controller.abort();
        image.removeAttribute('href');
        for (const entry of cache.values()) URL.revokeObjectURL(entry.uri);
        cache.clear(); themeCache.clear();
        // An in-flight render owns these immutable paints until it finishes.
        // They are reusable after clear; the three native bitmaps stay bounded.
      },
      stats() {
        return {cachedImages:cache.size, cachedThemeDocuments:themeCache.size, pendingRasters:(active ? 1 : 0)+(refinementJob ? 1 : 0),
          queuedRasters:(queued ? 1 : 0)+(queuedRefinement ? 1 : 0), rasterizations, currentKey, wantedKey:wanted?.key || null,
          viewportRasterizations, refinementDensity:refined?.density || 0,
          refinementWidth:refined?.pixelWidth || 0, refinementHeight:refined?.pixelHeight || 0,
          viewportCovered:!!covers(refined,camera),
          cachedPhysicalPaints:physicalPaint.size,
          width, height, rasterWidth:width*rasterScale, rasterHeight:height*rasterScale};
      },
      sourceGeometry() { return source; },
    };
  }

  window.WorldAtlasOverview = {create};
})();
