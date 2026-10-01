/* Spherical measurements on the atlas's native 2:1 Plate Carree surface.
 * No camera subscriptions, input handlers, animation loop, or map assets.
 */
(function (root) {
  'use strict';
  const RAD = Math.PI / 180;
  const SVG_NS = 'http://www.w3.org/2000/svg';
  const dot = (a, b) => a.reduce((value, component, i) => value + component * b[i], 0);
  const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
  const norm = a => Math.hypot(...a);
  const normalize = a => { const length = norm(a); return a.map(value => value / length); };

  function geoPoint(point) {
    if (!Array.isArray(point) || point.length !== 2 || !point.every(Number.isFinite)) {
      throw new TypeError('A ruler location needs finite longitude and latitude');
    }
    if (Math.abs(point[1]) > 90) throw new RangeError('Latitude must be within -90..90');
    let longitude = point[0];
    if (longitude < -180 || longitude > 180) longitude = ((longitude + 180) % 360 + 360) % 360 - 180;
    if (longitude === -180 && point[0] > 0) longitude = 180;
    return [longitude, point[1]];
  }

  function vector(point) {
    const [longitude, latitude] = geoPoint(point).map(value => value * RAD);
    // All longitudes at either pole represent exactly the same location.
    return [Math.cos(latitude)*Math.cos(longitude), Math.cos(latitude)*Math.sin(longitude), Math.sin(latitude)]
      .map(value => Math.abs(value) < 1e-15 ? 0 : value);
  }

  function greatCircleDistanceKm(a, b, radiusKm) {
    if (!Number.isFinite(radiusKm) || radiusKm <= 0) throw new RangeError('Planet radius must be positive kilometres');
    const first = vector(a), last = vector(b);
    // Same atan2(|cross|, dot) formula as physical/planetary_grid.py:
    // accurate for tiny and antipodal distances, without a geometry library.
    return radiusKm * Math.atan2(norm(cross(first, last)), dot(first, last));
  }

  function greatCircleArc(a, b, {maxStepDegrees = 2} = {}) {
    if (!Number.isFinite(maxStepDegrees) || maxStepDegrees <= 0 || maxStepDegrees > 90) {
      throw new RangeError('Arc sample step must be within 0..90 degrees');
    }
    const first = geoPoint(a), last = geoPoint(b), u = vector(first), v = vector(last);
    const normal = cross(u, v), length = norm(normal);
    const angle = Math.atan2(length, dot(u, v));
    if (angle < 1e-14) return [first, last];
    let tangent;
    if (length > 1e-14) tangent = normalize(cross(normal.map(value => value / length), u));
    else {
      // Exact antipodes have no unique shortest arc. Choose a deterministic
      // plane; the same plane and route are retained in the reverse direction.
      const minimum = Math.min(...u.map(Math.abs));
      const axis = [0, 0, 0]; axis[u.findIndex(value => Math.abs(value) <= minimum + 1e-12)] = 1;
      tangent = normalize(cross(normalize(cross(u, axis)), u));
    }
    const count = Math.ceil(angle / (maxStepDegrees * RAD));
    const points = [first];
    for (let i = 1; i < count; i++) {
      const distance = angle * i / count;
      const point = u.map((value, j) => value*Math.cos(distance) + tangent[j]*Math.sin(distance));
      points.push([Math.atan2(point[1], point[0])/RAD, Math.atan2(point[2], Math.hypot(point[0], point[1]))/RAD]);
    }
    points.push(last);
    return points;
  }

  function seamLatitude(a, b) {
    const normal = cross(vector(a), vector(b));
    let intersection = [-normal[2], 0, normal[0]];
    if (intersection[0] > 0) intersection = intersection.map(value => -value);
    return Math.atan2(intersection[2], Math.abs(intersection[0])) / RAD;
  }

  function splitArc(points) {
    const parts = [], pole = point => Math.abs(point[1]) >= 90 - 1e-10;
    let current = [points[0]];
    for (let i = 1; i < points.length; i++) {
      const previous = points[i-1], point = points[i];
      if (pole(previous)) current = [[point[0], previous[1]]];
      if (pole(point)) {
        current.push([previous[0], point[1]]);
        if (current.length > 1) parts.push(current);
        current = [point];
        continue;
      }
      const delta = point[0] - previous[0];
      // A sampled meridian can straddle a pole without sampling its apex.
      // The two longitude columns meet at the same physical pole; they must
      // not be joined by a diagonal across the expanded map.
      if (!pole(previous) && Math.abs(Math.abs(delta)-180) < 1e-10 && previous[1]*point[1] > 0) {
        const latitude = point[1] > 0 ? 90 : -90;
        current.push([previous[0], latitude]);
        if (current.length > 1) parts.push(current);
        current = [[point[0], latitude]];
      }
      else if (!pole(previous) && Math.abs(delta) > 180) {
        const edge = delta < 0 ? 180 : -180, latitude = seamLatitude(previous, point);
        current.push([edge, latitude]);
        if (current.length > 1) parts.push(current);
        current = [[-edge, latitude]];
      }
      current.push(point);
    }
    if (current.length > 1) parts.push(current);
    return parts;
  }

  function create({svgOverlay, mapNativeToGeo, worldWidth, radiusKm, onChange = () => {}}) {
    if (!svgOverlay || svgOverlay.namespaceURI !== SVG_NS || typeof svgOverlay.appendChild !== 'function') {
      throw new TypeError('Ruler overlay must be a native SVG container');
    }
    if (typeof mapNativeToGeo !== 'function' || typeof onChange !== 'function') throw new TypeError('Ruler callbacks must be functions');
    if (!Number.isFinite(worldWidth) || worldWidth <= 0 || !Number.isFinite(radiusKm) || radiusKm <= 0) {
      throw new RangeError('Ruler needs positive native world width and planet radius');
    }
    const document = svgOverlay.ownerDocument, worldHeight = worldWidth / 2;
    const element = (name, attributes = {}) => {
      const node = document.createElementNS(SVG_NS, name);
      for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
      return node;
    };
    const group = element('g', {'data-atlas-ruler': '', 'pointer-events':'none', role:'img'});
    svgOverlay.appendChild(group);
    let active = false, destroyed = false;
    const stations = [], segments = [];
    const total = () => segments.reduce((sum, segment) => sum + segment.distanceKm, 0);
    const snapshot = () => ({active, radiusKm, stations:stations.map(point => ({native:[...point.native], geo:[...point.geo]})),
      segments:segments.map(segment => ({...segment})), totalKm:total()});
    const coordinate = geo => `${geo[1].toFixed(6)}°, ${geo[0].toFixed(6)}°`;
    const native = ([longitude, latitude]) => [(longitude + 180)/360*worldWidth, (90 - latitude)/180*worldHeight];
    const title = (node, text) => { const label = element('title'); label.textContent = text; node.appendChild(label); };
    const path = (data, color, width, part) => element('path', {
      d:data, fill:'none', stroke:color, 'stroke-width':width,
      'stroke-linecap':'round', 'stroke-linejoin':'round', 'vector-effect':'non-scaling-stroke', 'data-ruler-part':part,
    });

    function render() {
      const fragment = document.createDocumentFragment();
      segments.forEach((segment, i) => {
        const parts = segment.distanceKm === 0 ? [] : splitArc(greatCircleArc(stations[i].geo, stations[i+1].geo));
        const data = parts.map(part => part.map((point, index) => `${index ? 'L' : 'M'}${native(point).join(' ')}`).join(' ')).join(' ');
        const route = element('g', {'data-ruler-segment':i, 'aria-label':`第 ${i+1} 段 ${segment.distanceKm.toFixed(3)} km`});
        title(route, `第 ${i+1} 段：${segment.distanceKm.toFixed(3)} km`);
        if (data) route.append(path(data, '#ffffff', 5, 'arc'), path(data, '#176bd1', 2.5, 'arc'));
        fragment.appendChild(route);
      });
      stations.forEach((station, i) => {
        const point = element('g', {'data-ruler-station':i, 'aria-label':`站点 ${i+1} · ${coordinate(station.geo)}`});
        title(point, `站点 ${i+1}：${coordinate(station.geo)}`);
        // SVG round caps render a zero-length path as a fixed screen circle;
        // its radius stays constant even when the parent camera is scaled.
        const data = `M${station.native.join(' ')}h0`;
        point.append(path(data, '#ffffff', 9, 'station'), path(data, '#176bd1', 5, 'station'));
        fragment.appendChild(point);
      });
      group.replaceChildren(fragment);
      group.setAttribute('aria-label', `球面测距 · ${stations.length} 个站点 · 累计 ${total().toFixed(3)} km`);
    }

    function changed() { render(); onChange(snapshot()); }
    render();
    return {
      get active() { return active; },
      state:snapshot,
      start() { if (destroyed || active) return false; active = true; changed(); return true; },
      finish() { if (destroyed || !active) return false; active = false; changed(); return true; },
      addNativePoint(point) {
        if (destroyed || !active || !Array.isArray(point) || point.length !== 2 || !point.every(Number.isFinite)
            || point[0] < 0 || point[0] > worldWidth || point[1] < 0 || point[1] > worldHeight) return false;
        let geo;
        try { geo = geoPoint(mapNativeToGeo([...point])); } catch (error) {
          if (error instanceof TypeError || error instanceof RangeError) return false;
          throw error;
        }
        if (stations.length) segments.push({from:stations.length-1, to:stations.length,
          distanceKm:greatCircleDistanceKm(stations.at(-1).geo, geo, radiusKm)});
        stations.push({native:[...point], geo}); changed(); return true;
      },
      undo() { if (destroyed || !stations.length) return false; stations.pop(); if (segments.length) segments.pop(); changed(); return true; },
      clear() { if (destroyed || !stations.length) return false; stations.length = 0; segments.length = 0; changed(); return true; },
      destroy() { if (destroyed) return; destroyed = true; active = false; stations.length = 0; segments.length = 0; group.remove(); },
    };
  }

  const api = {greatCircleDistanceKm, greatCircleArc, splitArc, create};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.WorldAtlasRuler = api;
})(typeof window !== 'undefined' ? window : globalThis);
