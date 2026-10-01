import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// Rasterize the atlas's own vector surfaces, never a separately generated planet.
const mount = document.getElementById('globe');
const status = document.getElementById('globe-status');
const payload = window.WorldAtlasGlobe;
try {
  const renderer = new THREE.WebGLRenderer({antialias: true, alpha: true, powerPreference: 'low-power'});
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  mount.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 100);
  camera.position.set(0, 0.25, 3.4);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enablePan = false;
  controls.minDistance = 1.08;
  controls.maxDistance = 5.5;
  controls.enableDamping = false;
  const textureWidth = Math.min(8192, renderer.capabilities.maxTextureSize);
  const baseSurface = new DOMParser().parseFromString(payload.surface, 'image/svg+xml').documentElement;
  const viewBox = baseSurface.getAttribute('viewBox').replaceAll(',', ' ').trim().split(/\s+/);
  const commonInk = new DOMParser().parseFromString(payload.ink, 'image/svg+xml').documentElement;
  commonInk.setAttribute('width', viewBox[2]);
  commonInk.setAttribute('height', viewBox[3]);
  const themeSelect = document.getElementById('globe-theme');
  const requestedTheme = new URLSearchParams(location.search).get('theme');
  let selection = 0;
  let selectedTheme = Object.hasOwn(payload.textures, requestedTheme) ? requestedTheme : 'terrain';
  themeSelect.value = selectedTheme;
  let appliedTheme = null;
  let disposed = false;
  let themeController = null;
  const themeSources = new Map();
  async function themeSource(key, signal) {
    const url = payload.textures[key];
    if (themeSources.has(key)) {
      const source = themeSources.get(key);
      themeSources.delete(key); themeSources.set(key, source);
      return source;
    }
    const response = await fetch(url, {signal, cache: 'force-cache'});
    if (!response.ok) throw new Error(`球体专题加载失败：HTTP ${response.status}`);
    const source = await response.text();
    if (signal.aborted) throw new DOMException('Superseded globe theme', 'AbortError');
    themeSources.set(key, source);
    while (themeSources.size > 4) themeSources.delete(themeSources.keys().next().value);
    return source;
  }
  const material = new THREE.MeshBasicMaterial();
  const planet = new THREE.Mesh(new THREE.SphereGeometry(1, 256, 128), material);
  scene.add(planet);
  const gridPoints = [];
  const globePoint = (longitude, latitude) => {
    const lat = latitude * Math.PI / 180, lon = longitude * Math.PI / 180;
    return new THREE.Vector3(Math.cos(lat) * Math.cos(lon), Math.sin(lat), -Math.cos(lat) * Math.sin(lon)).multiplyScalar(1.001);
  };
  for (let latitude = -60; latitude <= 60; latitude += 30) {
    for (let longitude = -180; longitude < 180; longitude += 2) gridPoints.push(globePoint(longitude, latitude), globePoint(longitude + 2, latitude));
  }
  for (let longitude = -180; longitude < 180; longitude += 30) {
    for (let latitude = -90; latitude < 90; latitude += 2) gridPoints.push(globePoint(longitude, latitude), globePoint(longitude, latitude + 2));
  }
  const graticule = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(gridPoints), new THREE.LineBasicMaterial({color: 0x34495a, transparent: true, opacity: 0.30}));
  graticule.visible = false;
  scene.add(graticule);
  let queued = false;
  let frames = 0;
  function draw() {
    if (queued || document.hidden || disposed) return;
    queued = true;
    requestAnimationFrame(() => {
      queued = false;
      if (document.hidden || disposed) return;
      renderer.render(scene, camera);
      frames++;
    });
  }
  function resize() {
    const {width, height} = mount.getBoundingClientRect();
    camera.aspect = width / Math.max(1, height);
    camera.updateProjectionMatrix();
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setSize(width, height, false);
    draw();
  }
  async function selectTheme(key) {
    if (!Object.hasOwn(payload.textures, key)) throw new Error('未知的球体专题');
    selectedTheme = key;
    const currentSelection = ++selection;
    themeController?.abort();
    if (appliedTheme === key) {
      status.textContent = '与平面地图共用同源矢量地形 · 拖动旋转 / 滚轮缩放';
      return;
    }
    const controller = new AbortController();
    themeController = controller;
    status.textContent = '正在载入清晰球体…';
    let sourceMarkup;
    try {
      sourceMarkup = await themeSource(key, controller.signal);
    } catch (error) {
      if (selection !== currentSelection || disposed || controller.signal.aborted) return;
      throw error;
    }
    if (selection !== currentSelection || disposed) return;
    const svg = baseSurface.cloneNode(true);
    const overlayDocument = new DOMParser().parseFromString(sourceMarkup, 'image/svg+xml');
    if (overlayDocument.querySelector('parsererror') || overlayDocument.documentElement.localName !== 'svg') throw new Error('球体专题 SVG 无效');
    const overlay = overlayDocument.documentElement;
    overlay.setAttribute('width', viewBox[2]);
    overlay.setAttribute('height', viewBox[3]);
    svg.appendChild(overlay);
    // Fine coasts and common linework are stored once, then painted above
    // each selected thematic fill in the same single vector rasterization.
    svg.appendChild(commonInk.cloneNode(true));
    svg.setAttribute('width', textureWidth);
    svg.setAttribute('height', textureWidth / 2);
    // Explicit intrinsic dimensions make the browser rasterize vector edges at
    // texture resolution instead of enlarging the original simulation pixels.
    const source = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(new XMLSerializer().serializeToString(svg));
    let image;
    try {
      image = await new THREE.ImageLoader().loadAsync(source);
    } catch (error) {
      if (selection !== currentSelection || disposed) return;
      throw error;
    }
    // A slow previous selection must never overwrite the user's latest choice.
    if (selection !== currentSelection || disposed) return;
    const texture = new THREE.Texture(image);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.anisotropy = renderer.capabilities.getMaxAnisotropy();
    texture.wrapS = THREE.RepeatWrapping;
    texture.needsUpdate = true;
    // A full-resolution texture is large; keep only the visible theme on the GPU.
    material.map?.dispose();
    material.map = texture;
    appliedTheme = key;
    material.needsUpdate = true;
    status.textContent = '与平面地图共用同源矢量地形 · 拖动旋转 / 滚轮缩放';
    draw();
  }
  controls.addEventListener('change', draw);
  const observer = new ResizeObserver(resize);
  observer.observe(mount);
  document.addEventListener('visibilitychange', draw);
  themeSelect.addEventListener('change', event => selectTheme(event.target.value).catch(error => { status.textContent = `纹理加载失败：${error.message}`; }));
  document.getElementById('globe-graticule').addEventListener('change', event => { graticule.visible = event.target.checked; draw(); });
  const coordinates = document.getElementById('globe-coordinates');
  const raycaster = new THREE.Raycaster();
  let pointerStart = null;
  renderer.domElement.addEventListener('pointerdown', event => { pointerStart = {x: event.clientX, y: event.clientY}; });
  renderer.domElement.addEventListener('pointerup', event => {
    if (!pointerStart || Math.hypot(event.clientX - pointerStart.x, event.clientY - pointerStart.y) > 5) return;
    const bounds = renderer.domElement.getBoundingClientRect();
    raycaster.setFromCamera(new THREE.Vector2((event.clientX - bounds.left) / bounds.width * 2 - 1, 1 - (event.clientY - bounds.top) / bounds.height * 2), camera);
    const hit = raycaster.intersectObject(planet)[0];
    if (hit) {
      const latitude = hit.uv.y * 180 - 90, longitude = hit.uv.x * 360 - 180;
      coordinates.textContent = `${Math.abs(latitude).toFixed(2)}° ${latitude >= 0 ? 'N' : 'S'} · ${Math.abs(longitude).toFixed(2)}° ${longitude >= 0 ? 'E' : 'W'}`;
    }
  });
  document.getElementById('globe-reset').addEventListener('click', () => {
    camera.position.set(0, 0.25, 3.4);
    controls.target.set(0, 0, 0);
    controls.update();
    draw();
  });
  window.WorldAtlasGlobeDiagnostics = () => ({frames, textures: renderer.info.memory.textures, cachedThemeSources: themeSources.size, theme: selectedTheme, appliedTheme, textureWidth, textureHeight: textureWidth / 2, pixelRatio: renderer.getPixelRatio(), anisotropy: material.map?.anisotropy || 0, graticule: graticule.visible, vertices: planet.geometry.attributes.position.count, calls: renderer.info.render.calls});
  window.addEventListener('pagehide', event => {
    if (event.persisted) return;
    disposed = true;
    themeController?.abort(); themeSources.clear();
    observer.disconnect(); controls.dispose();
    material.map?.dispose();
    planet.geometry.dispose(); material.dispose();
    graticule.geometry.dispose(); graticule.material.dispose(); renderer.dispose();
  });
  resize();
  selectTheme(selectedTheme).catch(error => { status.textContent = `纹理加载失败：${error.message}`; });
} catch (error) {
  status.textContent = `此设备无法启用 WebGL 球体：${error.message}。平面地图仍可使用。`;
}
