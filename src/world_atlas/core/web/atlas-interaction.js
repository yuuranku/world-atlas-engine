/* Map gestures. The camera and terrain renderer remain owned by the atlas. */
(function (root) {
  'use strict';

  function wheelPixels(event, pageHeight) {
    const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? pageHeight : 1;
    return Math.max(-240, Math.min(240, event.deltaY * unit));
  }

  function attach({viewport, getCamera, setCamera, zoomAt, zoomBy, reset,
                   onTap, onCommit, onGestureStart = () => {},
                   onDoubleClick = () => false, onEscape = () => false}) {
    const listeners = [], pointers = new Map();
    let frame = 0, wheelExponent = 0, wheelX = 0, wheelY = 0, wheelCommit = 0;
    let gestureMoved = false, gestureMultiple = false, tapTarget = null;
    const listen = (target, name, callback, options) => {
      target.addEventListener(name, callback, options);
      listeners.push(() => target.removeEventListener(name, callback, options));
    };
    const editable = target => target.closest('button,a,input,textarea,select,[contenteditable="true"]');
    const gesture = () => {
      const points = Array.from(pointers.values()).slice(0, 2);
      if (points.length === 1) return {...points[0], distance:0};
      return {x:(points[0].x+points[1].x)/2, y:(points[0].y+points[1].y)/2,
        distance:Math.hypot(points[0].x-points[1].x, points[0].y-points[1].y)};
    };

    listen(viewport, 'wheel', event => {
      if (editable(event.target)) return;
      event.preventDefault();
      onGestureStart();
      wheelExponent -= wheelPixels(event, viewport.clientHeight) * (event.ctrlKey ? .01 : .0025);
      wheelX = event.clientX; wheelY = event.clientY;
      if (!frame) frame = root.requestAnimationFrame(() => {
        frame = 0;
        const factor = Math.exp(Math.max(-Math.log(4), Math.min(Math.log(4), wheelExponent)));
        wheelExponent = 0;
        zoomAt(factor, wheelX, wheelY);
      });
      root.clearTimeout(wheelCommit);
      wheelCommit = root.setTimeout(onCommit, 140);
    }, {passive:false});

    listen(viewport, 'pointerdown', event => {
      if (![0,1].includes(event.button) || editable(event.target)) return;
      event.preventDefault();
      root.getSelection()?.removeAllRanges();
      if (!pointers.size) {
        gestureMoved = false; gestureMultiple = false;
        tapTarget = event.button === 0 ? event.target : null;
      }
      pointers.set(event.pointerId, {x:event.clientX, y:event.clientY,
        startX:event.clientX, startY:event.clientY});
      if (pointers.size > 1) gestureMultiple = true;
      onGestureStart();
      viewport.focus({preventScroll:true});
      viewport.classList.add('dragging');
      viewport.setPointerCapture(event.pointerId);
    });

    listen(viewport, 'pointermove', event => {
      const previous = pointers.get(event.pointerId);
      if (!previous) return;
      const before = gesture();
      pointers.set(event.pointerId, {...previous, x:event.clientX, y:event.clientY});
      const after = gesture();
      if (Math.hypot(event.clientX-previous.startX, event.clientY-previous.startY) > 5) gestureMoved = true;
      const camera = getCamera(), bounds = viewport.getBoundingClientRect();
      const ratio = before.distance > 0 && after.distance > 0 ? after.distance/before.distance : 1;
      const nextScale = Math.max(camera.minimumScale, Math.min(camera.maximumScale, camera.scale*ratio));
      // Both pinch scaling and centroid panning preserve the same world point.
      const x = (before.x-bounds.left-camera.translateX)/camera.scale;
      const y = (before.y-bounds.top-camera.translateY)/camera.scale;
      setCamera({scale:nextScale,
        translateX:after.x-bounds.left-x*nextScale,
        translateY:after.y-bounds.top-y*nextScale});
    });

    const stop = event => {
      if (!pointers.has(event.pointerId)) return;
      pointers.delete(event.pointerId);
      if (pointers.size) return;
      viewport.classList.remove('dragging');
      onCommit();
      if (event.type === 'pointerup' && !gestureMoved && !gestureMultiple && tapTarget) {
        onTap({clientX:event.clientX, clientY:event.clientY, target:tapTarget});
      }
      tapTarget = null;
    };
    for (const name of ['pointerup','pointercancel','lostpointercapture']) listen(viewport, name, stop);
    listen(viewport, 'selectstart', event => { if (!editable(event.target)) event.preventDefault(); });
    listen(viewport, 'dragstart', event => event.preventDefault());
    listen(viewport, 'dblclick', event => {
      event.preventDefault();
      if (!onDoubleClick(event)) {
        zoomAt(event.shiftKey ? .5 : 2, event.clientX, event.clientY);
        onCommit();
      }
    });
    listen(viewport, 'keydown', event => {
      if (editable(event.target)) return;
      if (event.key === 'Escape' && onEscape()) {event.preventDefault(); return;}
      if (['+','='].includes(event.key)) {event.preventDefault(); zoomBy(2);}
      else if (['-','_'].includes(event.key)) {event.preventDefault(); zoomBy(.5);}
      else if (event.key === '0') {event.preventDefault(); reset();}
      else if (['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key)) {
        event.preventDefault(); onGestureStart();
        const camera = getCamera(), distance = event.shiftKey ? 180 : 80;
        setCamera({scale:camera.scale,
          translateX:camera.translateX+({ArrowLeft:distance,ArrowRight:-distance}[event.key] || 0),
          translateY:camera.translateY+({ArrowUp:distance,ArrowDown:-distance}[event.key] || 0)});
        onCommit();
      }
    });
    return {get dragging() {return pointers.size > 0;}, destroy() {
      for (const remove of listeners) remove();
      if (frame) root.cancelAnimationFrame(frame);
      root.clearTimeout(wheelCommit); pointers.clear();
      viewport.classList.remove('dragging');
    }};
  }

  const api = {wheelPixels, attach};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.WorldAtlasInteraction = api;
})(typeof window !== 'undefined' ? window : globalThis);
