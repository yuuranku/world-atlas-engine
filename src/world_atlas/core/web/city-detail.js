/* Pure city planning and detached SVG drawing for World Atlas recipes. */
(function (global) {
  "use strict";

  const SVG_NS = "http://www.w3.org/2000/svg";
  const VALID_ERAS = new Set(["tribal", "ancient", "medieval", "early-modern", "preindustrial", "industrial", "contemporary"]);

  /* Return detached geometry for the selected city on the atlas. */
  function render(recipe, { grid, clipPrefix = "city-detail" } = {}) {
    return cityNode(recipe, normalizeGrid(grid), clipPrefix);
  }

  /* Pure planner; pass options.grid to unwrap canonical world cells at the longitude seam. */
  function plan(recipe, options = {}) {
    validateRecipe(recipe);
    const grid = normalizeGrid(options.grid);
    if (grid) recipe = recipeInLocalLongitude(recipe, grid.width);
    const urban = recipe.urban || {};
    const morphology = recipe.morphology || {};
    const terrain = recipe.terrain || {};
    const water = recipe.water || {};
    const center = recipe.location;
    const radiusRows = Math.max(0.004, finiteOr(urban.radiusRows, 0.05));
    const radiusColumns = Math.max(0.004, finiteOr(urban.radiusColumns, 0.05));
    const seed = unsigned(recipe.seed);
    const bearing = Number.isFinite(water.waterBearingDegrees) ? normalizeAngle(water.waterBearingDegrees) : null;
    const request = {
      recipe,
      center,
      radiusRows,
      radiusColumns,
      seed,
      bearing,
      morphology: morphology.kind || "plain-organic",
      streetPattern: morphology.streetPattern || "organic",
      slopeClass: terrain.slopeClass || "flat",
      orientation: finiteOr(morphology.orientationDegrees, 0),
      buildingPattern: morphology.buildingPattern || "perimeter-blocks",
      blockAspect: finiteOr(morphology.blockAspect, 1),
      streetIrregularity: finiteOr(morphology.streetIrregularity, 0.35),
      population: Math.max(0, finiteOr(recipe.population && recipe.population.estimate, 0)),
      era: recipe.era,
      culturalStyle: (recipe.culture || {}).style || "vernacular-mixed",
      tier: recipe.tier || "site",
      areaKm2: Math.max(0, finiteOr(urban.targetFootprintKm2, 0)),
      character: global.WorldAtlasCityCharacter.profile(recipe),
    };
    const fabric = urbanFabric(request);
    const buildable = normalizeBuildable(terrain.buildable);
    const cores = Array.isArray(urban.coreZones) && urban.coreZones.length
      ? urban.coreZones.map((zone) => ({
          id: String(zone.id),
          role: String(zone.role || "central-core"),
          row: finiteOr(zone.row, center.row),
          column: finiteOr(zone.column, center.column),
          radiusKm: Math.max(0.1, finiteOr(zone.radiusKm, 0.25)),
        }))
      : [{ id: recipe.id + "-core-1", role: "central-core", row: center.row, column: center.column, radiusKm: 0.25 }];
    return {
      era: recipe.era,
      footprint: fabric.footprint,
      blocks: fabric.blocks,
      districts: fabric.districts,
      streets: fabric.streets,
      buildings: fabric.buildings,
      walls: fabric.walls,
      innerWalls: fabric.innerWalls,
      towers: fabric.towers,
      plazas: fabric.plazas,
      landmarks: fabric.landmarks,
      countryside: fabric.countryside,
      farmland: fabric.farmland,
      fortress: fabric.fortress,
      gates: fabric.gates,
      barbicans: fabric.barbicans,
      moats:fabric.moats,
      countryLanes: fabric.countryLanes,
      hamlets: fabric.hamlets,
      character: request.character,
      annotations: featureAnnotations(recipe,fabric,request.character).concat(recipe.localSite?.annotations||[]),
      streetWidths: streetWidthsInGrid(recipe, radiusRows, radiusColumns),
      footprintRadiusRows: radiusRows,
      footprintRadiusColumns: radiusColumns,
      gridCellKilometres: gridCellKilometres(recipe, radiusRows, radiusColumns),
      cores,
      buildable,
    };
  }

  function recipeInLocalLongitude(recipe, width) {
    const centre = recipe.location.column;
    const columnNearCity = column => column + Math.round((centre - column) / width) * width;
    const cellsNearCity = cells => cells.map(([row, column]) => [row, columnNearCity(column + 0.5) - 0.5]);
    const buildable = recipe.terrain.buildable;
    return {
      ...recipe,
      urban: {...recipe.urban, coreZones: (recipe.urban.coreZones || []).map(core => ({...core, column: columnNearCity(core.column)}))},
      terrain: {...recipe.terrain, buildable: {...buildable,
        allowedCells: cellsNearCity(buildable.allowedCells),
        blockedCells: cellsNearCity(buildable.blockedCells || []),
      }},
      transport: {...recipe.transport, corridors: (recipe.transport?.corridors || []).map(corridor => ({...corridor,
        points: corridor.points.map(point => ({...point, column: columnNearCity(point.column)})),
      }))},
      localSite: recipe.localSite ? {...recipe.localSite,
        bounds: recipe.localSite.bounds ? {...recipe.localSite.bounds, west: columnNearCity(recipe.localSite.bounds.west), east: columnNearCity(recipe.localSite.bounds.east)} : undefined,
        surface: recipe.localSite.surface ? {...recipe.localSite.surface, bounds: {...recipe.localSite.surface.bounds,
          west: columnNearCity(recipe.localSite.surface.bounds.west), east: columnNearCity(recipe.localSite.surface.bounds.east)}} : undefined,
        channels: (recipe.localSite.channels || []).map(channel => ({...channel, points: channel.points.map(point => ({...point, column: columnNearCity(point.column)}))})),
        farmland: (recipe.localSite.farmland || []).map(field => ({...field, points: field.points.map(point => ({...point, column: columnNearCity(point.column)})),
          lines: (field.lines || []).map(line => line.map(point => ({...point, column: columnNearCity(point.column)})))})),
      } : undefined,
    };
  }

  function cityNode(recipe, grid, clipPrefix) {
    const city = svg("g", {
      "data-city-detail-id": recipe.id,
      "data-city-era": recipe.era,
      "data-city-morphology": (recipe.morphology && recipe.morphology.kind) || "plain-organic",
      class: "city-detail city-detail-" + (recipe.tier || "site"),
      "pointer-events": "none",
    });
    const cityPlan = plan(recipe, {grid});
    const clipId = clipIdentifier(recipe, clipPrefix);
    const definitions = svg("defs", {});
    const clip = svg("clipPath", { id: clipId, clipPathUnits: "userSpaceOnUse" });
    if(recipe.localSite){
      const b=recipe.localSite.bounds;
      clip.appendChild(svg('rect',{x:b.west,y:b.north,width:b.east-b.west,height:b.south-b.north}));
    }else cityPlan.buildable.allowedCells.forEach((cell) => {
      clip.appendChild(svg("rect", {
        x: cell[1], y: cell[0], width: 1, height: 1,
      }));
    });
    definitions.appendChild(clip);
    const urbanClipId = clipId + "-urban";
    const urbanClip = svg("clipPath", { id: urbanClipId, clipPathUnits: "userSpaceOnUse" });
    cityPlan.districts.forEach((points) => urbanClip.appendChild(svg("path", { d: pathData(points, true) })));
    definitions.appendChild(urbanClip);
    city.appendChild(definitions);
    const landGeometry = svg("g", {
      id: clipId + "-geometry",
      class: "city-land-geometry",
      "clip-path": "url(#" + clipId + ")",
    });
    const geometry = svg("g", {
      class: "city-geometry",
      "clip-path": "url(#" + urbanClipId + ")",
    });
    appendCountryside(landGeometry, cityPlan);
    if(recipe.localSite)for(const line of recipe.localSite.contours)landGeometry.appendChild(svg('path',{d:line.d,fill:'none',stroke:'#7c7b61',
      'stroke-opacity':line.major?.47:.28,'stroke-width':line.major?.9:.45,'vector-effect':'non-scaling-stroke',class:'city-land-contours'}));
    appendMoats(landGeometry,cityPlan);
    geometry.appendChild(svg('path',{class:'city-street-ground',d:cityPlan.districts.map(points=>pathData(points,true)).join(' '),fill:'#e9e4d5','fill-opacity':.64}));
    const blocks = svg("g", { class: "city-blocks" });
    cityPlan.blocks.forEach((block) => blocks.appendChild(svg("path", {
      d: pathData(block.points, true),
      class: "city-block city-block-" + block.kind,
      fill: block.fill,
      "fill-opacity": block.opacity,
      stroke: "none",
    })));
    geometry.appendChild(blocks);
    geometry.appendChild(svg("path", {class: "city-plazas",
      d: cityPlan.plazas.map(plaza => pathData(plaza.points, true)).join(" "),
      fill: "#e7ddc4", stroke: "#a39479", "stroke-width": 0.45,
      "vector-effect": "non-scaling-stroke"}));
    const buildings = svg("g", { class: "city-buildings" });
    geometry.appendChild(svg('path',{class:'city-building-shadows',d:cityPlan.buildings.map(building=>pathData(building.points,true)).join(' '),
      fill:'#3d4441',opacity:.20,transform:`translate(${.0014/cityPlan.gridCellKilometres} ${.0019/cityPlan.gridCellKilometres})`}));
    for (const key of new Set(cityPlan.buildings.map(building => building.kind + ":" + building.fill))) {
      const members = cityPlan.buildings.filter(building => building.kind + ":" + building.fill === key);
      const kind = members[0].kind;
      buildings.appendChild(svg("path", {
        d: members.map(building => pathData(building.points, true) + (building.courtyard ? " " + pathData(building.courtyard, true) : "")).join(" "),
        class: "city-building city-building-" + kind,
        "data-building-count": members.length,
        fill: members[0].fill, stroke: "#596053", "stroke-width": .00045 / cityPlan.gridCellKilometres,
        "stroke-opacity": .72, "stroke-linejoin": "round", "fill-rule": "evenodd",
      }));
    }
    geometry.appendChild(buildings);
    appendRoofPaint(geometry,cityPlan.buildings,'city-roof-faces',cityPlan.gridCellKilometres);
    geometry.appendChild(svg("path", {class: "city-roof-ridges",
      d: cityPlan.buildings.flatMap(building => building.roofLines || []).map(points => pathData(points, false)).join(" "),
      fill: "none", stroke: "#eee7cf", "stroke-width": .00028 / cityPlan.gridCellKilometres,
      "stroke-linecap": "round", opacity: 0.48}));
    const roadEdges = svg("g", {class: "city-street-edges", fill: "none", stroke: "#c7bca7",
      "stroke-linecap": "round", "stroke-linejoin": "round", opacity: 0.75});
    const minorRoads = svg("g", {
      class: "city-streets city-streets-minor",
      fill: "none",
      stroke: "#fff9ef",
      "stroke-width": cityPlan.streetWidths.local,
      "stroke-linecap": "round",
      "stroke-linejoin": "round",
    });
    const majorRoads = svg("g", {
      class: "city-streets city-streets-major",
      fill: "none",
      stroke: "#fffdf7",
      "stroke-width": cityPlan.streetWidths.arterial,
      "stroke-linecap": "round",
      "stroke-linejoin": "round",
    });
    cityPlan.streets.forEach((street) => {
      if (street.kind === "rail") return;
      const width = cityPlan.streetWidths[street.kind] || cityPlan.streetWidths.local;
      roadEdges.appendChild(svg("path", {
        d: pathData(street.points, false), "stroke-width": width * 1.2,
      }));
      (street.kind === "arterial" ? majorRoads : minorRoads).appendChild(svg("path", {
        d: pathData(street.points, false),
        "stroke-width": street.kind === "collector" ? cityPlan.streetWidths.collector : undefined,
      }));
    });
    geometry.appendChild(roadEdges);
    geometry.appendChild(minorRoads);
    geometry.appendChild(majorRoads);
    appendLandmarks(geometry, cityPlan);
    if (cityPlan.fortress) appendFortress(geometry, cityPlan);
    const rail = cityPlan.streets.filter(street => street.kind === "rail").map(street => pathData(street.points, false)).join(" ");
    if (rail) {
      const width = cityPlan.streetWidths.arterial * 0.65;
      geometry.appendChild(svg("path", {d: rail, fill: "none", stroke: "#3b3c38", "stroke-width": width}));
      geometry.appendChild(svg("path", {d: rail, fill: "none", stroke: "#fffdf4", "stroke-width": width * 0.55, "stroke-dasharray": `${width * 3} ${width * 2}`}));
    }
    landGeometry.appendChild(geometry);
    appendWalls(landGeometry, cityPlan);
    appendLabels(landGeometry, cityPlan);
    city.appendChild(landGeometry);
    // Plan once in continuous local longitude, then reuse the clipped geometry
    // at the opposite edge of the rectangular atlas.
    if (grid) {
      const bounds = recipe.urban.bounds;
      if (bounds.west < 0) city.appendChild(svg("use", {href: "#" + clipId + "-geometry", transform: `translate(${grid.width} 0)`}));
      if (bounds.east > grid.width) city.appendChild(svg("use", {href: "#" + clipId + "-geometry", transform: `translate(${-grid.width} 0)`}));
    }
    return city;
  }

  function appendLandmarks(city, cityPlan) {
    const group = svg("g", {class: "city-landmarks"});
    group.appendChild(svg('path',{class:'city-landmark-shadows',
      d:cityPlan.landmarks.map(landmark=>pathData(landmark.points,true)).join(' '),
      fill:'#38473f',opacity:.23,transform:`translate(${.003/cityPlan.gridCellKilometres} ${.004/cityPlan.gridCellKilometres})`}));
    for (const landmark of cityPlan.landmarks) group.appendChild(svg("path", {
      d: pathData(landmark.points, true), class: "city-landmark city-landmark-" + landmark.kind,
      fill: landmark.fill, stroke: "#736750", "stroke-width": .0007 / cityPlan.gridCellKilometres,
      "stroke-linejoin": "round",
    }));
    group.appendChild(svg("path", {d: cityPlan.landmarks.flatMap(landmark => landmark.roofLines || [])
      .map(points => pathData(points, false)).join(" "), fill: "none", stroke: "#f8e3b9",
      "stroke-width": .0004 / cityPlan.gridCellKilometres}));
    appendRoofPaint(group,cityPlan.landmarks,'city-landmark-faces',cityPlan.gridCellKilometres);
    city.appendChild(group);
  }

  function appendRoofPaint(parent,buildings,className,metric){
    const dark=[],light=[],hipLines=[],flat=[];
    for(const building of buildings){
      const polygon=building.points.map(p=>({x:p.column,y:p.row})),c=polygonCentre(polygon);
      if(building.roof==='flat'){const inside=insetPolygon(polygon,Math.sqrt(polygonArea(polygon))*.10);
        flat.push(pathData(inside.map(p=>({column:p.x,row:p.y})),true));continue;}
      if(building.courtyard){
        const inner=building.courtyard;
        for(let i=0;i<building.points.length;i++){
          const a=building.points[i],b=building.points[(i+1)%building.points.length],
            near=p=>inner.reduce((best,q)=>Math.hypot(p.column-q.column,p.row-q.row)<Math.hypot(p.column-best.column,p.row-best.row)?q:best,inner[0]);
          const ring=[a,b,near(b),near(a)],value=(b.column-a.column)-(b.row-a.row);
          (value>0?light:dark).push(pathData(ring,true));
        }continue;
      }
      const ridge=building.roofLines?.find(line=>line.length===2);
      if(!ridge){
        for(let i=0;i<polygon.length;i++){
          const a=polygon[i],b=polygon[(i+1)%polygon.length],value=b.x-a.x-(b.y-a.y);
          (value>0?light:dark).push(pathData([a,b,c].map(p=>({column:p.x,row:p.y})),true));
        }continue;
      }
      const [a,b]=ridge,nx=b.row-a.row,ny=a.column-b.column,d=nx*a.column+ny*a.row;
      for(const sign of [-1,1]){
        const face=clipHalfPlane(polygon,nx*sign,ny*sign,d*sign);
        if(face.length>=3)(sign*(nx+ny)>0?light:dark).push(pathData(face.map(p=>({column:p.x,row:p.y})),true));
      }
      if(building.roof==='hip'||building.roof==='gate')for(const tip of [a,b]){
        const corners=building.points.slice().sort((p,q)=>Math.hypot(p.column-tip.column,p.row-tip.row)-Math.hypot(q.column-tip.column,q.row-tip.row)).slice(0,2);
        hipLines.push(...corners.map(p=>pathData([p,tip],false)));
      }
    }
    const group=svg('g',{class:className});
    group.appendChild(svg('path',{d:dark.join(' '),fill:'#233e42',opacity:.18}));
    group.appendChild(svg('path',{d:light.join(' '),fill:'#ffedb9',opacity:.18}));
    group.appendChild(svg('path',{d:flat.join(' '),fill:'none',stroke:'#f3e8cd',opacity:.55,'stroke-width':.0003/metric}));
    group.appendChild(svg('path',{d:hipLines.join(' '),fill:'none',stroke:'#f3e8cd',opacity:.5,'stroke-width':.0003/metric}));
    parent.appendChild(group);
  }

  function appendMoats(city,plan){
    const group=svg('g',{class:'city-defensive-ditches','data-moat-count':plan.moats.length,fill:'none','stroke-linejoin':'round'});
    for(const moat of plan.moats){const d=pathData(moat.points,true),width=moat.widthMetres/1000/plan.gridCellKilometres;
      group.appendChild(svg('path',{d,stroke:'#a6b39a','stroke-width':width*1.5}));
      group.appendChild(svg('path',{d,stroke:moat.wet?'#87b6c4':'#b1a082','stroke-width':width}));
      for(const crossing of moat.crossings)group.appendChild(svg('path',{d:pathData(crossing,false),stroke:'#f1e9d5','stroke-width':plan.streetWidths.arterial*1.25}));
    }city.appendChild(group);
  }

  function appendWalls(city, cityPlan) {
    const group = svg("g", {class: "city-fortifications", "stroke-linejoin": "round"});
    for (const [name, walls, width] of [["outer", cityPlan.walls, 1.8], ["inner", cityPlan.innerWalls, 2.3]]) {
      const d = walls.map(points => pathData(points, false)).join(" ");
      group.appendChild(svg('path',{d,class:'city-wall-shadow',fill:'none',stroke:'#3d4639',opacity:.22,
        'stroke-width':cityPlan.streetWidths.collector*(width+.8),
        transform:`translate(${.0025/cityPlan.gridCellKilometres} ${.0035/cityPlan.gridCellKilometres})`}));
      group.appendChild(svg("path", {d, class: "city-" + name + "-walls", fill: "none", stroke: "#756c57",
        "stroke-width": cityPlan.streetWidths.collector * (width + 0.8)}));
      group.appendChild(svg("path", {d, fill: "none", stroke: "#e8dfc5", "stroke-width": cityPlan.streetWidths.collector * width}));
      group.appendChild(svg("path", {d, fill: "none", stroke: "#96866a", "stroke-width": cityPlan.streetWidths.local * 0.6,
        "stroke-dasharray": `${cityPlan.streetWidths.local * 2} ${cityPlan.streetWidths.local * 3}`}));
    }
    for(const plaza of cityPlan.plazas.filter(p=>p.precinct)){
      const d=plaza.walls.map(p=>pathData(p,false)).join(' ');
      group.appendChild(svg('path',{d,fill:'none',stroke:'#626456','stroke-width':cityPlan.streetWidths.collector*1.9,class:'city-palace-precinct'}));
      group.appendChild(svg('path',{d,fill:'none',stroke:'#d8ceb0','stroke-width':cityPlan.streetWidths.collector*1.2}));
    }
    group.appendChild(svg("path", {class: "city-wall-towers", "data-wall-tower-count": cityPlan.towers.length,
      d: cityPlan.towers.map(tower => pathData(tower.points, true)).join(" "), fill: "#b5aa8b",
      stroke: "#6e624d", "stroke-width": 0.8, "vector-effect": "non-scaling-stroke"}));
    group.appendChild(svg("path", {class: "city-tower-roofs",
      d: cityPlan.towers.flatMap(tower => tower.roofLines || []).map(points => pathData(points, false)).join(" "),
      fill: "none", stroke: "#f7efd9", "stroke-width": 0.6, "vector-effect": "non-scaling-stroke"}));
    const gates=svg('g',{class:'city-gatehouses','data-gate-count':cityPlan.gates.length});
    for(const gate of cityPlan.gates){
      gates.appendChild(svg('path',{d:gate.towers.map(points=>pathData(points,true)).join(' '),fill:'#b0a18a',stroke:'#726653',
        'stroke-width':.75,'vector-effect':'non-scaling-stroke'}));
      gates.appendChild(svg('path',{d:pathData(gate.passage,false),fill:'none',stroke:'#fff9eb','stroke-width':cityPlan.streetWidths.arterial,'stroke-linecap':'round'}));
      gates.appendChild(svg('path',{d:gate.towers.flatMap(points=>roofLines(points)).map(points=>pathData(points,false)).join(' '),
        fill:'none',stroke:'#f4ead4','stroke-width':.7,'vector-effect':'non-scaling-stroke'}));
    }
    const barbicans=svg('g',{class:'city-barbicans','data-barbican-count':cityPlan.barbicans.length});
    for(const court of cityPlan.barbicans){
      barbicans.appendChild(svg('path',{d:pathData(court.points,true),fill:'#e9e1cb',stroke:'none'}));
      barbicans.appendChild(svg('path',{d:court.walls.map(p=>pathData(p,false)).join(' '),fill:'none',stroke:'#766b57','stroke-width':cityPlan.streetWidths.collector*2.8,'stroke-linejoin':'round'}));
      barbicans.appendChild(svg('path',{d:court.walls.map(p=>pathData(p,false)).join(' '),fill:'none',stroke:'#e9dfc4','stroke-width':cityPlan.streetWidths.collector*1.8,'stroke-linejoin':'round'}));
      barbicans.appendChild(svg('path',{d:pathData(court.access,false),fill:'none',stroke:'#fff8e9','stroke-width':cityPlan.streetWidths.arterial,'stroke-linecap':'round'}));
    }group.appendChild(barbicans);
    group.appendChild(gates);
    city.appendChild(group);
  }

  function appendCountryside(city, cityPlan) {
    const group = svg("g", {class: "city-countryside"});
    for(const fill of new Set(cityPlan.farmland.map(field=>field.fill))){
      const fields=cityPlan.farmland.filter(field=>field.fill===fill);
      group.appendChild(svg("path", {class: "city-farmland", "data-farmland-count": fields.length,
        d: fields.map(field => pathData(field.points, true)).join(" "), fill, "fill-opacity": 0.60,
        stroke: "#9da371", "stroke-opacity":.56,"stroke-width": 0.50, "vector-effect": "non-scaling-stroke"}));
    }
    group.appendChild(svg("path", {class: "city-field-rows", d: cityPlan.farmland.flatMap(field => field.lines || [])
      .map(points => pathData(points, false)).join(" "), fill: "none", stroke: "#a7b07a", "stroke-width": 0.5,
      "vector-effect": "non-scaling-stroke", opacity: 0.50}));
    for(const fill of new Set(cityPlan.countryside.map(house=>house.fill)))group.appendChild(svg("path", {
      class: "city-country-houses", d: cityPlan.countryside.filter(house=>house.fill===fill).map(house => pathData(house.points, true)).join(" "),
      fill, stroke: "#5e6353", "stroke-width": .00045 / cityPlan.gridCellKilometres, "stroke-opacity": .72}));
    appendRoofPaint(group,cityPlan.countryside,'city-country-roof-faces',cityPlan.gridCellKilometres);
    group.appendChild(svg('path',{class:'city-country-lanes',d:cityPlan.countryLanes.map(points=>pathData(points,false)).join(' '),
      fill:'none',stroke:'#ece5ce','stroke-width':cityPlan.streetWidths.local*1.4,'stroke-linejoin':'round','stroke-linecap':'round'}));
    group.appendChild(svg("path", {d: cityPlan.countryside.flatMap(house => house.roofLines || []).map(points => pathData(points, false)).join(" "),
      fill: "none", stroke: "#f0d6ac", "stroke-width": .0003 / cityPlan.gridCellKilometres}));
    city.appendChild(group);
  }

  function appendLabels(city, cityPlan) {
    const group=svg('g',{class:'city-place-labels','pointer-events':'none',fill:'#495247',stroke:'#fffdf5',
      'paint-order':'stroke fill','stroke-linejoin':'round','font-family':'Microsoft YaHei, sans-serif',
      'text-anchor':'middle','dominant-baseline':'middle'});
    for(const feature of cityPlan.annotations){
      const node=svg('text',{x:0,y:0,'data-anchor-column':feature.anchor.column,'data-anchor-row':feature.anchor.row,
        transform:`translate(${feature.anchor.column} ${feature.anchor.row})`,'data-city-screen-label':feature.name,
        'data-feature-kind':feature.kind,'data-feature-id':feature.id,'data-priority':feature.priority,
        'data-detail-level':feature.detailLevel,'data-length-km':feature.lengthKm||0,
        'data-angle':feature.angle||0,'font-size':cityPlan.footprintRadiusRows*.04,
        'stroke-width':cityPlan.streetWidths.local,fill:feature.kind==='road'?'#7b7569':'#3f635b'});
      node.textContent=feature.name;group.appendChild(node);
    }
    city.appendChild(group);
  }

  function featureAnnotations(recipe,fabric,character) {
    const result=[],prefix=character.prefix,center=points=>{
      const p=polygonCentre(points.map(p=>({x:p.column,y:p.row})));return {row:p.y,column:p.x};
    };
    const point=(id,name,kind,points,priority=3,detailLevel=1)=>result.push({id,name,kind,anchor:center(points),priority,detailLevel});
    for(const [index,plaza] of fabric.plazas.entries())if(plaza.market)point('square-'+index,plaza.name,'square',plaza.points,2,0);
    const seen=new Set();
    for(const landmark of fabric.landmarks)if(landmark.name&&!seen.has(landmark.compoundId)){
      const principal=['royal-palace','imperial-palace','council-house','holy-tomb','water-sanctuary','observatory-sanctuary','pilgrimage-monastery','harbor-shrine','great-sanctuary'].includes(landmark.kind);
      seen.add(landmark.compoundId);point(landmark.compoundId,landmark.name,'landmark',landmark.compoundPoints,principal?0:2,0);
    }
    if(fabric.fortress)point('citadel',prefix+(recipe.siteType==='pass'?'关堡':'卫城'),'landmark',fabric.fortress.points,1,0);
    for(const [index,gate] of fabric.gates.entries())point('gate-'+index,gate.name,'gate',gate.towers.flat(),2,1);
    for(const [index,court] of fabric.barbicans.entries())point('barbican-'+index,court.name,'gate',court.points,3,2);
    for(const [index,moat] of fabric.moats.entries())point('moat-'+index,moat.name,'water',moat.points.slice(0,2),4,1);
    for(const [index,hamlet] of fabric.hamlets.entries())point('hamlet-'+index,hamlet.name,'hamlet',hamlet.points,5,0);
    const longest=new Map(),metric=recipe.urban.gridCellKilometres;
    for(const street of fabric.streets)if(street.name){
      const a=street.points[0],b=street.points.at(-1),dx=(b.column-a.column)*metric.column,dy=(b.row-a.row)*metric.row,lengthKm=Math.hypot(dx,dy);
      if(lengthKm<.06||longest.get(street.name)?.lengthKm>=lengthKm)continue;
      let angle=Math.atan2(dy,dx)*180/Math.PI;if(angle>90)angle-=180;if(angle< -90)angle+=180;
      longest.set(street.name,{id:street.id,name:street.name,kind:'road',anchor:{row:(a.row+b.row)/2,column:(a.column+b.column)/2},
        priority:street.kind==='arterial'?3:6,detailLevel:street.kind==='arterial'?1:2,lengthKm,angle});
    }
    return result.concat(Array.from(longest.values()));
  }

  // A convex partition creates streets and blocks from the same cuts.  Every
  // cut terminates on an existing boundary; buildings are subdivided afterwards.
  // No Watabou source is used: this is a bounded BSP parcel planner.
  function urbanFabric(request) {
    const modern = request.era === "industrial" || request.era === "contemporary";
    const character=request.character;
    const tribal = request.era === "tribal"||character.mobile;
    const defensiveEra = ["ancient", "medieval", "early-modern", "preindustrial"].includes(request.era);
    const fortifiedSite = ["fortress", "pass"].includes(request.recipe.siteType) || (request.recipe.landmarks || []).some(landmark => landmark === "citadel" || landmark === "government-seat");
    const courtyardStyle = character.court;
    const rowHouseStyle = ["terraced", "canal-side"].includes(request.culturalStyle) || request.buildingPattern === "terraced-compounds";
    const radiusKm = Math.max(0.05, finiteOr(request.recipe.urban.radiusKm, 1));
    const widths = streetWidthsInGrid(request.recipe, request.radiusRows, request.radiusColumns);
    const metric = gridCellKilometres(request.recipe, request.radiusRows, request.radiusColumns);
    const roadGap = widths.arterial * metric / radiusKm * 0.58;
    const targetBlocks = character.small?Math.max(6,Math.min(24,Math.round(request.population/80))):Math.max(25,Math.min(235,Math.round(character.buildingCount/13)));
    const targetArea = Math.PI / (targetBlocks * (tribal ? 0.48 : 1));
    const groundForm=request.recipe.localSite?.form;
    let angle = groundForm?groundForm.axisRadians:request.orientation*Math.PI/180;
    const planned=['axial','orthogonal'].includes(character.plan)&&(!groundForm||['plain','hills'].includes(groundForm.kind));
    if(character.plan==='axial'&&groundForm?.kind==='plain')angle=0;
    if(!groundForm&&['hill-town','fortress','border-pass'].includes(character.kind)&&request.recipe.localSite?.surface){
      const field=request.recipe.localSite.surface,b=field.bounds,x=Math.max(1,Math.min(field.columns-2,Math.round((request.center.column-b.west)/(b.east-b.west)*(field.columns-1)))),
        y=Math.max(1,Math.min(field.rows-2,Math.round((request.center.row-b.north)/(b.south-b.north)*(field.rows-1)))),index=y*field.columns+x;
      const dx=field.elevationMetres[index+1]-field.elevationMetres[index-1],dy=field.elevationMetres[index+field.columns]-field.elevationMetres[index-field.columns];
      if(Math.hypot(dx,dy)>.1)angle=Math.atan2(dx,-dy);
    }
    const toWorld = point => ({
      column: request.center.column + (point.x * Math.cos(angle) - point.y * Math.sin(angle)) * request.radiusColumns,
      row: request.center.row + (point.x * Math.sin(angle) + point.y * Math.cos(angle)) * request.radiusRows,
    });
    const toLocal=p=>{const x=(p.column-request.center.column)/request.radiusColumns,y=(p.row-request.center.row)/request.radiusRows;
      return {x:x*Math.cos(angle)+y*Math.sin(angle),y:-x*Math.sin(angle)+y*Math.cos(angle)};};
    const outline = planned&&character.plan==='axial'?[{x:-.94,y:-.86},{x:.94,y:-.86},{x:.94,y:.86},{x:-.94,y:.86}]:Array.from({length: 48}, (_, index) => {
      const theta = (index + (random(request.seed, 2500 + index) - 0.5) * 0.5) * Math.PI / 24;
      // All vertices lie on one convex ellipse: subdivision cannot self-intersect.
      const elongated=['hill-town','border-pass','fortress','river-town','river-port','seaport','lake-port'].includes(character.kind);
      return {x: Math.cos(theta) * (groundForm?groundForm.alongScale:elongated?1.02:.95),
        y:Math.sin(theta)*(groundForm?groundForm.crossScale:['steep','escarpment'].includes(request.slopeClass)?.30:elongated?.73:.92)};
    });
    const segments = [];
    const parcels = [];
    const split = (polygon, depth, channel) => {
      const area = polygonArea(polygon);
      if (depth >= 11 || area < targetArea * (0.74 + random(request.seed, channel) * 0.65)) {
        parcels.push({polygon, channel}); return;
      }
      const centre = polygonCentre(polygon);
      const historicCore = Math.hypot(centre.x, centre.y) < (modern ? 0.30 : request.era === "early-modern" ? 0.56 : 0.85);
      const ordered = planned||!historicCore && (modern || request.era === "early-modern");
      const chaos = ordered ? (request.era === "industrial" ? 0.035 : 0.015) : (tribal ? 0.8 : 0.22 + request.streetIrregularity * 0.58);
      const result = bisectParcel(polygon, request.seed, channel, chaos);
      if (!result) { parcels.push({polygon, channel}); return; }
      const names=character.mobile?['集会道','牧场道','驿道','水源道']:character.kind==='oasis'?['井泉街','商旅街','驼铃街','绿荫街']:
        ['市集街','长街','粮仓街','石匠街','榆树街','工匠街','学舍街','花园街'];
      const kind=depth<2?'arterial':depth<5?'collector':'local';
      const worldCut=result.cut.map(toWorld),a=worldCut[0],b=worldCut[1];
      segments.push({id:'street-'+channel,kind,name:kind==='local'?null:depth<2?character.prefix+(Math.abs(b.row-a.row)>Math.abs(b.column-a.column)?'南北大街':'东西大街'):names[(channel+Math.round(random(request.seed,7)*7))%names.length],points:worldCut});
      split(result.first, depth + 1, channel * 2 + 1);
      split(result.second, depth + 1, channel * 2 + 2);
    };
    split(outline, 0, 31);
    const throughRiver=request.recipe.localSite?.riverMode==='through';
    const candidates = parcels.filter(parcel => {
      const centre = polygonCentre(parcel.polygon),along=groundForm?.alongScale||.95,cross=groundForm?.crossScale||.92,
        bearing = Math.atan2(centre.y/cross, centre.x/along);
      const growthLimit = 0.78 + Math.sin(bearing * 3 + random(request.seed, 7100) * Math.PI * 2) * 0.11
        + Math.sin(bearing * 5 + random(request.seed, 7101) * Math.PI * 2) * 0.075;
      return (planned&&character.plan==='axial'?Math.max(Math.abs(centre.x/.94),Math.abs(centre.y/.86))<=.98:
        Math.hypot(centre.x/along, centre.y/cross)<=growthLimit)
        &&(request.recipe.localSite||parcel.polygon.map(toWorld).every(point=>segmentOnLand(request.center,point,request.recipe)));
    }).flatMap(parcel=>request.recipe.localSite?clipToLand(parcel.polygon.map(toWorld),request.recipe).map(points=>({...parcel,polygon:points.map(toLocal)})):[parcel]);
    // Keep the connected settlement around the centre. A road or bridge is
    // required before development can grow across a forbidden water corridor.
    const selectedParcels = connectedParcels(candidates,throughRiver?(request.recipe.localSite.bridges||[]).map(b=>b.points.map(toLocal)):[]);
    const districtPolygons=selectedParcels.map(parcel=>parcel.polygon.map(toWorld));
    const footprint = outline.map(toWorld);
    const corridors = [...((request.recipe.transport || {}).corridors || []),...(request.recipe.localSite?.ruralRoads||[])].filter(corridor =>
      Array.isArray(corridor.points) && corridor.points.length > 1);
    const streets = segments.filter(street=>streetCrossingsValid(street,request.recipe)).flatMap(street=>clipToDistricts(street,districtPolygons));
    // Real routes are authoritative. Connect each to the existing street graph
    // by a short land-only access, never move its line or fabricate a bridge.
    for (const corridor of corridors) {
      const a=corridor.points[0],b=corridor.points.at(-1),dx=b.column-a.column,dy=b.row-a.row;
      const direction=Math.abs(dx)>Math.abs(dy)?(dx>0?'东':'西'):(dy>0?'南':'北');
      streets.push(...clipToDistricts({id:corridor.id,kind:corridor.kind==='rail'?'rail':corridor.local?'collector':'arterial',name:corridor.local?direction+'庄道':direction+'驿路',points:corridor.points,authoritative:true},districtPolygons));
      if (corridor.kind === "rail") continue;
      let closest = null;
      for (const street of streets) for (const point of street.points) {
        for (let index = 1; index < corridor.points.length; index++) {
          const target = nearestPoint(point, corridor.points[index - 1], corridor.points[index]);
          const distance = Math.hypot(point.row - target.row, point.column - target.column);
          if ((!closest || distance < closest.distance) && segmentOnLand(point, target, request.recipe)) closest = {point, target, distance};
        }
      }
      if (closest && closest.distance < Math.max(request.radiusRows, request.radiusColumns) * 0.7) {
        streets.push({kind: "collector", points: [closest.point, closest.target], access: true});
      }
    }
    const publicSpaces = createPublicSpaces(request, metric, streets,districtPolygons);
    const {plazas, landmarks} = publicSpaces;
    const fortress = defensiveEra && fortifiedSite ? createFortress(selectedParcels, request, toWorld, radiusKm, streets, plazas) : null;
    if (fortress) streets.push({kind: "collector", points: fortress.access, access: true});
    const protectedRoutes = streets.filter(street => street.access || street.authoritative);
    const buildings = [];
    const blocks = [];
    const districts = [];
    const occupiedParcels = [], innerParcels = [];
      const palette = character.roof;
    for (const parcel of selectedParcels) {
      const centre = polygonCentre(parcel.polygon);
        const radial = Math.hypot(centre.x/(groundForm?.alongScale||.95), centre.y/(groundForm?.crossScale||.92));
      const vacant = random(request.seed, parcel.channel + 17000) < (tribal ? 0.24 : radial > 0.8 ? 0.10 : 0.035);
      const worldDistrict = parcel.polygon.map(toWorld);
      const occupied = insetPolygon(parcel.polygon, roadGap);
      if (occupied.length < 3) continue;
      occupiedParcels.push(parcel.polygon);
      if (radial < 0.46) innerParcels.push(parcel.polygon);
      let kind = radial < (modern ? 0.30 : 0.46) ? "old-town" : modern ? "planned-extension" : "residential";
      const worldCentre = toWorld(centre);
      const cores = request.recipe.urban.coreZones || [];
      const nearestCore = cores.reduce((best, core) => {
        const distance = Math.hypot((worldCentre.row - core.row) / request.radiusRows, (worldCentre.column - core.column) / request.radiusColumns);
        return !best || distance < best.distance ? {core, distance} : best;
      }, null);
      if (nearestCore && nearestCore.distance < 0.12) kind = "civic";
      if (modern && radial > 0.6 && corridors.some(corridor => corridor.kind === "rail" && routeNear(worldCentre, corridor.points, request.radiusRows * 0.25))) kind = "industrial";
      if (vacant) kind = "park";
      for (const points of clipToLand(occupied.map(toWorld), request.recipe)) {
        blocks.push({points, kind, fill: kind === "park" ? "#c6d3b3" : "#ede5d4", opacity: "0.90"});
        districts.push(worldDistrict);
      }
      if (vacant) continue;
      const lots = [];
      const density = radial < 0.46 ? 1.1 : 0.65 + (1 - radial) * 0.30;
      const perBlock=Math.max(2,Math.min(28,character.buildingCount/Math.max(1,selectedParcels.length)));
      const lotTarget = Math.max(polygonArea(occupied) / (perBlock * density), (modern ? 0.0007 : 0.00028) / (radiusKm * radiusKm));
      const subdivideLots = (polygon, depth, channel) => {
        if (depth >= 5 || polygonArea(polygon) < lotTarget * (0.8 + random(request.seed, channel) * 0.65)) { lots.push({polygon, channel}); return; }
        const division = bisectParcel(polygon, request.seed, channel, kind === "old-town" && !rowHouseStyle ? 0.20 : 0.025);
        if (!division) { lots.push({polygon, channel}); return; }
        // Each subdivision leaves an alley that reaches the parent street.
        const alley = Math.max(0.0009 / radiusKm, roadGap * 0.15);
        streets.push({id:'lane-'+channel,kind:'local',points:division.cut.map(toWorld),alley:true});
        const first = insetPolygon(division.first, alley), second = insetPolygon(division.second, alley);
        if (first.length >= 3) subdivideLots(first, depth + 1, channel * 2 + 1);
        if (second.length >= 3) subdivideLots(second, depth + 1, channel * 2 + 2);
      };
      subdivideLots(occupied, 0, parcel.channel + 30000);
      const lotBounds={west:Math.min(...worldDistrict.map(p=>p.column))-widths.arterial*3,east:Math.max(...worldDistrict.map(p=>p.column))+widths.arterial*3,
        north:Math.min(...worldDistrict.map(p=>p.row))-widths.arterial*3,south:Math.max(...worldDistrict.map(p=>p.row))+widths.arterial*3};
      const parcelStreets=streets.filter(s=>s.points.some((p,i)=>i&&Math.max(p.column,s.points[i-1].column)>=lotBounds.west
        &&Math.min(p.column,s.points[i-1].column)<=lotBounds.east&&Math.max(p.row,s.points[i-1].row)>=lotBounds.north&&Math.min(p.row,s.points[i-1].row)<=lotBounds.south));
      for (const lot of lots) {
        const shrink = character.small?.36:tribal?.50:kind === "planned-extension" ? (request.era === "contemporary" ? 0.26 : 0.17) : Math.max(character.shrink,radial>.58?.19:0);
        const inset = Math.sqrt(polygonArea(lot.polygon)) * shrink * (0.7 + random(request.seed, lot.channel) * 0.5);
        let shape = insetPolygon(lot.polygon, inset);
        if (rowHouseStyle && shape.length >= 3) {
          const values = shape.map(point => point.y);
          const minimum = Math.min(...values), maximum = Math.max(...values);
          shape = clipHalfPlane(shape, 0, 1, maximum - (maximum - minimum) * 0.24);
        }
        if (!modern && !rowHouseStyle && !courtyardStyle && random(request.seed, lot.channel + 9100) < 0.30) {
          shape = chamferPolygon(shape, 0.08 + random(request.seed, lot.channel + 9101) * 0.09);
        }
        if (shape.length < 3) continue;
        if(character.mobile){const c=polygonCentre(shape),r=Math.sqrt(polygonArea(shape))*.31;
          shape=Array.from({length:12},(_,i)=>({x:c.x+Math.cos(i*Math.PI/6)*r,y:c.y+Math.sin(i*Math.PI/6)*r}));}
        let points = shape.map(toWorld);
        if(!character.mobile)points=frontageBuilding(points,parcelStreets,request.recipe,lot.channel,courtyardStyle);
        if(points.length<3)continue;
        if (fortress && polygonsOverlap(points, fortress.clearancePoints)) continue;
        if (protectedRoutes.some(route => polygonNearRoute(points, route.points, (route.kind === "collector" ? widths.collector : widths.arterial) * 0.75))) continue;
        if (plazas.some(plaza => polygonsOverlap(points, plaza.points))) continue;
        const foundation=foundationAt(points,request.recipe);
        if(foundation.reliefMetres>3.5)continue;
        const harbor=request.recipe.harbor;
        if(harbor?.reservedLand&&polygonsOverlap(points,harbor.reservedLand))continue;
        for (const land of clipToLand(points, request.recipe)) {
          const building = {points: land, kind,foundation, material:character.material.name,
            roof:character.mobile?'dome':courtyardStyle?'hip':['arcaded','terraced'].includes(character.style)?'flat':'gable',
            fill: palette[Math.floor(random(request.seed,lot.channel+9200)*palette.length)]};
          if (courtyardStyle && !tribal && kind !== "industrial") {
            const polygon = land.map(point => ({x: point.column, y: point.row}));
            const inner = insetPolygon(polygon, Math.sqrt(polygonArea(polygon)) * (request.culturalStyle === "walled-compound" ? 0.16 : 0.23));
            if (inner.length >= 3) building.courtyard = inner.map(point => ({column: point.x, row: point.y}));
          }
          building.roofLines = roofLines(building.points, building.courtyard);
          buildings.push(building);
        }
      }
    }
    joinLaneEnds(streets,widths.arterial*2.5,request.recipe);
    let safeStreets=streets;
    for(const plaza of plazas)safeStreets=safeStreets.flatMap(street=>street.alley||plaza.precinct&&!street.palacePerimeter&&!street.plaza?
      streetOutsidePolygon(street,plaza.points):[street]);
    if(fortress)safeStreets=safeStreets.flatMap(street=>street.alley?streetOutsidePolygon(street,fortress.clearancePoints):[street]);
    safeStreets=safeStreets.flatMap(street=>clipStreetToLand(street,request.recipe));
    const walled = defensiveEra && !character.small && (request.tier !== "site" || fortifiedSite);
    const gates=[];
    const boundary = (parcels,layer) => {
      const firstGate=gates.length;
      const lines = [];
      for (const polygon of parcels) for (let index = 0; index < polygon.length; index++) {
      const a = polygon[index], b = polygon[(index + 1) % polygon.length];
      const length = Math.hypot(b.x - a.x, b.y - a.y);
      if (length < 1e-10) continue;
      // A BSP neighbour may share only part of an edge. Split there before
      // classifying exposed perimeter; whole-edge midpoint tests leave walls
      // through the city and gaps around its outside.
      const cuts=[0,1];
      for(const other of parcels)for(const point of other){
        const t=((point.x-a.x)*(b.x-a.x)+(point.y-a.y)*(b.y-a.y))/(length*length);
        if(t>1e-8&&t<1-1e-8&&Math.abs((point.x-a.x)*(b.y-a.y)-(point.y-a.y)*(b.x-a.x))<length*1e-8)cuts.push(t);
      }
      cuts.sort((x,y)=>x-y);
      for(let part=1;part<cuts.length;part++){
        const low=cuts[part-1],high=cuts[part];if(high-low<1e-8)continue;
        const t=(low+high)/2,outside={x:a.x+(b.x-a.x)*t+(b.y-a.y)/length*1e-6,y:a.y+(b.y-a.y)*t-(b.x-a.x)/length*1e-6};
        if(parcels.some(other=>containsPoint(other,outside)))continue;
        const point=value=>toWorld({x:a.x+(b.x-a.x)*value,y:a.y+(b.y-a.y)*value});
        if(layer==='outer'&&request.recipe.harbor?.kind==='sea'){
          const p=point(t),m=request.recipe.urban.gridCellKilometres,f=request.recipe.localSite.surface,
            reach=Math.max(.06,Math.min(.20,radiusKm*.14));
          if(Array.from({length:12},(_,i)=>({column:p.column+Math.cos(i*Math.PI/6)*reach/m.column,row:p.row+Math.sin(i*Math.PI/6)*reach/m.row}))
            .some(q=>global.WorldAtlasCitySite.elevationAt(f,q)<=0))continue;
        }
        lines.push(...wallSegmentsWithGates(point(low),point(high),safeStreets.filter(street=>!street.alley),widths.arterial*1.4,gates,request));
      }
      }
      for(const gate of gates.slice(firstGate)){gate.layer=layer;if(layer==='inner')gate.name='内城'+gate.name;}
      return lines.flatMap(points => clipStreetToLand({points}, request.recipe).map(wall => wall.points));
    };
    const defensiveEdges=lines=>lines.flatMap(points=>{
      let pieces=[{points}];
      for(const plaza of plazas)pieces=pieces.flatMap(piece=>streetOutsidePolygon(piece,plaza.points));
      return pieces.flatMap(piece=>usefulWallSegments(piece.points,request));
    });
    const walls = walled ? defensiveEdges(boundary(occupiedParcels,'outer')) : [];
    const innerWalls = walled && character.defence.layers>1 ? defensiveEdges(boundary(innerParcels,'inner')) : [];
    // Gates and towers belong to the final defence line, never the former
    // parcel boundary inside a palace or alongside a natural water barrier.
    for(let i=gates.length-1;i>=0;i--)if(![...walls,...innerWalls].some(line=>routeNear(gates[i].center,line,widths.arterial*2)))gates.splice(i,1);
    const barbicans=character.defence.barbicans?createBarbicans(gates,request,widths):[];
    const towers = wallTowers([...walls, ...innerWalls], request, metric, safeStreets);
    const rural=countryHouses(request,corridors,occupiedParcels.map(polygon=>polygon.map(toWorld)),metric,widths);
    const reservations=gates.flatMap(gate=>gate.towers).concat(towers.map(t=>t.points),barbicans.map(b=>b.points));
    const roadIndex=streetClearanceIndex(safeStreets,widths,request.center,.075/metric);
    let safeBuildings=buildings.filter(building=>!reservations.some(points=>polygonsOverlap(building.points,points))
      &&![...walls,...innerWalls].some(wall=>polygonNearRoute(building.points,wall,widths.collector*1.8))
      &&!roadIndex.intersects(building.points));
    rural.houses=rural.houses.filter(house=>!barbicans.some(b=>polygonsOverlap(house.points,b.points)));
    assignStreetNames(safeStreets,request,gates,landmarks);
    const moats=defensiveDitches(request,fortress,innerParcels,toWorld,safeStreets,landmarks);
    safeBuildings=safeBuildings.filter(building=>!moats.some(m=>polygonNearRoute(building.points,m.points.concat(m.points[0]),m.widthMetres/1500/metric)));
    const farmland=(request.recipe.localSite?.farmland||[]).filter(field=>!request.recipe.harbor?.reservedLand||!polygonsOverlap(field.points,request.recipe.harbor.reservedLand))
      .filter(field=>!rural.lanes.some(lane=>polygonNearRoute(field.points,lane,widths.local*1.2))
      &&!rural.houses.some(house=>polygonsOverlap(field.points,house.points)));
    return {footprint,blocks,districts,streets:safeStreets,buildings:safeBuildings,farmland,
      walls,innerWalls,towers,gates,barbicans,moats,plazas,landmarks,countryside:rural.houses,countryLanes:rural.lanes,hamlets:rural.hamlets,fortress};
  }

  function streetClearanceIndex(streets,widths,center,size){
    const bins=new Map(),keys=b=>{const out=[];
      for(let y=Math.floor((b.north-center.row)/size);y<=Math.floor((b.south-center.row)/size);y++)
        for(let x=Math.floor((b.west-center.column)/size);x<=Math.floor((b.east-center.column)/size);x++)out.push(x+','+y);
      return out;
    };
    for(const street of streets)for(let i=1;i<street.points.length;i++){
      const a=street.points[i-1],b=street.points[i],width=(widths[street.kind]||widths.local)*.60,entry={points:[a,b],width};
      for(const key of keys({west:Math.min(a.column,b.column)-width,east:Math.max(a.column,b.column)+width,
        north:Math.min(a.row,b.row)-width,south:Math.max(a.row,b.row)+width})){
        if(!bins.has(key))bins.set(key,[]);bins.get(key).push(entry);
      }
    }
    return {intersects(points){const entries=new Set(keys(polygonBounds(points)).flatMap(key=>bins.get(key)||[]));
      for(const entry of entries)if(polygonNearRoute(points,entry.points,entry.width))return true;return false;}};
  }

  function usefulWallSegments(points,request){
    const site=request.recipe.localSite;if(!site)return [points];
    const m=request.recipe.urban.gridCellKilometres,a=points[0],z=points.at(-1),
      dx=(z.column-a.column)*m.column,dy=(z.row-a.row)*m.row,length=Math.hypot(dx,dy);
    if(length<1e-8)return [];
    const steps=Math.max(1,Math.ceil(length/.02)),out=[];let run=[];
    const at=t=>({column:a.column+(z.column-a.column)*t,row:a.row+(z.row-a.row)*t});
    const useful=p=>!site.channels.some(channel=>{
      const near=global.WorldAtlasCitySite.distanceToChannel(p,channel,m);
      if(near.distance>near.widthMetres/2000+.035)return false;
      // Only a parallel bank wall is redundant. A transverse land wall may
      // terminate at a river, leaving the water as the remaining barrier.
      return channel.points.slice(1).some((b,i)=>{const c=channel.points[i],q=nearestPoint(p,c,b),
        cx=(b.column-c.column)*m.column,cy=(b.row-c.row)*m.row,d=Math.hypot((p.column-q.column)*m.column,(p.row-q.row)*m.row);
        return d<near.widthMetres/2000+.035&&Math.abs(dx*cx+dy*cy)/(length*Math.hypot(cx,cy))>.75;});
    });
    for(let i=0;i<steps;i++){
      if(useful(at((i+.5)/steps))){if(!run.length)run.push(at(i/steps));run.push(at((i+1)/steps));}
      else if(run.length){out.push([run[0],run.at(-1)]);run=[];}
    }
    if(run.length)out.push([run[0],run.at(-1)]);return out;
  }

  function assignStreetNames(streets,request,gates,landmarks){
    const character=request.character,groups=new Map(),used=new Set(),words=character.tradition.streets,
      rotation=Math.floor(random(request.recipe.seed,request.recipe.culture?.civilizationId||0)*words.length);
    for(const street of streets){if(street.kind==='rail'||street.kind==='local'||street.alley)continue;
      const key=street.id||'access-'+groups.size;if(!groups.has(key))groups.set(key,[]);groups.get(key).push(street);}
    let index=0;
    for(const [id,segments] of groups){const main=segments.reduce((a,b)=>Math.hypot(...[a.points.at(-1).column-a.points[0].column,a.points.at(-1).row-a.points[0].row])>
      Math.hypot(...[b.points.at(-1).column-b.points[0].column,b.points.at(-1).row-b.points[0].row])?a:b),points=main.points;
      let name=null,nearest=Infinity;
      for(const destination of gates.map(g=>({name:g.name,p:g.center})).concat(landmarks.filter(l=>l.kind.indexOf('-wing')<0).map(l=>({name:l.name,p:{row:l.points.reduce((s,p)=>s+p.row,0)/l.points.length,column:l.points.reduce((s,p)=>s+p.column,0)/l.points.length}})))){
        const d=Math.min(...points.map(p=>Math.hypot(p.row-destination.p.row,p.column-destination.p.column)));
        const candidate=destination.name+(character.plan==='axial'?'大街':'路');
        if(d<request.radiusRows*.08&&d<nearest&&!used.has(candidate)){nearest=d;name=candidate;}
      }
      if(id.includes('harbor-access'))name=character.prefix+'港埠路';
      if(id==='urban-river-spine')name=character.prefix+'通桥大街';
      if(!name||used.has(name)){
        const word=words[(rotation+index)%words.length],cycle=Math.floor(index/words.length),
          zone=cycle?['上','下','东','西','南','北'][cycle%6]:main.kind==='arterial'?character.prefix:'';
        name=zone+word+(main.kind==='arterial'&&character.plan==='axial'?'大街':character.mobile?'道':'街');
        while(used.has(name))name=character.prefix+name;
      }
      used.add(name);for(const street of segments)street.name=name;index++;
    }
  }

  function defensiveDitches(request,fortress,innerParcels,toWorld,streets,landmarks){
    const site=request.recipe.localSite;if(!site||!request.character.defence.layers||!['plain','hills'].includes(site.form.kind))return [];
    const candidates=[];
    if(fortress)candidates.push({points:fortress.clearancePoints,name:request.character.prefix+'卫城'});
    if(request.character.capital&&innerParcels.length){
      const points=innerParcels.flatMap(p=>p.map(toWorld)),sorted=points.map(p=>({x:p.column,y:p.row})).sort((a,b)=>a.x-b.x||a.y-b.y),cross=(a,b,c)=>(b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x),lower=[],upper=[];
      for(const p of sorted){while(lower.length>1&&cross(lower.at(-2),lower.at(-1),p)<=0)lower.pop();lower.push(p);}
      for(const p of sorted.slice().reverse()){while(upper.length>1&&cross(upper.at(-2),upper.at(-1),p)<=0)upper.pop();upper.push(p);}
      candidates.push({points:lower.slice(0,-1).concat(upper.slice(0,-1)).map(p=>({column:p.x,row:p.y})),name:'内城'});
    }
    const result=[],metric=gridCellKilometres(request.recipe,request.radiusRows,request.radiusColumns);
    for(const candidate of candidates){if(candidate.points.length<3)continue;
      const centre=polygonCentre(candidate.points.map(p=>({x:p.column,y:p.row}))),points=candidate.points.map(p=>{
        const dx=p.column-centre.x,dy=p.row-centre.y,len=Math.hypot(dx,dy);return {column:p.column+dx/len*.024/metric,row:p.row+dy/len*.024/metric};});
      if(!points.every((p,i)=>segmentOnLand(p,points[(i+1)%points.length],request.recipe))||landmarks.some(l=>polygonNearRoute(l.compoundPoints,points.concat(points[0]),.012/metric)))continue;
      const foundation=foundationAt(points,request.recipe);if(foundation.reliefMetres>8)continue;
      const wet=foundation.reliefMetres<1.8&&site.channels.length>0,widthMetres=candidate.name==='内城'?18:12,crossings=[];
      for(const street of streets.filter(s=>s.kind!=='local'))for(let i=1;i<street.points.length;i++)for(let j=0;j<points.length;j++){
        const a=street.points[i-1],b=street.points[i],c=points[j],d=points[(j+1)%points.length],dx=b.column-a.column,dy=b.row-a.row,sx=d.column-c.column,sy=d.row-c.row,den=dx*sy-dy*sx;
        if(Math.abs(den)<1e-12)continue;const t=((c.column-a.column)*sy-(c.row-a.row)*sx)/den,u=((c.column-a.column)*dy-(c.row-a.row)*dx)/den;
        if(t<0||t>1||u<0||u>1)continue;const len=Math.hypot(dx,dy),half=widthMetres/1500/metric;
        crossings.push([-1,1].map(sign=>({column:a.column+dx*t+dx/len*half*sign,row:a.row+dy*t+dy/len*half*sign})));
      }
      result.push({points,name:candidate.name+(wet?'护城河':'防御壕'),wet,widthMetres,crossings,foundation});
    }return result;
  }

  function createBarbicans(gates,request,widths){
    const courts=[],metric=gridCellKilometres(request.recipe,request.radiusRows,request.radiusColumns);
    for(const gate of gates.filter(g=>g.layer==='outer')){
      if(courts.length>=4)break;
      const [a,b]=gate.passage,dx=b.column-a.column,dy=b.row-a.row,length=Math.hypot(dx,dy);
      if(!length)continue;
      const dot=dx*(gate.center.column-request.center.column)+dy*(gate.center.row-request.center.row),
        outward={x:dx/length*(dot>=0?1:-1),y:dy/length*(dot>=0?1:-1)},tangent={x:-outward.y,y:outward.x},
        half=Math.max(widths.arterial*2.2,.026/metric),depth=Math.max(widths.arterial*4,.065/metric);
      const at=(x,y)=>({column:gate.center.column+tangent.x*x+outward.x*y,row:gate.center.row+tangent.y*x+outward.y*y});
      const points=[at(-half,0),at(half,0),at(half,depth),at(-half,depth)];
      if(!points.every(p=>segmentOnLand(gate.center,p,request.recipe)))continue;
      const walls=[[points[0],points[3]],[points[1],points[2]],[points[3],at(-widths.arterial*.9,depth)],[at(widths.arterial*.9,depth),points[2]]];
      const access=[gate.center,at(0,depth+widths.arterial)];
      if(!segmentOnLand(access[0],access[1],request.recipe))continue;
      courts.push({points,walls,access,name:gate.name+'瓮城'});
    }return courts;
  }

  function frontageBuilding(polygon,streets,recipe,channel,courtyard) {
    const center=polygonCentre(polygon.map(p=>({x:p.column,y:p.row}))),point={column:center.x,row:center.y};
    let nearest=null;
    for(const street of streets)if(street.kind!=='rail')for(let i=1;i<street.points.length;i++){
      const a=street.points[i-1],b=street.points[i],target=nearestPoint(point,a,b),distance=Math.hypot(point.row-target.row,point.column-target.column);
      if(!nearest||distance<nearest.distance)nearest={a,b,target,distance};
    }
    if(!nearest)return polygon;
    const dx=nearest.b.column-nearest.a.column,dy=nearest.b.row-nearest.a.row,length=Math.hypot(dx,dy);
    if(length<1e-9)return polygon;
    const u={x:dx/length,y:dy/length},v={x:-u.y,y:u.x};
    const projected=polygon.map(p=>({x:(p.column-center.x)*u.x+(p.row-center.y)*u.y,y:(p.column-center.x)*v.x+(p.row-center.y)*v.y}));
    const spanX=Math.max(...projected.map(p=>p.x))-Math.min(...projected.map(p=>p.x)),
      spanY=Math.max(...projected.map(p=>p.y))-Math.min(...projected.map(p=>p.y)),
      front=(nearest.target.column-center.x)*v.x+(nearest.target.row-center.y)*v.y,
      setback=courtyard?0:Math.sign(front)*spanY*.06;
    let halfX=spanX*(courtyard?.46:.33+random(recipe.seed,channel+12100)*.13),
      halfY=spanY*(courtyard?.46:.30+random(recipe.seed,channel+12101)*.16);
    const shape=()=>[[-1,-1],[1,-1],[1,1],[-1,1]].map(([x,y])=>({column:center.x+u.x*x*halfX+v.x*(y*halfY+setback),row:center.y+u.y*x*halfX+v.y*(y*halfY+setback)}));
    const local=polygon.map(p=>({x:p.column,y:p.row}));
    for(let attempt=0;attempt<24;attempt++,halfX*=.91,halfY*=.91){
      const rect=shape();if(rect.every(p=>containsPoint(local,{x:p.column,y:p.row})))return rect;
    }
    return [];
  }

  function joinLaneEnds(streets,maximum,recipe) {
    const origin=recipe.location,bins=new Map(),order=new Map(streets.map((s,i)=>[s,i])),
      cell=p=>[Math.floor((p.column-origin.column)/maximum),Math.floor((p.row-origin.row)/maximum)],key=p=>cell(p).join(',');
    for(const street of streets)if(street.kind!=='rail')for(let index=1;index<street.points.length;index++){
      const a=street.points[index-1],b=street.points[index],count=Math.max(1,Math.ceil(Math.hypot(b.column-a.column,b.row-a.row)/maximum));
      const segment={street,index},keys=new Set();
      for(let i=0;i<=count;i++)keys.add(key({column:a.column+(b.column-a.column)*i/count,row:a.row+(b.row-a.row)*i/count}));
      for(const k of keys){if(!bins.has(k))bins.set(k,[]);bins.get(k).push(segment);}
    }
    for(const lane of streets)if(lane.alley)for(const index of [0,lane.points.length-1]){
      const p=lane.points[index];let closest=null;
      const candidates=new Set(),[x,y]=cell(p);
      for(let dx=-1;dx<=1;dx++)for(let dy=-1;dy<=1;dy++)for(const segment of bins.get(`${x+dx},${y+dy}`)||[])candidates.add(segment);
      for(const {street,index:i} of Array.from(candidates).sort((a,b)=>order.get(a.street)-order.get(b.street)||a.index-b.index))if(street!==lane){
        const q=nearestPoint(p,street.points[i-1],street.points[i]),distance=Math.hypot(p.row-q.row,p.column-q.column);
        if(distance<1e-10) {closest={q,distance};break;}
        if(distance<=maximum&&(!closest||distance<closest.distance-1e-10)&&segmentOnLand(p,q,recipe))closest={q,distance};
      }
      if(closest)lane.points[index]=closest.q;
    }
  }

  function streetOutsidePolygon(street,polygon) {
    const inside=clipToDistricts(street,[polygon]);if(!inside.length)return [street];
    const a=street.points[0],b=street.points.at(-1),dx=b.column-a.column,dy=b.row-a.row,squared=dx*dx+dy*dy;
    const t=p=>((p.column-a.column)*dx+(p.row-a.row)*dy)/squared;
    const intervals=[[0,Math.max(0,t(inside[0].points[0]))],[Math.min(1,t(inside.at(-1).points.at(-1))),1]];
    return intervals.filter(([low,high])=>high-low>1e-7).map(([low,high])=>({...street,points:[low,high].map(v=>({row:a.row+dy*v,column:a.column+dx*v}))}));
  }

  function connectedParcels(parcels,bridges=[]) {
    if(!parcels.length)return [];
    const neighbours=(first,second)=>first.some((a,index)=>{
      const b=first[(index+1)%first.length],dx=b.x-a.x,dy=b.y-a.y,length=Math.hypot(dx,dy);
      if(length<1e-9)return false;
      return second.some((c,edge)=>{
        const d=second[(edge+1)%second.length];
        if(Math.abs(dx*(c.y-a.y)-dy*(c.x-a.x))>length*1e-7||Math.abs(dx*(d.y-a.y)-dy*(d.x-a.x))>length*1e-7)return false;
        const t=p=>((p.x-a.x)*dx+(p.y-a.y)*dy)/(length*length);
        return Math.min(1,Math.max(t(c),t(d)))-Math.max(0,Math.min(t(c),t(d)))>1e-7;
      });
    });
    const core=parcels.reduce((best,parcel)=>{
      const c=polygonCentre(parcel.polygon),b=polygonCentre(best.polygon);
      return Math.hypot(c.x,c.y)<Math.hypot(b.x,b.y)?parcel:best;
    },parcels[0]);
    const result=[core],remaining=new Set(parcels.filter(parcel=>parcel!==core));
    const near=(polygon,p)=>containsPoint(polygon,p)||polygon.some((a,i)=>{
      const b=polygon[(i+1)%polygon.length],q=nearestPoint({column:p.x,row:p.y},{column:a.x,row:a.y},{column:b.x,row:b.y});
      return Math.hypot(q.column-p.x,q.row-p.y)<.12;});
    for(let index=0;index<result.length;index++)for(const parcel of remaining)if(neighbours(result[index].polygon,parcel.polygon)
      ||bridges.some(([a,b])=>near(result[index].polygon,a)&&near(parcel.polygon,b)||near(result[index].polygon,b)&&near(parcel.polygon,a))){result.push(parcel);remaining.delete(parcel);}
    return result;
  }

  function clipToDistricts(street,polygons) {
    const result=[];
    for(let segment=1;segment<street.points.length;segment++){
      const a=street.points[segment-1],b=street.points[segment],intervals=[];
      for(const polygon of polygons){
        let low=0,high=1;
        for(let edge=0;edge<polygon.length&&low<=high;edge++){
          const c=polygon[edge],d=polygon[(edge+1)%polygon.length],dx=d.column-c.column,dy=d.row-c.row;
          const length=Math.hypot(dx,dy);if(length<1e-12)continue;
          const raw=(dx*(a.row-c.row)-dy*(a.column-c.column))/length,
            start=Math.abs(raw)<1e-10?0:raw,delta=(dx*(b.row-a.row)-dy*(b.column-a.column))/length;
          if(Math.abs(delta)<1e-10){if(start<0){low=1;high=0;}continue;}
          const t=-start/delta;if(delta>0)low=Math.max(low,t);else high=Math.min(high,t);
        }
        if(high-low>1e-8)intervals.push([low,high]);
      }
      intervals.sort((x,y)=>x[0]-y[0]);const joined=[];
      for(const interval of intervals){const last=joined.at(-1);if(last&&interval[0]<=last[1]+1e-7)last[1]=Math.max(last[1],interval[1]);else joined.push(interval.slice());}
      const point=t=>({row:a.row+(b.row-a.row)*t,column:a.column+(b.column-a.column)*t});
      for(const [low,high] of joined)result.push({...street,points:[point(low),point(high)]});
    }return result;
  }

  function createFortress(parcels, request, toWorld, radiusKm, streets, plazas) {
    const core = (request.recipe.urban.coreZones || []).find(zone => zone.role === "government-core") || request.center;
    const candidates = parcels.slice().sort((a, b) => {
      const first = toWorld(polygonCentre(a.polygon)), second = toWorld(polygonCentre(b.polygon));
      return Math.hypot(first.row - core.row, first.column - core.column) - Math.hypot(second.row - core.row, second.column - core.column);
    });
    for (const parcel of candidates) {
      const center = polygonCentre(parcel.polygon);
      let radius = Math.min(0.24 / radiusKm, Math.sqrt(polygonArea(parcel.polygon)) * 0.37);
      const square = (point, half) => [{x: point.x - half, y: point.y - half}, {x: point.x + half, y: point.y - half}, {x: point.x + half, y: point.y + half}, {x: point.x - half, y: point.y + half}];
      for (let attempt = 0; attempt < 5; attempt++, radius *= 0.78) {
        const local = square(center, radius);
        const clearancePoints = square(center, radius * 1.22);
        if (!clearancePoints.every(point => containsPoint(parcel.polygon, point))) continue;
        const points = local.map(toWorld);
        if(foundationAt(points,request.recipe).reliefMetres>4)continue;
        if (plazas.some(plaza => polygonsOverlap(points, plaza.points))) continue;
        if (!clearancePoints.map(toWorld).every(point => segmentOnLand(toWorld(center), point, request.recipe))) continue;
        if (streets.some(street => street.kind !== "local" && polygonNearRoute(points, street.points, request.radiusRows * 0.002))) continue;
        let gate = null;
        for (let edge = 0; edge < 4; edge++) {
          const a = points[edge], b = points[(edge + 1) % 4], midpoint = {row: (a.row + b.row) / 2, column: (a.column + b.column) / 2};
          for (const street of streets) if (street.kind !== "rail") for (let index = 1; index < street.points.length; index++) {
            const target = nearestPoint(midpoint, street.points[index - 1], street.points[index]);
            const distance = Math.hypot(midpoint.row - target.row, midpoint.column - target.column);
            if ((!gate || distance < gate.distance) && segmentOnLand(midpoint, target, request.recipe)) gate = {edge, midpoint, target, distance};
          }
        }
        if (!gate) continue;
        const walls = [];
        for (let edge = 0; edge < 4; edge++) {
          const a = points[edge], b = points[(edge + 1) % 4];
          for (const [start, end] of edge === gate.edge ? [[0, 0.42], [0.58, 1]] : [[0, 1]]) walls.push([start, end].map(t => ({row: a.row + (b.row - a.row) * t, column: a.column + (b.column - a.column) * t})));
        }
        const ancient=request.era==='ancient',kind=ancient?(request.character.court?'palace-citadel':'garrison'):'castle';
        const interiors=[];
        if(ancient){
          const part=(x,y,w,h)=>rectangle({row:center.y+y*radius,column:center.x+x*radius},w*radius,h*radius).map(p=>toWorld({x:p.column,y:p.row}));
          interiors.push({points:part(0,-.48,.62,.18),kind:'command-hall'});
          for(const x of [-.55,.55])for(const y of [-.05,.43])interiors.push({points:part(x,y,.20,.15),kind:'barracks'});
          if(request.character.court)interiors.push({points:part(0,.42,.26,.12),kind:'sanctuary'});
        }
        return {
          era: request.era, points, clearancePoints: clearancePoints.map(toWorld), walls, access: [gate.midpoint, gate.target],
          kind,interiors,foundation:foundationAt(points,request.recipe),
          towers: local.map(point => square(point, radius * 0.14).map(toWorld)),
          ...(kind==='castle'?{keep:square({x: center.x - radius * 0.12, y: center.y - radius * 0.1}, radius * 0.33).map(toWorld)}:{}),
        };
      }
    }
    return null;
  }

  function rectangle(center, halfColumn, halfRow) {
    return [{row:center.row-halfRow,column:center.column-halfColumn}, {row:center.row-halfRow,column:center.column+halfColumn},
      {row:center.row+halfRow,column:center.column+halfColumn}, {row:center.row+halfRow,column:center.column-halfColumn}];
  }

  function createPublicSpaces(request, metric, streets,districts) {
    const plazas = [], landmarks = [];
    const used=new Set(),placedKinds=new Set(),character=request.character;
    const cores = request.recipe.urban.coreZones?.length ? request.recipe.urban.coreZones.slice()
      : [{...request.center, role:"central-core", radiusKm:request.recipe.urban.radiusKm * 0.22}];
    const civicSite=(x,y,role)=>({row:request.center.row+request.radiusRows*y,column:request.center.column+request.radiusColumns*x,
      role,radiusKm:request.recipe.urban.radiusKm*.2});
      const military=['fortress','border-pass'].includes(character.kind),maritime=['seaport','lake-port','river-port'].includes(character.kind);
    if(character.capital&&!cores.some(c=>c.role==='government-core'))cores.unshift(civicSite(0,-.36,'government-core'));
    if(character.plan==='axial')for(const core of cores)if(core.role==='government-core'){core.column=request.center.column;core.row=request.center.row-request.radiusRows*.36;}
    if(!cores.some(c=>c.role==='government-core'))cores.unshift(civicSite(-.18,-.12,'local-government'));
    if(request.recipe.culture?.holyReligion&&!cores.some(c=>c.role==='sacred-core'))cores.unshift(civicSite(.2,-.25,'sacred-core'));
    if(!character.small&&!military)cores.push(civicSite(.34,.1,'market-core'));
    if(!character.small&&request.population>7000&&!military)cores.push(civicSite(-.38,.24,'military-core'));
    if(!character.small&&request.population>3000&&!cores.some(c=>['sacred-core','pilgrimage-core'].includes(c.role)))
      cores.push({row:request.center.row-request.radiusRows*.3,column:request.center.column+request.radiusColumns*.28,
        role:'sacred-core',radiusKm:request.recipe.urban.radiusKm*.19});
    for (const [coreIndex,core] of cores.entries()) {
      const sacred=['sacred-core','pilgrimage-core'].includes(core.role),government=['government-core','local-government'].includes(core.role);
      const regime=request.recipe.culture?.government,royal=['bureaucratic-monarchy','court-monarchy','estate-monarchy','hereditary-feudalism'].includes(regime),
        imperial=/帝国|王朝/.test(request.recipe.culture?.stateName||''),holy=request.recipe.culture?.holyReligion,
        holyKind={'ancestral-rite':'holy-tomb','river-mysteries':'water-sanctuary','sky-law':'observatory-sanctuary',
          'mountain-vow':'pilgrimage-monastery','tide-covenant':'harbor-shrine','pilgrim-way':'great-sanctuary'}[holy?.tradition];
      const kind=character.mobile?'assembly-tent':sacred?(holyKind||'sanctuary'):government?(core.role==='government-core'?
        royal?imperial?'imperial-palace':'royal-palace':'council-house':regime==='hereditary-feudalism'?'lord-manor':'town-hall')
        :core.role==='military-core'?'barracks':core.role==='market-core'?'market-hall':character.kind==='oasis'?'caravanserai':maritime?'customs-house'
        :military?'barracks':coreIndex<2?'market-hall':'granary';
      if(placedKinds.has(kind))continue;
      const typeName={'imperial-palace':'皇宫','royal-palace':'王宫','council-house':'议会宫', 'holy-tomb':'先圣陵寝',
        'water-sanctuary':'圣水神殿','observatory-sanctuary':'天律观星院','pilgrimage-monastery':'圣山修院','harbor-shrine':'潮汐圣祠','great-sanctuary':'巡礼大圣殿',
        'assembly-tent':'议事大帐',sanctuary:character.tradition.temple, 'lord-manor':'领主府','town-hall':character.tradition.hall,caravanserai:'商旅驿馆',
        'customs-house':'海关商馆',barracks:'卫戍营', 'market-hall':'市集大厅',granary:'公仓'}[kind];
      const civicName=character.small&&government&&!character.capital?'议事屋':typeName;
      const candidates=districts.map((points,index)=>({points,index,center:polygonCentre(points.map(p=>({x:p.column,y:p.row})))}))
        .filter(p=>!used.has(p.index)).sort((a,b)=>Math.hypot(a.center.x-core.column,a.center.y-core.row)-Math.hypot(b.center.x-core.column,b.center.y-core.row));
      const palace=['royal-palace','imperial-palace'].includes(kind),design=character.palace;
      const parts=global.WorldAtlasCityCharacter.template(kind,character.style,request.population,design);
      const aspect=palace?design.aspect:[1.35,1],m=request.recipe.urban.gridCellKilometres;
      const frame=(parcel,half)=>{
        const center={row:parcel.center.y,column:parcel.center.x};
        let angle=request.recipe.localSite?.form.axisRadians||0,nearest=Infinity,target=null;
        for(const street of streets)if(street.kind!=='rail'&&!street.alley&&!street.palacePerimeter)for(let i=1;i<street.points.length;i++){
          const a=street.points[i-1],b=street.points[i],q=nearestPoint(center,a,b),distance=Math.hypot((q.column-center.column)*m.column,(q.row-center.row)*m.row);
          if(distance<nearest){nearest=distance;target=q;angle=Math.atan2((b.row-a.row)*m.row,(b.column-a.column)*m.column);}
        }
        if(target&&-(target.column-center.column)*m.column*Math.sin(angle)+(target.row-center.row)*m.row*Math.cos(angle)<0)angle+=Math.PI;
        const at=(x,y)=>({column:center.column+(Math.cos(angle)*x-Math.sin(angle)*y)*metric/m.column,
          row:center.row+(Math.sin(angle)*x+Math.cos(angle)*y)*metric/m.row});
        let corners=[[-1,-1],[1,-1],[1,1],[-1,1]];
        if(palace&&design.fortified)corners=[[-.84,-1],[.81,-1],[1,-.68],[1,1],[-1,1],[-1,-.71]];
        const square=corners.map(([x,y])=>at(x*aspect[0]*half,y*aspect[1]*half));
        return {center,half,square,index:parcel.index,at,angle,gate:at(0,aspect[1]*half)};
      };
      const componentPoints=chosen=>parts.map(part=>part.points.map(([x,y])=>chosen.at(x/1.35*aspect[0]*chosen.half,y*aspect[1]*chosen.half)));
      const fitsGround=chosen=>(!request.recipe.harbor?.reservedLand||!polygonsOverlap(chosen.square,request.recipe.harbor.reservedLand))
        &&chosen.square.every((p,i)=>segmentOnLand(p,chosen.square[(i+1)%chosen.square.length],request.recipe))
        &&componentPoints(chosen).every(points=>points.every(p=>segmentOnLand(p,p,request.recipe))&&foundationAt(points,request.recipe).reliefMetres<=4);
      let chosen=null;
      if(palace&&character.capital&&request.population>=30000){
        const groundsKm2=Math.min(2.4,.04+request.population*(imperial?3e-6:2e-6)),desired=Math.sqrt(groundsKm2/(4*aspect[0]*aspect[1]))/metric;
        for(const parcel of candidates.slice(0,36)){
          for(const factor of [1,.85,.7,.55]){
            const candidate=frame(parcel,desired*factor),{half,square,at}=candidate,
              interior=Array.from({length:49},(_,i)=>at(((i%7)/6*2-1)*half*aspect[0],(Math.floor(i/7)/6*2-1)*half*aspect[1]));
            if(!interior.every(p=>districts.some(d=>containsPoint(d.map(q=>({x:q.column,y:q.row})),{x:p.column,y:p.row})))
              ||!fitsGround(candidate)
              ||plazas.some(p=>polygonsOverlap(square,p.points))
              ||streets.some(s=>s.authoritative&&polygonNearRoute(square,s.points,streetWidthsInGrid(request.recipe,request.radiusRows,request.radiusColumns).arterial)))continue;
            chosen={...candidate,precinct:true};break;
          }if(chosen)break;
        }
      }
      for(const parcel of chosen?[]:candidates){
        const palaceHalf=Math.sqrt(Math.min(2.4,.04+request.population*(imperial?3e-6:2e-6))/(4*aspect[0]*aspect[1]));
        let half=Math.min(character.small?.032:government&&character.capital?palaceHalf:sacred&&holy?.18:.12,Math.max(.025,core.radiusKm*(government&&character.capital?.4:.28)))/metric;
        for(let i=0;i<8;i++,half*=.8){
          const candidate=frame(parcel,half),{square}=candidate;
          if(!square.every(p=>containsPoint(parcel.points.map(q=>({x:q.column,y:q.row})),{x:p.column,y:p.row}))
            ||!fitsGround(candidate)
            ||plazas.some(p=>polygonsOverlap(square,p.points))
            ||streets.some(street=>polygonNearRoute(square,street.points,streetWidthsInGrid(request.recipe,request.radiusRows,request.radiusColumns).collector)))continue;
          chosen={...candidate,precinct:palace};break;
        }
        if(chosen)break;
      }
      if(!chosen)continue;
      const {center,half,square,at}=chosen;used.add(chosen.index);placedKinds.add(kind);
      const name=character.prefix+(kind==='market-hall'?'市集广场':civicName);
      const gateHalf=streetWidthsInGrid(request.recipe,request.radiusRows,request.radiusColumns).arterial*1.2;
      const wallPieces=square.flatMap((p,i)=>{
        const z=square[(i+1)%square.length],q=nearestPoint(chosen.gate,p,z);
        if(Math.hypot(q.row-chosen.gate.row,q.column-chosen.gate.column)>1e-8)return [[p,z]];
        const len=Math.hypot(z.column-p.column,z.row-p.row),delta=Math.min(.18,gateHalf/len),t=Math.hypot(q.column-p.column,q.row-p.row)/len;
        const at=v=>({column:p.column+(z.column-p.column)*v,row:p.row+(z.row-p.row)*v});
        return [[p,at(t-delta)],[at(t+delta),z]];
      });
      plazas.push({points:square,role:core.role,name,market:kind==='market-hall',precinct:chosen.precinct,
        walls:wallPieces,gate:chosen.gate,design:palace?design.family:null});
      if(chosen.precinct){
        // Reserve the entire civic compound before walls, houses and lanes.
        const old=streets.splice(0);streets.push(...old.flatMap(street=>streetOutsidePolygon(street,square)));
        const ring=square.map(p=>({column:center.column+(p.column-center.column)*1.04,row:center.row+(p.row-center.row)*1.04}));
        for(let i=0;i<ring.length;i++)streets.push({kind:'collector',points:[ring[i],ring[(i+1)%ring.length]],access:true,palacePerimeter:true});
      }
      let access = null;
      for (const street of streets) if (street.kind !== "rail") for (let index=1;index<street.points.length;index++) {
        const target=nearestPoint(chosen.precinct?chosen.gate:center,street.points[index-1],street.points[index]);
        const distance=Math.hypot(target.column-center.column,target.row-center.row);
        const entrance=chosen.precinct?chosen.gate:square.map((p,i)=>nearestPoint(target,p,square[(i+1)%square.length])).sort((a,b)=>
          Math.hypot(a.row-target.row,a.column-target.column)-Math.hypot(b.row-target.row,b.column-target.column))[0];
        if((!access || distance<access.distance) && segmentOnLand(entrance,target,request.recipe))access={points:[entrance,target],distance};
      }
      if(access && access.distance<Math.max(request.radiusRows,request.radiusColumns)*0.5)
        streets.push({kind:"collector",points:access.points,access:true,plaza:true});
      for(const [index,part] of parts.entries()){
        const points=componentPoints(chosen)[index];
        const roof=part.roof==='dome'?[points.map((p,i)=>i%4===0?[center,p]:null).filter(Boolean)].flat():roofLines(points);
        const fill=palace?character.material.palace[index%3]:character.roof[index%character.roof.length];
        landmarks.push({points,kind:index===0?kind:kind+'-wing',fill,roofLines:roof,roof:part.roof,design:palace?design.family:null,foundation:foundationAt(points,request.recipe),
          compoundId:'compound-'+coreIndex,compoundPoints:square,name:character.prefix+civicName});
      }
    }
    return {plazas,landmarks};
  }


  function foundationAt(points,recipe){
    const f=recipe.localSite?.surface;if(!f)return {elevationMetres:0,reliefMetres:0};
    const heights=points.map(p=>global.WorldAtlasCitySite.elevationAt(f,p));
    return {elevationMetres:Math.max(...heights),reliefMetres:Math.max(...heights)-Math.min(...heights)};
  }

  function chamferPolygon(points, fraction) {
    return points.flatMap((point,index)=>{
      const previous=points[(index+points.length-1)%points.length],next=points[(index+1)%points.length];
      return [{x:point.x+(previous.x-point.x)*fraction,y:point.y+(previous.y-point.y)*fraction},
        {x:point.x+(next.x-point.x)*fraction,y:point.y+(next.y-point.y)*fraction}];
    });
  }

  function roofLines(points, courtyard) {
    const polygon=points.map(point=>({x:point.column,y:point.row}));
    if(courtyard){
      const ring=insetPolygon(polygon,Math.sqrt(polygonArea(polygon))*0.10).map(point=>({column:point.x,row:point.y}));
      return ring.length>=3?[ring.concat(ring[0])]:[];
    }
    const center=polygonCentre(polygon);
    let direction=null,longest=0;
    for(let index=0;index<polygon.length;index++){
      const a=polygon[index],b=polygon[(index+1)%polygon.length],dx=b.x-a.x,dy=b.y-a.y,length=Math.hypot(dx,dy);
      if(length>longest){longest=length;direction={x:dx/length,y:dy/length};}
    }
    if(!direction)return [];
    const intersections=[];
    for(let index=0;index<polygon.length;index++){
      const a=polygon[index],b=polygon[(index+1)%polygon.length],dx=b.x-a.x,dy=b.y-a.y;
      const denominator=direction.x*dy-direction.y*dx;
      if(Math.abs(denominator)<1e-12)continue;
      const t=((a.x-center.x)*dy-(a.y-center.y)*dx)/denominator;
      const u=((a.x-center.x)*direction.y-(a.y-center.y)*direction.x)/denominator;
      if(u>=-1e-9&&u<=1+1e-9)intersections.push(t);
    }
    if(intersections.length<2)return [];
    const minimum=Math.min(...intersections),maximum=Math.max(...intersections),inset=(maximum-minimum)*0.15;
    return [[minimum+inset,maximum-inset].map(t=>({column:center.x+direction.x*t,row:center.y+direction.y*t}))];
  }

  function wallTowers(walls, request, metric, streets) {
    const towers=[],seen=new Set(),radius=Math.max(0.004,Math.min(0.008,request.recipe.urban.radiusKm*0.01))/metric;
    const spacing=Math.max(0.045,Math.min(0.09,request.recipe.urban.radiusKm*0.1))/metric;
    for(const wall of walls){
      const a=wall[0],b=wall[1],length=Math.hypot(b.column-a.column,b.row-a.row),count=Math.max(1,Math.ceil(length/spacing));
      for(let index=0;index<=count;index++){
        const center={row:a.row+(b.row-a.row)*index/count,column:a.column+(b.column-a.column)*index/count};
        const key=[Math.round((center.row-request.center.row)/radius),Math.round((center.column-request.center.column)/radius)].join(",");
        if(seen.has(key))continue;seen.add(key);
        const points=rectangle(center,radius,radius);
        if(streets.some(street=>street.kind!=="rail"&&polygonNearRoute(points,street.points,
          (street.kind==="arterial"?streetWidthsInGrid(request.recipe,request.radiusRows,request.radiusColumns).arterial:
            streetWidthsInGrid(request.recipe,request.radiusRows,request.radiusColumns).local)*0.55)))continue;
        for(const land of clipToLand(points,request.recipe))towers.push({points:land,roofLines:roofLines(land)});
      }
    }
    return towers;
  }

  function countryHouses(request, corridors, districts, metric, widths) {
    const houses=[],lanes=[],hamlets=[],seen=[];
    for(const [corridorIndex,corridor] of corridors.entries())if(corridor.kind!=="rail")for(let segment=1;segment<corridor.points.length;segment++){
      const a=corridor.points[segment-1],b=corridor.points[segment],dx=b.column-a.column,dy=b.row-a.row,length=Math.hypot(dx,dy);
      if(!length)continue;
      const count=Math.min(40,Math.max(1,Math.ceil(length*metric/0.23)));
      for(let index=0;index<count;index++){
        const channel=48000+corridorIndex*7919+segment*300+index, fraction=(index+0.2+random(request.seed,channel)*0.6)/count;
        const half=(0.008+random(request.seed,channel+1)*0.006)/metric, side=random(request.seed,channel+2)>.5?1:-1;
        const offset=widths.arterial*1.8+half*2.8;
        const center={row:a.row+dy*fraction+dx/length*offset*side,column:a.column+dx*fraction-dy/length*offset*side};
        const radial=Math.hypot((center.row-request.center.row)/request.radiusRows,(center.column-request.center.column)/request.radiusColumns);
        if(radial<0.85||radial>3.7||seen.some(p=>Math.hypot(p.row-center.row,p.column-center.column)*metric<.19))continue;
        const points=rectangle(center,half*(1+random(request.seed,channel+3)*0.5),half*0.7);
        if(districts.some(district=>polygonsOverlap(points,district)))continue;
        if(corridors.some(route=>polygonNearRoute(points,route.points,widths.arterial*0.8)))continue;
        if(!points.every(p=>segmentOnLand(center,p,request.recipe)))continue;
        const access=nearestPoint(center,a,b);if(!segmentOnLand(center,access,request.recipe))continue;
        const cluster=[];
        const spine=[],paths=[],localHouses=[],alongUnit={column:dx/length,row:dy/length},normal={column:-dy/length,row:dx/length};
        for(let home=0;home<(radial<1.6?8:5);home++){
          const along=(home%4-1.5)*half*(3.1+random(request.seed,channel+home+21)*.7),offset=(Math.floor(home/4)*2-1)*half*(2.1+random(request.seed,channel+home+31)*.9);
          const c={row:center.row+dy/length*along+dx/length*offset,column:center.column+dx/length*along-dy/length*offset};
          const footprint=[[-1,-.72],[1,-.72],[1,.72],[-1,.72]].map(([x,y])=>({column:c.column+alongUnit.column*x*half+normal.column*y*half,
            row:c.row+alongUnit.row*x*half+normal.row*y*half}));
          if(!footprint.every(p=>segmentOnLand(c,p,request.recipe))||!segmentOnLand(c,center,request.recipe)
            ||foundationAt(footprint,request.recipe).reliefMetres>3.5
            ||districts.some(d=>polygonsOverlap(footprint,d))||corridors.some(r=>polygonNearRoute(footprint,r.points,widths.arterial*1.2)))continue;
          const junction={row:c.row-dx/length*offset,column:c.column+dy/length*offset};
          const entrance=footprint.map((p,i)=>nearestPoint(junction,p,footprint[(i+1)%footprint.length])).sort((a,b)=>
            Math.hypot(a.column-junction.column,a.row-junction.row)-Math.hypot(b.column-junction.column,b.row-junction.row))[0];
          if(!segmentOnLand(entrance,junction,request.recipe))continue;
          localHouses.push({points:footprint,roofLines:roofLines(footprint),roof:'hip',fill:request.character.roof[Math.floor(random(request.seed,channel+home+102)*request.character.roof.length)],
            entrance,foundation:foundationAt(footprint,request.recipe)});cluster.push(...footprint);
          spine.push(junction);paths.push([entrance,junction]);
        }
        if(cluster.length){
          spine.push(center);spine.sort((a,b)=>(a.column-b.column)*alongUnit.column+(a.row-b.row)*alongUnit.row);
          const backbone=[spine[0],spine.at(-1)];
          if(!segmentOnLand(...backbone,request.recipe))continue;
          houses.push(...localHouses);lanes.push(...paths,backbone,[center,access]);seen.push(center);
          if(radial>1.65&&hamlets.length<8&&request.recipe.localSite?.nameRoots?.length){const roots=request.recipe.localSite.nameRoots,root=roots[(hamlets.length+Math.floor(random(request.seed,4180)*20))%roots.length];
            hamlets.push({points:cluster,name:root+['庄','村','屯'][hamlets.length%3]});}
        }
      }
    }
    return {houses,lanes,hamlets};
  }

  const polygonBoundsCache=new WeakMap();
  function polygonBounds(points){
    let b=polygonBoundsCache.get(points);if(!b){b={west:Math.min(...points.map(p=>p.column)),east:Math.max(...points.map(p=>p.column)),
      north:Math.min(...points.map(p=>p.row)),south:Math.max(...points.map(p=>p.row))};polygonBoundsCache.set(points,b);}return b;
  }
  function polygonsOverlap(first, second) {
    const a=polygonBounds(first),b=polygonBounds(second);
    if(a.west>=b.east||a.east<=b.west||a.north>=b.south||a.south<=b.north)return false;
    let polygon = first.map(point => ({x: point.column, y: point.row}));
    for (let index = 0; index < second.length; index++) {
      const a = second[index], b = second[(index + 1) % second.length];
      const nx = b.row - a.row, ny = a.column - b.column;
      polygon = clipHalfPlane(polygon, nx, ny, a.column * nx + a.row * ny);
    }
    return polygon.length >= 3 && polygonArea(polygon) > 1e-12;
  }

  function wallSegmentsWithGates(a, b, streets, width,gates,request) {
    const dx = b.column - a.column, dy = b.row - a.row, length = Math.hypot(dx, dy);
    const openings = [];
    for (const street of streets) if (street.kind !== "rail") for (let index = 1; index < street.points.length; index++) {
      const c = street.points[index - 1], d = street.points[index];
      const sx = d.column - c.column, sy = d.row - c.row, denominator = dx * sy - dy * sx;
      if (Math.abs(denominator) < 1e-14) continue;
      const qx = c.column - a.column, qy = c.row - a.row;
      const t = (qx * sy - qy * sx) / denominator, u = (qx * dy - qy * dx) / denominator;
      if (t >= -1e-8 && t <= 1 + 1e-8 && u >= -1e-8 && u <= 1 + 1e-8){
        openings.push([Math.max(0,t-width/length),Math.min(1,t+width/length)]);
        const center={row:a.row+dy*t,column:a.column+dx*t};
        if(gates&&!gates.some(g=>Math.hypot(g.center.row-center.row,g.center.column-center.column)<width*4)){
          const vector={x:dx/length,y:dy/length},half=width*.48;
          const towers=[-1,1].map(side=>{const c={row:center.row+vector.y*width*1.38*side,column:center.column+vector.x*width*1.38*side};
            return [[-1,-1],[1,-1],[1,1],[-1,1]].map(([x,y])=>({row:c.row+(vector.y*x+vector.x*y)*half,column:c.column+(vector.x*x-vector.y*y)*half}));});
          const ox=center.column-request.center.column,oy=center.row-request.center.row;
          const direction=Math.abs(ox)>Math.abs(oy)?ox>0?'东':'西':oy>0?'南':'北';
          if(towers.every(points=>points.every(p=>segmentOnLand(p,p,request.recipe))))gates.push({center,towers,
            name:direction+(['fortress','pass'].includes(request.recipe.siteType)?'关门':'城门'),passage:[-1,1].map(side=>({row:center.row+vector.x*width*side,column:center.column-vector.y*width*side}))});
        }
      }
    }
    openings.sort((first, second) => first[0] - second[0]);
    const result = [];
    let cursor = 0;
    const span = (start, end) => [start, end].map(t => ({row: a.row + dy * t, column: a.column + dx * t}));
    for (const [start, end] of openings) { if (start > cursor) result.push(span(cursor, start)); cursor = Math.max(cursor, end); }
    if (cursor < 1) result.push(span(cursor, 1));
    return result;
  }

  function polygonArea(points) {
    if(points.length<3)return 0;
    const origin=points[0];
    return Math.abs(points.reduce((sum, point, index) => { const next = points[(index + 1) % points.length];
      return sum+(point.x-origin.x)*(next.y-origin.y)-(next.x-origin.x)*(point.y-origin.y); }, 0)) * 0.5;
  }

  function polygonCentre(points) {
    return {x: points.reduce((sum, point) => sum + point.x, 0) / points.length, y: points.reduce((sum, point) => sum + point.y, 0) / points.length};
  }

  function clipHalfPlane(points, nx, ny, distance) {
    const result = [];
    for (let index = 0; index < points.length; index++) {
      const first = points[index], second = points[(index + 1) % points.length];
      const a = first.x * nx + first.y * ny - distance, b = second.x * nx + second.y * ny - distance;
      if (a <= 1e-12) result.push(first);
      if ((a < 0 && b > 0) || (a > 0 && b < 0)) {
        const t = a / (a - b);
        result.push({x: first.x + (second.x - first.x) * t, y: first.y + (second.y - first.y) * t});
      }
    }
    return result;
  }

  function bisectParcel(points, seed, channel, chaos) {
    const xs = points.map(point => point.x), ys = points.map(point => point.y);
    const horizontal = Math.max(...xs) - Math.min(...xs) >= Math.max(...ys) - Math.min(...ys);
    const angle = (horizontal ? 0 : Math.PI / 2) + (random(seed, channel + 1) - 0.5) * chaos;
    const nx = Math.cos(angle), ny = Math.sin(angle);
    const projections = points.map(point => point.x * nx + point.y * ny);
    const low = Math.min(...projections), high = Math.max(...projections);
    const distance = low + (high - low) * (0.38 + random(seed, channel + 2) * 0.24);
    const first = clipHalfPlane(points, nx, ny, distance), second = clipHalfPlane(points, -nx, -ny, -distance);
    if (first.length < 3 || second.length < 3 || Math.min(polygonArea(first), polygonArea(second)) < 1e-10) return null;
    const cut = first.filter(point => Math.abs(point.x * nx + point.y * ny - distance) < 1e-9);
    return cut.length >= 2 ? {first, second, cut: [cut[0], cut[cut.length - 1]]} : null;
  }

  function insetPolygon(points, margin) {
    let result = points;
    for (let index = 0; index < points.length && result.length >= 3; index++) {
      const a = points[index], b = points[(index + 1) % points.length];
      const length = Math.hypot(b.x - a.x, b.y - a.y);
      if (length < 1e-12) continue;
      const nx = (b.y - a.y) / length, ny = (a.x - b.x) / length;
      result = clipHalfPlane(result, nx, ny, a.x * nx + a.y * ny - margin);
    }
    return result;
  }

  function clipToLand(points, recipe) {
    if(recipe.localSite)return clipToLocalLand(points.map(p=>({x:p.column,y:p.row})),recipe).map(polygon=>polygon.map(p=>({column:p.x,row:p.y})));
    const local = points.map(point => ({x: point.column, y: point.row}));
    const result = [];
    for (const [row, column] of recipe.terrain.buildable.allowedCells) {
      let polygon = clipHalfPlane(local, -1, 0, -column);
      polygon = clipHalfPlane(polygon, 1, 0, column + 1);
      polygon = clipHalfPlane(polygon, 0, -1, -row);
      polygon = clipHalfPlane(polygon, 0, 1, row + 1);
      if (polygon.length >= 3 && polygonArea(polygon) > 1e-12) result.push(...clipToLocalLand(polygon,recipe)
        .map(piece=>piece.map(point => ({row: point.y, column: point.x}))));
    }
    return result;
  }

  function clipRectangle(points, west, north, east, south) {
    let result=clipHalfPlane(points,-1,0,-west);
    result=clipHalfPlane(result,1,0,east);
    result=clipHalfPlane(result,0,-1,-north);
    return clipHalfPlane(result,0,1,south);
  }

  // Cut a forbidden convex area into disjoint convex remnants. Streets,
  // buildings and public squares share the same stream and slope mask.
  function subtractPlanes(polygon, planes) {
    const outside=[];let inside=polygon;
    for(const [nx,ny,distance] of planes){
      const piece=clipHalfPlane(inside,-nx,-ny,-distance);
      if(piece.length>=3&&polygonArea(piece)>1e-12)outside.push(piece);
      inside=clipHalfPlane(inside,nx,ny,distance);
      if(inside.length<3)break;
    }
    return outside;
  }

  function clipToLocalLand(polygon, recipe) {
    const site=recipe.localSite;if(!site)return [polygon];
    let pieces=[polygon];const surface=site.surface;
    if(surface){
      const {west,north,east,south}=surface.bounds,columns=surface.columns-1,rows=surface.rows-1;
      const dx=(east-west)/columns,dy=(south-north)/rows;
      const xs=polygon.map(point=>point.x),ys=polygon.map(point=>point.y);
      const minimumColumn=Math.max(0,Math.floor((Math.min(...xs)-west)/dx)),maximumColumn=Math.min(columns-1,Math.floor((Math.max(...xs)-west)/dx));
      const minimumRow=Math.max(0,Math.floor((Math.min(...ys)-north)/dy)),maximumRow=Math.min(rows-1,Math.floor((Math.max(...ys)-north)/dy));
      const allAllowed=polygon.every(p=>surfaceAllows({column:p.x,row:p.y},surface,.20));
      if(!allAllowed){
        pieces=[];
        for(let row=minimumRow;row<=maximumRow;row++)for(let column=minimumColumn;column<=maximumColumn;column++){
        const index=row*surface.columns+column;
        for(const indices of [[index,index+1,index+surface.columns+1],[index,index+surface.columns+1,index+surface.columns]]){
          const triangle=indices.map(i=>({x:west+(i%surface.columns)*dx,y:north+Math.floor(i/surface.columns)*dy,
            value:Math.min(surface.elevationMetres[i],(.20-surface.slopes[i])*1000,-(surface.lakeDepths?.[i]??-1000))})),dry=[];
          for(let edge=0;edge<3;edge++){const a=triangle[edge],b=triangle[(edge+1)%3];
            if(a.value>0)dry.push(a);if((a.value>0)!==(b.value>0)){const t=a.value/(a.value-b.value);dry.push({x:a.x+(b.x-a.x)*t,y:a.y+(b.y-a.y)*t});}}
          if(dry.length<3)continue;let piece=polygon;
          for(let edge=0;edge<dry.length;edge++){const a=dry[edge],b=dry[(edge+1)%dry.length],nx=b.y-a.y,ny=a.x-b.x;piece=clipHalfPlane(piece,nx,ny,a.x*nx+a.y*ny);}
          if(piece.length>=3&&polygonArea(piece)>1e-12)pieces.push(piece);
        }
        }
      }
    }
    const metric=gridCellKilometres(recipe,recipe.urban.radiusRows,recipe.urban.radiusColumns);
    for(const channel of site.channels || [])for(let index=1;index<channel.points.length;index++){
      const a=channel.points[index-1],b=channel.points[index],dx=b.column-a.column,dy=b.row-a.row,length=Math.hypot(dx,dy);
      if(!length)continue;
      const clearance=(Math.max(a.widthMetres,b.widthMetres)*0.5+3)/(metric*1000);
      const planes=[[dy/length,-dx/length,(a.column*dy-a.row*dx)/length+clearance],
        [-dy/length,dx/length,(-a.column*dy+a.row*dx)/length+clearance],
        [-dx/length,-dy/length,(-a.column*dx-a.row*dy)/length+clearance],
        [dx/length,dy/length,(b.column*dx+b.row*dy)/length+clearance]];
      pieces=pieces.flatMap(piece=>{
        if(piece.every(point=>point.x<Math.min(a.column,b.column)-clearance)||piece.every(point=>point.x>Math.max(a.column,b.column)+clearance)
          ||piece.every(point=>point.y<Math.min(a.row,b.row)-clearance)||piece.every(point=>point.y>Math.max(a.row,b.row)+clearance))return [piece];
        return subtractPlanes(piece,planes);
      });
    }
    return pieces;
  }

  function nearestPoint(point, a, b) {
    const dx = b.column - a.column, dy = b.row - a.row;
    const t = Math.max(0, Math.min(1, ((point.column - a.column) * dx + (point.row - a.row) * dy) / (dx * dx + dy * dy || 1)));
    return {column: a.column + dx * t, row: a.row + dy * t};
  }

  function appendFortress(geometry, cityPlan) {
    const fort = cityPlan.fortress;
    const group = svg("g", {class: "city-fortress", "data-fortress-era": fort.era,'data-fortress-kind':fort.kind});
    group.appendChild(svg("path", {d: pathData(fort.points, true), fill: "#e9e2cf"}));
    group.appendChild(svg("path", {d: fort.walls.map(points => pathData(points, false)).join(" "), fill: "none", stroke: "#645547", "stroke-width": cityPlan.streetWidths.collector * 2.3}));
    group.appendChild(svg("path", {d: fort.towers.map(points => pathData(points, true)).join(" "), fill: "#71614f", stroke: "#f5ead3", "stroke-width": cityPlan.streetWidths.local * 0.4}));
    if(fort.kind==='castle')group.appendChild(svg("path", {d: pathData(fort.keep, true), fill: cityPlan.character.roof[0], stroke: "#76634d", "stroke-width": cityPlan.streetWidths.local}));
    else {
      for(const building of fort.interiors)group.appendChild(svg('path',{d:pathData(building.points,true),fill:cityPlan.character.roof[0],stroke:'#857563',
        'stroke-width':.55,'vector-effect':'non-scaling-stroke','data-fort-building':building.kind}));
      group.appendChild(svg('path',{d:fort.interiors.flatMap(b=>roofLines(b.points)).map(p=>pathData(p,false)).join(' '),fill:'none',stroke:'#f4ead6',
        'stroke-width':.6,'vector-effect':'non-scaling-stroke'}));
    }
    geometry.appendChild(group);
  }

  function containsPoint(polygon, point) {
    return polygon.every((a, index) => { const b = polygon[(index + 1) % polygon.length]; return (b.x - a.x) * (point.y - a.y) - (b.y - a.y) * (point.x - a.x) >= -1e-12; });
  }

  function clipStreetToLand(street, recipe) {
    if(recipe.localSite)return street.points.slice(1).flatMap((point,index)=>clipStreetToLocalLand({...street,points:[street.points[index],point]},recipe));
    const result = [];
    for (let index = 1; index < street.points.length; index++) {
      const a = street.points[index - 1], b = street.points[index];
      for (const [row, column] of recipe.terrain.buildable.allowedCells) {
        let minimum = 0, maximum = 1;
        for (const [start, delta, low, high] of [[a.row, b.row - a.row, row, row + 1], [a.column, b.column - a.column, column, column + 1]]) {
          if (Math.abs(delta) < 1e-12) { if (start < low || start > high) maximum = -1; continue; }
          const t1 = (low - start) / delta, t2 = (high - start) / delta;
          minimum = Math.max(minimum, Math.min(t1, t2)); maximum = Math.min(maximum, Math.max(t1, t2));
        }
        if (maximum - minimum > 1e-10) result.push(...clipStreetToLocalLand({...street, points: [minimum, maximum].map(t => ({row: a.row + (b.row - a.row) * t, column: a.column + (b.column - a.column) * t}))},recipe));
      }
    }
    return result;
  }

  function streetCrossingsValid(street,recipe){
    const site=recipe.localSite;if(!site)return true;const m=recipe.urban.gridCellKilometres;
    for(let i=1;i<street.points.length;i++)for(const channel of site.channels)for(let j=1;j<channel.points.length;j++){
      const a=street.points[i-1],b=street.points[i],c=channel.points[j-1],d=channel.points[j],dx=b.column-a.column,dy=b.row-a.row,sx=d.column-c.column,sy=d.row-c.row,den=dx*sy-dy*sx;
      if(Math.abs(den)<1e-12)continue;const t=((c.column-a.column)*sy-(c.row-a.row)*sx)/den,u=((c.column-a.column)*dy-(c.row-a.row)*dx)/den;
      if(t<0||t>1||u<0||u>1)continue;const p={column:a.column+dx*t,row:a.row+dy*t};
      if(!(site.bridges||[]).some(bridge=>Math.hypot((p.column-bridge.center.column)*m.column,(p.row-bridge.center.row)*m.row)<.018))return false;
    }return true;
  }

  function clipStreetToLocalLand(street, recipe) {
    if(!recipe.localSite)return [street];
    const a=street.points[0],b=street.points[1],dx=b.column-a.column,dy=b.row-a.row;
    let intervals=[[0,1]];
    const remove=planes=>{
      let minimum=0,maximum=1;
      for(const [nx,ny,distance] of planes){
        const start=a.column*nx+a.row*ny,delta=dx*nx+dy*ny;
        if(Math.abs(delta)<1e-12){if(start>distance+1e-12)return;continue;}
        const fraction=(distance-start)/delta;
        if(delta>0)maximum=Math.min(maximum,fraction);else minimum=Math.max(minimum,fraction);
      }
      if(maximum-minimum<1e-10)return;
      intervals=intervals.flatMap(([first,last])=>{
        if(maximum<=first||minimum>=last)return [[first,last]];
        const pieces=[];if(minimum-first>1e-10)pieces.push([first,Math.min(last,minimum)]);
        if(last-maximum>1e-10)pieces.push([Math.max(first,maximum),last]);return pieces;
      });
    };
    const surface=recipe.localSite.surface;
    if(surface){
      const cell=Math.min((surface.bounds.east-surface.bounds.west)/(surface.columns-1),(surface.bounds.south-surface.bounds.north)/(surface.rows-1)),
        steps=Math.max(2,Math.ceil(Math.hypot(dx,dy)/cell*5)),valid=[],at=t=>({column:a.column+dx*t,row:a.row+dy*t});
      let previous=surfaceAllows(a,surface,.20),start=0;
      for(let i=1;i<=steps;i++){
        const t=i/steps,allowed=surfaceAllows(at(t),surface,.20);
        if(allowed!==previous){let low=(i-1)/steps,high=t;for(let j=0;j<18;j++){
          const middle=(low+high)/2;if(surfaceAllows(at(middle),surface,.20)===previous)low=middle;else high=middle;}
          const edge=(low+high)/2;if(previous)valid.push([start,edge]);else start=edge;
        }previous=allowed;
      }if(previous)valid.push([start,1]);
      intervals=intervals.flatMap(([a,b])=>valid.map(([c,d])=>[Math.max(a,c),Math.min(b,d)]).filter(([a,b])=>b-a>1e-10));
    }
    const metric=gridCellKilometres(recipe,recipe.urban.radiusRows,recipe.urban.radiusColumns);
    for(const channel of recipe.localSite.channels || [])for(let index=1;index<channel.points.length;index++){
      const first=channel.points[index-1],last=channel.points[index],sx=last.column-first.column,sy=last.row-first.row,length=Math.hypot(sx,sy);
      if(!length)continue;const clearance=(Math.max(first.widthMetres,last.widthMetres)*0.5+3)/(metric*1000);
      remove([[sy/length,-sx/length,(first.column*sy-first.row*sx)/length+clearance],
        [-sy/length,sx/length,(-first.column*sy+first.row*sx)/length+clearance],
        [-sx/length,-sy/length,(-first.column*sx-first.row*sy)/length+clearance],
        [sx/length,sy/length,(last.column*sx+last.row*sy)/length+clearance]]);
    }
    for(const bridge of recipe.localSite.bridges||[])if(bridge.routeId===street.id||routeNear(bridge.center,[a,b],.02/metric)){
      const fractions=bridge.points.map(p=>((p.column-a.column)*dx+(p.row-a.row)*dy)/(dx*dx+dy*dy));
      const low=Math.max(0,Math.min(...fractions)),high=Math.min(1,Math.max(...fractions));
      if(high>low)intervals.push([low,high]);
    }
    intervals.sort((a,b)=>a[0]-b[0]);
    const connected=[];for(const interval of intervals){const last=connected.at(-1);
      if(last&&interval[0]<=last[1]+1e-9)last[1]=Math.max(last[1],interval[1]);else connected.push(interval);}
    return connected.map(([minimum,maximum])=>({...street,points:[minimum,maximum].map(t=>({row:a.row+dy*t,column:a.column+dx*t}))}));
  }

  function routeNear(point, points, clearance) {
    return points.slice(1).some((end, index) => { const near = nearestPoint(point, points[index], end); return Math.hypot(point.row - near.row, point.column - near.column) < clearance; });
  }

  function polygonNearRoute(polygon, points, clearance) {
    const bounds=polygonBounds(polygon);
    for (let index = 1; index < points.length; index++) {
      const start = points[index - 1], end = points[index];
      if(bounds.south<Math.min(start.row,end.row)-clearance||bounds.north>Math.max(start.row,end.row)+clearance
        ||bounds.east<Math.min(start.column,end.column)-clearance||bounds.west>Math.max(start.column,end.column)+clearance)continue;
      const dx = end.column - start.column, dy = end.row - start.row;
      const length = Math.hypot(dx, dy);
      if (!length) continue;
      // Intersect the building with the finite expanded transport corridor.
      let clipped = polygon.map(point => ({x: point.column-start.column, y: point.row-start.row}));
      for (const [nx, ny, distance] of [
        [dy / length, -dx / length, clearance],
        [-dy / length, dx / length, clearance],
        [-dx / length, -dy / length, clearance],
        [dx / length, dy / length, length + clearance],
      ]) clipped = clipHalfPlane(clipped, nx, ny, distance);
      if (polygonArea(clipped)>1e-14) return true;
    }
    return false;
  }

  function segmentOnLand(start, end, recipe) {
    const allowed = new Set(recipe.terrain.buildable.allowedCells.map(cell => cell.join(",")));
    const surface=recipe.localSite?.surface;
    const cellSize=surface?Math.min((surface.bounds.east-surface.bounds.west)/(surface.columns-1),
      (surface.bounds.south-surface.bounds.north)/(surface.rows-1)):1;
    const steps = Math.max(2, Math.ceil(Math.hypot(end.row - start.row, end.column - start.column) / cellSize * 2));
    for (let index = 0; index <= steps; index++) {
      const t = index / steps;
      const row=start.row+(end.row-start.row)*t,column=start.column+(end.column-start.column)*t;
      if (!surface&&!allowed.has(Math.floor(row) + "," + Math.floor(column))) return false;
      if(surface&&!surfaceAllows({row,column},surface,.20))return false;
    }
    for(const channel of recipe.localSite?.channels || []){
      const clearance=(channel.widthMetres*0.5+3)/(gridCellKilometres(recipe,recipe.urban.radiusRows,recipe.urban.radiusColumns)*1000);
      if(polygonNearRoute(rectangle(start,1e-8,1e-8),channel.points,clearance)
        ||polygonNearRoute(rectangle(end,1e-8,1e-8),channel.points,clearance))return false;
      for(let index=1;index<channel.points.length;index++){
        const a=channel.points[index-1],b=channel.points[index];
        const dx=end.column-start.column,dy=end.row-start.row,sx=b.column-a.column,sy=b.row-a.row;
        const denominator=dx*sy-dy*sx;if(Math.abs(denominator)<1e-12)continue;
        const qx=a.column-start.column,qy=a.row-start.row,t=(qx*sy-qy*sx)/denominator,u=(qx*dy-qy*dx)/denominator;
        if(t>=0&&t<=1&&u>=0&&u<=1)return false;
      }
    }
    return true;
  }

  function surfaceAllows(p,f,maxSlope){
    const b=f.bounds;if(p.column<b.west||p.column>b.east||p.row<b.north||p.row>b.south)return false;
    const x=(p.column-b.west)/(b.east-b.west)*(f.columns-1),y=(p.row-b.north)/(b.south-b.north)*(f.rows-1),
      i=Math.min(f.columns-2,Math.floor(x)),j=Math.min(f.rows-2,Math.floor(y)),u=x-i,v=y-j,k=j*f.columns+i;
    const value=a=>(a[k]*(1-u)+a[k+1]*u)*(1-v)+(a[k+f.columns]*(1-u)+a[k+f.columns+1]*u)*v;
    return global.WorldAtlasCitySite.isDry(f,p)&&value(f.slopes)<maxSlope;
  }

  function streetWidthsInGrid(recipe, radiusRows, radiusColumns) {
    const declared = recipe.urban && recipe.urban.streetWidthsMetres || {};
    const fallback = { arterial: 12, collector: 5.5, local: 2.8 };
    const scale = gridCellKilometres(recipe, radiusRows, radiusColumns);
    const toGrid = (metres) => Math.max(0.00018, metres / (scale * 1000));
    return {
      arterial: toGrid(finiteOr(declared.arterial, fallback.arterial)),
      collector: toGrid(finiteOr(declared.collector, fallback.collector)),
      local: toGrid(finiteOr(declared.local, fallback.local)),
    };
  }

  function gridCellKilometres(recipe, radiusRows, radiusColumns) {
    const metrics = recipe.urban && recipe.urban.gridCellKilometres || {};
    const row = finiteOr(metrics.row, NaN);
    const column = finiteOr(metrics.column, NaN);
    if (row > 0 && column > 0) return (row + column) * 0.5;
    const radiusKm = finiteOr(recipe.urban && recipe.urban.radiusKm, 1);
    const radius = (Math.max(0.00001, radiusRows) + Math.max(0.00001, radiusColumns)) * 0.5;
    return Math.max(0.001, radiusKm / radius);
  }

  function pathData(points, closed) {
    if (!points.length) return "";
    const segments = points.map((point, index) => (index ? "L " : "M ") + Number(point.column.toFixed(7)) + " " + Number(point.row.toFixed(7)));
    return segments.join(" ") + (closed ? " Z" : "");
  }

  function svg(tag, attributes) {
    const node = document.createElementNS(SVG_NS, tag);
    Object.keys(attributes).forEach((key) => {
      if (attributes[key] !== undefined && attributes[key] !== null) {
        node.setAttribute(key, String(attributes[key]));
      }
    });
    return node;
  }

  function validateRecipe(recipe) {
    if (!recipe || typeof recipe !== "object" || typeof recipe.id !== "string" || !recipe.id || !recipe.location) {
      throw new TypeError("a city recipe requires id and location");
    }
    if (!Number.isFinite(recipe.location.row) || !Number.isFinite(recipe.location.column)) {
      throw new TypeError("city recipe location must use finite world-grid coordinates");
    }
    if (!VALID_ERAS.has(recipe.era)) {
      throw new TypeError("city recipe requires a strict technology era");
    }
    const buildable = recipe.terrain && recipe.terrain.buildable;
    if (!buildable || buildable.coordinateSpace !== "world-grid-cells" || !Array.isArray(buildable.allowedCells) || !buildable.allowedCells.length) {
      throw new TypeError("city recipe requires non-empty native-grid buildable cells");
    }
    buildable.allowedCells.forEach((cell) => {
      if (!Array.isArray(cell) || cell.length !== 2 || !Number.isInteger(cell[0]) || !Number.isInteger(cell[1])) {
        throw new TypeError("city buildable cells must be [row, column] integer pairs");
      }
    });
  }

  function normalizeBuildable(buildable) {
    return {
      allowedCells: buildable.allowedCells.map((cell) => [cell[0], cell[1]]),
      bounds: buildable.bounds || null,
      precision: buildable.precision || "native-grid-cell",
    };
  }

  function clipIdentifier(recipe, prefix) {
    const safe = recipe.id.replace(/[^A-Za-z0-9_.-]/g, "-").slice(0, 72) || "city";
    const safePrefix = String(prefix).replace(/[^A-Za-z0-9_.-]/g, "-").slice(0, 72) || "city-detail";
    return safePrefix + "-" + safe + "-" + unsigned(recipe.seed).toString(36);
  }

  function normalizeGrid(grid) {
    if (!grid || typeof grid !== "object") return null;
    const width = finiteOr(grid.width, 0);
    const height = finiteOr(grid.height, 0);
    return width > 0 && height > 0 ? { width, height } : null;
  }

  function finiteOr(value, fallback) {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric : fallback;
  }

  function unsigned(value) {
    return (Number(value) >>> 0);
  }

  function normalizeAngle(value) {
    const numeric = finiteOr(value, 0);
    return ((numeric % 360) + 360) % 360;
  }

  function random(seed, channel) {
    let value = (unsigned(seed) ^ Math.imul(unsigned(channel) + 0x9e3779b9, 0x85ebca6b)) >>> 0;
    value ^= value >>> 16;
    value = Math.imul(value, 0x7feb352d) >>> 0;
    value ^= value >>> 15;
    value = Math.imul(value, 0x846ca68b) >>> 0;
    value ^= value >>> 16;
    return (value >>> 0) / 4294967296;
  }

  const api = Object.freeze({ plan, render });
  global.WorldAtlasCities = api;
})(typeof window !== "undefined" ? window : globalThis);
