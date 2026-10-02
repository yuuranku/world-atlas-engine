import json
from pathlib import Path
import shutil
import subprocess
import textwrap
import unittest
import shapely


@unittest.skipUnless(shutil.which("node"), "Node.js is required for the browser component test")
class CityDetailScriptTests(unittest.TestCase):
    def test_city_planner_is_deterministic_water_constrained_and_detached(self):
        script = Path(__file__).resolve().parents[1] / "src/world_atlas/core/web/city-detail.js"
        program = textwrap.dedent(
            """
            const assert = require('node:assert/strict');
            const fs = require('node:fs');
            const vm = require('node:vm');
            class Element {
              constructor(tag) { this.tag = tag; this.attrs = {}; this.children = []; this.parentNode = null; }
              setAttribute(key, value) { this.attrs[key] = String(value); }
              appendChild(node) {
                if (node.parentNode) node.parentNode.removeChild(node);
                this.children.push(node); node.parentNode = this; return node;
              }
              removeChild(node) {
                const index = this.children.indexOf(node);
                if (index < 0) throw new Error('missing child');
                this.children.splice(index, 1); node.parentNode = null; return node;
              }
            }
            const document = { createElementNS: (_namespace, tag) => new Element(tag) };
            const window = {};
            const context = { window, document, console, Math, Map, Set, Number, String, Object, Array, TypeError, Error };
            context.globalThis = window;
            vm.runInNewContext(fs.readFileSync(process.argv[1].replace('city-detail.js','city-site.js'),'utf8'),context);
            vm.runInNewContext(fs.readFileSync(require('node:path').join(require('node:path').dirname(process.argv[1]),'city-character.js'),'utf8'),context);
            vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), context, { filename: 'city-detail.js' });
            const api = window.WorldAtlasCities;
            assert.ok(api);
            for(const siteType of ['port','island-port','lake-port','river-city']) {
              const c=window.WorldAtlasCityCharacter.profile({siteType,tier:'site',era:'preindustrial',
                population:{estimate:1200},culture:{style:'stone-masonry',government:'nomadic-confederacy'}});
              assert.equal(c.mobile,false,'permanent waterside settlements keep buildings under a nomadic government');
              assert.equal(c.style,'stone-masonry');
            }
            const city = {
              id: 'port-city', name: 'Port City', seed: 991, era: 'industrial', tier: 'metropolis',
              location: { row: 10.5, column: 10.5 },
              population: { estimate: 450000 },
              urban: {
                radiusKm: 4.5, radiusRows: 0.28, radiusColumns: 0.31, coreCount: 3,
                bounds: { north: 10.22, west: 10.19, south: 10.78, east: 10.81 },
                gridCellKilometres: { row: 16.2, column: 15.9 },
                streetWidthsMetres: { arterial: 23, collector: 11, local: 5 },
                coreZones: [{ id: 'port-city-core-1', role: 'harbor-core', row: 10.5, column: 10.5, radiusKm: 1.1 }]
              },
              morphology: { kind: 'waterfront-harbor', streetPattern: 'quay-linear' },
              terrain: {
                slopeClass: 'gentle',
                buildable: {
                  coordinateSpace: 'world-grid-cells', precision: 'native-grid-cell',
                  bounds: { north: 10, west: 10, south: 11, east: 11 },
                  allowedCells: [[10, 10], [10, 11], [11, 10], [11, 11]],
                  blockedCells: [[10, 12]], clipCityGeometry: true
                }
              },
              water: { kind: 'seaport', waterBearingDegrees: 0 },
              transport: { interfaces: ['port-quay'], bridgeCount: 0 }
            };
            const firstPlan = api.plan(city);
            assert.deepEqual(firstPlan, api.plan(city));
            assert.ok(firstPlan.blocks.length >= 100, 'a metropolis must have a dense block fabric');
            assert.ok(firstPlan.streets.length >= 30, 'a metropolis must have a full street graph, not an icon');
            assert.equal(firstPlan.buildable.allowedCells.length, 4);
            assert.equal(firstPlan.districts.length, firstPlan.blocks.length);
            assert.equal(firstPlan.waterfront, undefined, 'a water bearing cannot invent a waterway');
            assert.notDeepEqual(firstPlan.blocks, api.plan({ ...city, seed: 995 }).blocks);
            assert.ok(firstPlan.buildings.length > firstPlan.blocks.length * 5, 'city zoom subdivides plots into real building footprints');
            assert.ok(firstPlan.buildings.some(building => building.roofLines.length), 'buildings have roof geometry rather than flat colour alone');
            assert.ok(firstPlan.plazas.length && firstPlan.landmarks.some(landmark => landmark.kind === 'market-hall'),
              'city cores reserve a public square with an actual civic building');
            assert.ok(firstPlan.streets.some(street => street.plaza), 'public squares connect to the existing street network');
            console.log(JSON.stringify({plazas:firstPlan.plazas.map(p=>p.points),buildings:firstPlan.buildings.map(b=>b.points)}));
            const medieval = api.plan({...city, era: 'medieval'});
            assert.notDeepEqual(medieval.streets, firstPlan.streets, 'era changes the street topology');
            assert.ok(medieval.walls.length > 0);
            const ancientFort = api.plan({...city, era:'ancient', siteType:'fortress', landmarks:['citadel']});
            assert.ok(ancientFort.walls.length > 0, 'classical cities retain defensive perimeter walls');
            assert.ok(ancientFort.innerWalls.length > 0, 'classical cities have a compact inner defensive perimeter');
            assert.ok(ancientFort.towers.length > 0, 'defensive walls include actual tower footprints');
            assert.ok(ancientFort.fortress && ancientFort.fortress.towers.length === 4, 'a classical fortress has a keep, courtyard, gate and corner towers');
            assert.ok(['garrison','palace-citadel'].includes(ancientFort.fortress.kind));
            assert.ok(ancientFort.fortress.interiors.some(part=>part.kind==='command-hall'));
            assert.equal(ancientFort.fortress.keep, undefined, 'a classical fort is not a medieval keep');
            assert.equal(ancientFort.fortress.access.length, 2);
            assert.equal(firstPlan.walls.length, 0);
            assert.equal(firstPlan.innerWalls.length, 0);
            assert.equal(firstPlan.towers.length, 0);
            const courtyards = api.plan({...city, culture: {style:'courtyard'}});
            assert.ok(courtyards.buildings.some(building => building.courtyard && building.courtyard.length >= 3), 'courtyard cultures create actual interior courts');
            const terraces = api.plan({...city, culture: {style:'terraced'}});
            assert.notDeepEqual(terraces.buildings.map(building => building.points), firstPlan.buildings.map(building => building.points), 'culture changes lot and building geometry');
            const clipped = api.plan({...city, location: {row: 10.9, column: 10.9}, terrain: {
              ...city.terrain, buildable: {...city.terrain.buildable, allowedCells: [[10,10]]}
            }});
            for (const feature of [...clipped.blocks, ...clipped.buildings, ...clipped.streets]) {
              for (const point of feature.points) assert.ok(point.row >= 10 - 1e-9 && point.row <= 11 + 1e-9 && point.column >= 10 - 1e-9 && point.column <= 11 + 1e-9,
                'native water cells are excluded in the geometry planner itself');
            }
            const diagonal = [{row:10.2,column:10.2}, {row:10.8,column:10.8}];
            const connected = api.plan({...city, transport: {...city.transport, corridors: [{id:'real-road',kind:'road',points:diagonal}]}});
            assert.ok(connected.streets.some(street => street.access), 'world routes connect to the city street network');
            const clearance = connected.streetWidths.arterial * 0.70;
            for (const building of connected.buildings) {
              const distances = building.points.map(point => (point.row - point.column) / Math.SQRT2);
              assert.ok(Math.min(...distances) > clearance || Math.max(...distances) < -clearance,
                'buildings leave the actual diagonal road corridor clear');
            }
            const country = api.plan({...city, transport: {corridors: [{id:'country-road',kind:'road',
              points:[{row:10.5,column:10.0},{row:10.5,column:11.1}]}]}});
            assert.ok(country.countryside.length > 0, 'known external roads support sparse homes outside the built-up districts');
            assert.equal(api.plan({...city, transport:{corridors:[]}}).countryside.length, 0,
              'the planner does not invent rural access roads to decorate the map');
            const surfaceSize=17;
            const fineMask=Array.from({length:surfaceSize*surfaceSize},(_,index)=>index%surfaceSize<12?1:0);
            const fineCity={...city,era:'ancient',localSite:{form:{kind:'plain',axisRadians:0,alongScale:.95,crossScale:.92},surface:{rows:surfaceSize,columns:surfaceSize,
              bounds:{north:10,west:10,south:11,east:11},buildable:fineMask,slopes:fineMask.map(v=>v?0:.5),elevationMetres:Array(surfaceSize*surfaceSize).fill(30)},
              channels:[{kind:'stream',widthMetres:160,points:[{row:10.1,column:10.4,widthMetres:160},{row:10.9,column:10.4,widthMetres:160}]}]}};
            const fineBefore=JSON.stringify(fineCity), finePlan=api.plan(fineCity);
            assert.ok(finePlan.buildings.length>0, 'fine terrain constraints preserve the remaining valid city');
            const channelHalfWidth=80/(finePlan.gridCellKilometres*1000);
            assert.ok(finePlan.blocks.every(block=>block.points.every(point=>point.column>=10.4+channelHalfWidth-1e-8)),
              'without a bridge the city remains on the bank connected to its civic centre, with no opposite-bank fragment');
            function districtsConnected(districts){
              const share=(first,second)=>first.some((a,index)=>{
                const b=first[(index+1)%first.length],dx=b.column-a.column,dy=b.row-a.row,length=Math.hypot(dx,dy);
                if(length<1e-9)return false;
                return second.some((c,edge)=>{
                  const d=second[(edge+1)%second.length];
                  if(Math.abs(dx*(c.row-a.row)-dy*(c.column-a.column))>length*1e-7
                    ||Math.abs(dx*(d.row-a.row)-dy*(d.column-a.column))>length*1e-7)return false;
                  const t=p=>((p.column-a.column)*dx+(p.row-a.row)*dy)/(length*length);
                  return Math.min(1,Math.max(t(c),t(d)))-Math.max(0,Math.min(t(c),t(d)))>1e-7;
                });
              });
              const visited=new Set([0]),queue=[0];
              for(let index=0;index<queue.length;index++)for(let other=0;other<districts.length;other++)
                if(!visited.has(other)&&share(districts[queue[index]],districts[other])){visited.add(other);queue.push(other);}
              return visited.size===districts.length;
            }
            for(const seed of [991,992,993,994,995,996])assert.ok(districtsConnected(api.plan({...city,seed}).districts),
              'the settlement around the market stays connected across random layouts');
            for(const feature of [...finePlan.blocks,...finePlan.buildings,...finePlan.plazas,...finePlan.landmarks,
                ...finePlan.towers,...finePlan.countryside,...finePlan.streets,
                ...finePlan.walls.map(points=>({points})),...finePlan.innerWalls.map(points=>({points}))])for(const point of feature.points){
              assert.ok(point.column<=10.7125+1e-8,'all city geometry respects the continuous fine slope boundary');
              assert.ok(Math.abs(point.column-10.4)>=channelHalfWidth-1e-8,
                'houses, plazas, towers and streets leave the actual generated water channel clear');
            }
            assert.equal(JSON.stringify(fineCity),fineBefore,'the fine terrain and water payload remains immutable');
            assert.deepEqual(finePlan,api.plan(fineCity),'fine terrain and detailed features reproduce from the same seed');
            const standalone = api.render(city, {grid: {width: 36, height: 18}});
            assert.equal(standalone.parentNode, null, 'rendering returns detached SVG geometry');
            const standaloneClip = standalone.children[0].children[0].attrs.id;
            assert.ok(standaloneClip.startsWith('city-detail-'));
            assert.equal(standalone.children[0].tag, 'defs');
            assert.equal(standalone.children[0].children[0].tag, 'clipPath');
            assert.equal(standalone.children[0].children[0].children.length, 4);
            assert.equal(standalone.children[1].attrs['clip-path'], `url(#${standaloneClip})`);
            const standaloneNodes = [];
            function visitStandalone(node) { standaloneNodes.push(node); node.children.forEach(visitStandalone); }
            visitStandalone(standalone);
            assert.ok(standaloneNodes.some(node => node.attrs.class === 'city-blocks'));
            assert.ok(standaloneNodes.some(node => node.attrs.class === 'city-buildings'), 'rendering includes detailed city buildings');
            assert.ok(!standaloneNodes.some(node => /city-waterfront|city-bridge-interface|city-rail-interface/.test(node.attrs.class || '')),
              'water and transport must use authoritative map coordinates, not decorative markers');
            const cityLabels=standaloneNodes.filter(node=>node.attrs['data-city-screen-label']);
            assert.ok(cityLabels.some(node=>node.textContent.includes('市集广场')),'the civic square receives a specific name');
            assert.ok(cityLabels.some(node=>node.attrs['data-feature-kind']==='road'),'major streets receive actual feature labels');
            assert.ok(cityLabels.every(node=>!['城郊','旧城','粮田','市集'].includes(node.textContent)),'anonymous district labels are removed');
            const group = new Element('g');
            group.appendChild(standalone);
            const snapshot = node => [node.tag, node.attrs, node.textContent, node.children.map(snapshot)];
            const beforeAnotherRender = JSON.stringify(snapshot(standalone));
            const otherDrawing = api.render(city, {grid: {width: 36, height: 18}, clipPrefix: 'another-city'});
            assert.equal(otherDrawing.parentNode, null, 'the caller owns mounting each city');
            assert.notEqual(otherDrawing.children[0].children[0].attrs.id, standaloneClip, 'separately mounted drawings use their declared clip prefix');
            assert.equal(standalone.parentNode, group, 'rendering another drawing leaves existing atlas geometry mounted');
            assert.equal(JSON.stringify(snapshot(standalone)), beforeAnotherRender, 'rendering does not mutate an existing drawing');
            assert.equal(group.children.length, 1, 'the renderer does not mount extra cities');
            assert.deepEqual(snapshot(api.render(city, {grid: {width: 36, height: 18}})), snapshot(standalone),
              'the same recipe renders the same detached geometry');
            const changedSeed = api.render({...city, seed: 994}, {grid: {width: 36, height: 18}});
            assert.notEqual(changedSeed.children[0].children[0].attrs.id, standaloneClip, 'regeneration gets a seed-specific clipping identifier');
            const seamCity = column => ({...city, id:'seam-city', location:{row:10.5,column},
              urban:{...city.urban, bounds:{north:10.22,south:10.78,west:column-.31,east:column+.31},
                coreZones:[{id:'seam-core',role:'central-core',row:10.5,column,radiusKm:1.1}]},
              terrain:{...city.terrain,buildable:{...city.terrain.buildable,allowedCells:[[10,35],[10,0]],blockedCells:[[10,1]]}},
              transport:{corridors:[{id:'seam-road',kind:'road',points:[{row:10.5,column:column-.4},{row:10.5,column:column+.4}]}]}
            });
            for (const [column, boundary, offset] of [[.05,0,36],[35.95,36,-36]]) {
              const seam = seamCity(column), original = JSON.stringify(seam);
              const planned = api.plan(seam,{grid:{width:36,height:18}});
              const inlandColumn=10+(column%1);
              const inland={...seam,location:{row:10.5,column:inlandColumn},
                urban:{...seam.urban,coreZones:[{...seam.urban.coreZones[0],column:inlandColumn}]},
                terrain:{...seam.terrain,buildable:{...seam.terrain.buildable,allowedCells:boundary===0?[[10,9],[10,10]]:[[10,10],[10,11]],blockedCells:[]}},
                transport:{corridors:[{id:'seam-road',kind:'road',points:seam.transport.corridors[0].points.map(point=>({...point,column:point.column-column+inlandColumn}))}]}};
              const inlandPlan=api.plan(inland,{grid:{width:36,height:18}});
              assert.equal(planned.buildings.length,inlandPlan.buildings.length,'wrapping must not remove valid buildings');
              const shape = (items,centre) => items.map(item=>item.points.map(point=>[Number((point.row-10.5).toFixed(7)),Number((point.column-centre).toFixed(7))]));
              assert.deepEqual(shape(planned.buildings,column),shape(inlandPlan.buildings,inlandColumn),'local city geometry is translation-invariant at both longitude seams');
              assert.ok(planned.buildings.some(building=>building.points.some(point=>point.column<boundary)));
              assert.ok(planned.buildings.some(building=>building.points.some(point=>point.column>boundary)));
              assert.ok(planned.streets.every(street=>Math.abs(street.points[1].column-street.points[0].column)<1),'no street jumps across the whole map');
              assert.equal(JSON.stringify(seam),original,'planning may not mutate canonical cells or recipe coordinates');
              const rendered=api.render(seam,{grid:{width:36,height:18},clipPrefix:'seam-detail'});
              assert.equal(rendered.parentNode,null);
              const copy=rendered.children.find(node=>node.tag==='use');
              assert.ok(copy,'a seam city must draw its other portion at the opposite atlas edge');
              assert.equal(copy.attrs.transform,`translate(${offset} 0)`);
              assert.equal(copy.attrs.href,'#'+rendered.children[1].attrs.id);
              const clip=rendered.children[0].children[0];
              assert.ok(clip.children.some(node=>Number(node.attrs.x)<boundary));
              assert.ok(clip.children.some(node=>Number(node.attrs.x)+1>boundary));
              assert.equal(JSON.stringify(seam),original,'rendering preserves canonical cells and coordinates at the longitude seam');
            }
            const shore=seamCity(.055);
            shore.terrain.buildable.allowedCells=[[10,0]];
            shore.terrain.buildable.blockedCells=[[10,35]];
            const shorePlan=api.plan(shore,{grid:{width:36,height:18}});
            assert.ok(shorePlan.buildings.length>0);
            assert.ok([...shorePlan.buildings,...shorePlan.streets].every(item=>item.points.every(point=>point.column>=-1e-9)),
              'refined coastal anchors preserve the forbidden water cell across the longitude seam');
            console.log(JSON.stringify({ ok: true, nodes: standaloneNodes.length, points: firstPlan.footprint.length }));
            """
        )
        result = subprocess.run(
            ["node", "-e", program, str(script)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        geometry=json.loads(result.stdout.splitlines()[0])
        polygon=lambda points:shapely.Polygon([(p['column'],p['row']) for p in points])
        buildings=[polygon(points) for points in geometry['buildings']]
        tree=shapely.STRtree(buildings)
        for points in geometry['plazas']:
            plaza=polygon(points)
            for index in tree.query(plaza,predicate='intersects'):
                self.assertLess(plaza.intersection(buildings[index]).area,1e-14,
                    'actual public square polygon must be reserved, including rotated street frontages')
        report = json.loads(result.stdout.splitlines()[-1])
        self.assertTrue(report["ok"])
        self.assertGreaterEqual(report["points"], 36)


if __name__ == "__main__":
    unittest.main()
