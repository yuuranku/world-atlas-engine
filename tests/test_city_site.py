import json
from pathlib import Path
import shutil
import subprocess
import unittest

import numpy as np
from types import SimpleNamespace
from scipy.ndimage import map_coordinates

from world_atlas.core.city_site import city_site_context
from world_atlas.core.presentation import derive_presentation_assets
from test_presentation import _fixture


def _terrain(raw):
    return SimpleNamespace(sample_points=lambda x,y:map_coordinates(raw,[y-.5,x-.5],order=1,mode='nearest'))


class CitySiteContextTests(unittest.TestCase):
    def test_saved_metre_context_is_small_and_does_not_change_parent_fields(self):
        grid, society = _fixture()
        recipe = derive_presentation_assets(grid, society).city_recipes[0].document()
        raw = grid.elevation.astype(float) * 200 + 120
        before = raw.copy()
        context = city_site_context(grid, recipe, _terrain(raw))
        self.assertEqual(len(context["elevationMetres"]), 289)
        self.assertEqual(len(context["water"]), 289)
        self.assertGreater(min(context["elevationMetres"]), 100)
        self.assertGreaterEqual(context["radiusKm"], 5)
        self.assertTrue(np.array_equal(raw, before))
        self.assertNotIn("localSite", context)

    def test_strategic_dry_city_has_groundwater_and_regular_city_uses_freshwater(self):
        grid, society = _fixture()
        grid.river_order[:] = 0
        grid.water[:] = 0
        assets = derive_presentation_assets(grid, society)
        recipe = assets.city_recipes[0].document()
        raw = np.ones(grid.shape) * 700
        ordinary = city_site_context(grid, recipe, _terrain(raw))
        self.assertEqual(ordinary["source"]["kind"], "spring-fed-stream")
        recipe["siteType"] = "fortress"
        strategic = city_site_context(grid, recipe, _terrain(raw))
        self.assertEqual(strategic["source"]["kind"], "groundwater")
        grid.water[3, 2] = 2
        recipe["siteType"] = "market"
        lake = city_site_context(grid, recipe, _terrain(raw))
        self.assertEqual(lake["source"]["kind"], "lake")
        self.assertEqual(lake["source"]["location"], {"row": 3.5, "column": 2.5})

    def test_client_refines_one_city_with_repeatable_seed_and_downhill_supply(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node.js is not installed")
        script = Path(__file__).resolve().parents[1] / "src/world_atlas/core/web/city-site.js"
        program = r'''
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const context={window:{}};vm.runInNewContext(fs.readFileSync(process.argv[1],'utf8'),context);
const generate=context.window.WorldAtlasCitySite.generate;
const bounds={north:10,west:20,south:11,east:21};
const elevations=Array.from({length:49},(_,i)=>600-Math.floor(i/7)*20+(i%7)*3);
const source={schema:'city-site-context-v1',rows:7,columns:7,bounds,elevationMetres:elevations,water:Array(49).fill(0),riverOrder:Array(49).fill(0),radiusKm:5,gridCellKilometres:{row:10,column:10},source:{kind:'river',location:{row:12,column:20.5},distanceKm:15}};
const recipe={id:'test-city',location:{row:10.5,column:20.5},siteType:'market',urban:{radiusKm:1},transport:{corridors:[{kind:'road',points:[{row:10.1,column:20.5},{row:10.9,column:20.5}]}]}};
const first=generate(recipe,source,{seed:123}), second=generate(recipe,source,{seed:123}), third=generate(recipe,source,{seed:124});
assert.equal(JSON.stringify(first),JSON.stringify(second));assert.notEqual(JSON.stringify(first),JSON.stringify(third));
assert.equal(first.localSite.surface.rows,65);assert.equal(first.localSite.surface.columns,65);
assert.equal(first.localSite.surface.buildable.length,4225);assert.equal(first.localSite.surface.elevationMetres.length,4225);
assert.ok(first.localSite.terrainBands.length>=3);assert.ok(first.localSite.contours.length>=2);assert.ok(first.localSite.channels.length>0);
for(const channel of first.localSite.channels) for(let i=1;i<channel.points.length;i++) assert.ok(channel.points[i].elevationMetres<=channel.points[i-1].elevationMetres);
const metric=source.gridCellKilometres,center=recipe.location;
const toKm=point=>({x:(point.column-center.column)*metric.column,y:(point.row-center.row)*metric.row});
const creek=first.localSite.channels[0].points.map(toKm);
for(let i=1;i<creek.length;i++){
 const a=creek[i-1],b=creek[i],dx=b.x-a.x,dy=b.y-a.y,t=Math.max(0,Math.min(1,-(a.x*dx+a.y*dy)/(dx*dx+dy*dy)));
 assert.ok(Math.hypot(a.x+dx*t,a.y+dy*t)>recipe.urban.radiusKm*.5,
   'water supply follows a settled river bank while leaving the civic centre dry');
}
let neighbouringFields=false;
for(let i=0;i<first.localSite.farmland.length;i++)for(let j=i+1;j<first.localSite.farmland.length;j++){
 const a=toKm(first.localSite.farmland[i].center),b=toKm(first.localSite.farmland[j].center);
 if(Math.hypot(a.x-b.x,a.y-b.y)<.11)neighbouringFields=true;
}
assert.ok(neighbouringFields,'food fields form adjacent holdings rather than isolated decorative rectangles');
const surface=first.localSite.surface;
const n=surface.rows, last=n-1;
const middle=first.localSite.channels[0].points[12];
const nx=Math.round((middle.column-bounds.west)/(bounds.east-bounds.west)*last),ny=Math.round((middle.row-bounds.north)/(bounds.south-bounds.north)*last);
const parentAtNode=600-ny/last*120+nx/last*18;
assert.ok(surface.elevationMetres[ny*n+nx]<parentAtNode,'The supply channel excavates the shared terrain, rather than only painting water');
assert.equal(surface.buildable[ny*n+nx],0,'The creek floodplain constrains the city layout');
for(let row=0;row<n;row++) for(let col=0;col<n;col++) if(row===0||row===last||col===0||col===last) {
 const expected=600-row/last*120+col/last*18; assert.ok(Math.abs(first.localSite.surface.elevationMetres[row*n+col]-expected)<.011);
}
const dry={...source,elevationMetres:Array(49).fill(500),source:{kind:'groundwater',location:recipe.location}};
const fort=generate({...recipe,siteType:'fortress'},dry,{seed:7});assert.equal(fort.localSite.channels.length,0);assert.equal(fort.localSite.waterSource.kind,'groundwater');
assert.ok(Math.max(...fort.localSite.surface.elevationMetres)-Math.min(...fort.localSite.surface.elevationMetres)>5);
const coastal={...source,elevationMetres:Array.from({length:49},(_,i)=>(i%7-2)*20),water:Array.from({length:49},(_,i)=>i%7<2?1:0)};
const shore=generate(recipe,coastal,{seed:55});
for(let row=0;row<n;row++) for(let col=0;col<n;col++) {
 const index=row*n+col,parent=(col/last*6-2)*20;
 assert.equal(shore.localSite.surface.elevationMetres[index]>0,parent>0);
 if(parent<=0) assert.equal(shore.localSite.surface.buildable[index],0);
}
assert.equal(JSON.stringify(source.elevationMetres),JSON.stringify(elevations));
const t=performance.now();for(let i=0;i<5;i++)generate(recipe,source,{seed:i});console.log(JSON.stringify({meanMs:(performance.now()-t)/5,farmland:first.localSite.farmland.length}));
'''
        result = subprocess.run([node, "-e", program, str(script)], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        timings = json.loads(result.stdout.strip())
        self.assertLess(timings["meanMs"], 300)
        self.assertGreater(timings["farmland"], 0)
