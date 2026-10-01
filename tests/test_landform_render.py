"""Classified terrain texture shares clips and leaves thematic ownership alone."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import shapely

from world_atlas.core.cartographic_features import overview_markup
from world_atlas.core.cartographic_tiles import TileFeature,TileLevel,feature_definitions,write_atlas_tiles
from world_atlas.core.landform_render import landform_features,mask_faces


def fixture():
    shape=(8,12)
    grid=SimpleNamespace(shape=shape,water=np.zeros(shape,dtype=np.uint8),
        metadata={"extents":{"west":-180,"east":180,"north":90,"south":-90},"planet":{"radiusKm":6371}})
    masks={kind:np.zeros(shape,dtype=bool) for kind in ("wetland","snow-mountain","arid-upland","steep-slope")}
    masks["wetland"][2:5,2:5]=True
    masks["snow-mountain"][6,9]=True
    masks["arid-upland"][3:6,7:9]=True
    masks["steep-slope"][2,3]=True;masks["steep-slope"][4,8]=True
    east=np.zeros(shape);north=np.zeros(shape)
    east[2,3]=1;north[4,8]=1
    return grid,SimpleNamespace(masks=masks,wetland_support=masks["wetland"].astype(np.float32),direction_east=east,direction_north=north)


class LandformRenderTests(unittest.TestCase):
    def test_native_area_classes_are_exact_and_have_no_cell_outlines(self):
        grid,inventory=fixture();features=landform_features(grid,inventory)
        y,x=np.indices(grid.shape)
        for kind in ("wetland","snow-mountain","arid-upland"):
            faces=shapely.union_all([feature.geometry for feature in features
                if feature.attributes["data-landform-type"]==kind and not feature.definitions])
            np.testing.assert_array_equal(shapely.covers(faces,shapely.points(x+.5,y+.5)),inventory.masks[kind])
            self.assertTrue(all(feature.attributes["stroke"]=="none" for feature in features
                if feature.attributes["data-landform-type"]==kind))
        self.assertEqual(mask_faces(np.zeros(grid.shape,dtype=bool)),[])

    def test_hachures_follow_measured_downslope_inside_the_supported_cell(self):
        grid,inventory=fixture();features=landform_features(grid,inventory)
        slope=next(feature for feature in features if feature.attributes.get("data-landform-hachure"))
        self.assertEqual(slope.geometry.geom_type,"MultiLineString")
        east,north=list(slope.geometry.geoms)
        first,last=np.asarray(east.coords)
        self.assertGreater(last[0],first[0]);self.assertEqual(first[1],last[1])
        first,last=np.asarray(north.coords)
        self.assertLess(last[1],first[1]);self.assertEqual(first[0],last[0])
        self.assertTrue(shapely.box(3,2,4,3).covers(east))
        self.assertTrue(shapely.box(8,4,9,5).covers(north))
        self.assertEqual(slope.attributes["data-min-scale"],"32.0")
        self.assertEqual(slope.attributes["data-max-scale"],"256.0")

    def test_pattern_is_tile_local_once_and_never_duplicated_into_themes(self):
        grid,inventory=fixture();features=landform_features(grid,inventory)
        theme=TileFeature(shapely.box(0,0,12,8),{"fill":"#ff0000"},"theme-fill",theme="political",section="theme")
        with tempfile.TemporaryDirectory() as directory:
            write_atlas_tiles(directory,12,8,[TileLevel("detail",64,shapely.box(0,0,12,8),[*features,theme])],tile_size=6)
            ids=[]
            for file in (Path(directory)/"tiles/detail").glob("*.json"):
                payload=json.loads(file.read_text())
                root=ET.fromstring('<svg>'+payload["surface"]+payload["ink"]+'</svg>')
                local={node.attrib["id"] for node in root.iter() if "id" in node.attrib}
                patterns=root.findall('.//pattern')
                self.assertEqual(len(patterns),len({node.attrib["id"] for node in patterns}))
                ids.extend(node.attrib["id"] for node in patterns)
                for node in root.findall('.//path'):
                    if node.attrib.get("fill","").startswith("url(#"):
                        self.assertIn(node.attrib["fill"][5:-1],local)
                        self.assertTrue(node.attrib["clip-path"].startswith("url(#tile-land-detail-"))
                self.assertNotIn('<pattern',payload["themes"]["political"])
            self.assertEqual(len(ids),len(set(ids)))

    def test_overview_retains_regional_hint_and_omits_local_texture(self):
        grid,inventory=fixture();features=landform_features(grid,inventory)
        overview=overview_markup(features,shapely.box(0,0,12,8),12,8,section="ink")
        self.assertIn('data-landform-type="wetland"',overview)
        self.assertNotIn('url(#landform-pattern',overview)
        self.assertNotIn('data-landform-hachure',overview)
        self.assertNotIn('data-landform-texture',overview)

    def test_missing_support_and_conflicting_pattern_definitions_fail(self):
        grid,inventory=fixture();grid.water[2,2]=1
        with self.assertRaises(ValueError):landform_features(grid,inventory)
        a=TileFeature(shapely.box(0,0,1,1),{},"geographic-textures",definitions={"a":'<pattern id="a"/>'})
        b=TileFeature(shapely.box(0,0,1,1),{},"geographic-textures",definitions={"a":'<pattern id="a" width="2"/>'})
        with self.assertRaises(ValueError):feature_definitions([a,b],"")


if __name__=="__main__":unittest.main()
