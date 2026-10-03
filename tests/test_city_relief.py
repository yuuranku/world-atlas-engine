"""City minor ink consumes the same true ground without replacing its base."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import shapely

from world_atlas.core.cartographic_tiles import TileLevel, write_atlas_tiles, write_city_relief_tiles
from world_atlas.core.city_relief import city_relief_supports, derive_city_relief
from world_atlas.core.continuous_terrain import PhysicalTerrainField
from world_atlas.core.hypsometry import elevation_display_thresholds


def fixture(raw):
    grid=SimpleNamespace(shape=raw.shape,metadata={"extents":{"west":0,"east":1,"north":1,"south":0},
        "planet":{"radiusKm":6371}})
    field=PhysicalTerrainField(raw,land_mask=raw>0,sea_level_m=671,elevation_scale_m=6000,elevation_exponent=1.06)
    city=SimpleNamespace(identifier="town-1",tier="town",row=4,column=4)
    levels=elevation_display_thresholds()
    return grid,field,city,levels


class CityReliefTests(unittest.TestCase):
    def test_verified_display_anchor_owns_support_and_is_required(self):
        grid,_,city,_=fixture(np.ones((12,24)))
        support=city_relief_supports(grid,[city],{city.identifier:(5.375,7.125)})[0]
        self.assertEqual((support["x"],support["y"]),(7.125,5.375))
        with self.assertRaises(KeyError):city_relief_supports(grid,[city],{})
        city.tier="site"
        self.assertEqual(city_relief_supports(grid,[city],{}),())

    def test_flat_ground_leaves_global_paint_and_major_contours_authoritative(self):
        for height in (1., 100.):
            grid,field,city,levels=fixture(np.full((12,24), height))
            relief=derive_city_relief(grid,field,[city],levels,settlement_locations={city.identifier:(4.5,4.5)})
            self.assertEqual(relief.features, ())
            self.assertEqual(relief.published_tiles, ())
            self.assertGreater(len(relief.supports), 0)

    def test_only_true_minor_lines_are_added_and_native_source_is_unchanged(self):
        y,x=np.indices((12,24));raw=100+x*180+y*115
        grid,field,city,levels=fixture(raw)
        original=field.native_m.copy()
        relief=derive_city_relief(grid,field,[city],levels,settlement_locations={city.identifier:(4.5,4.5)})
        self.assertGreater(len(relief.features), 0)
        kinds=set()
        for feature in relief.features:
            self.assertEqual(feature.section,"ink")
            self.assertEqual(feature.layer,"elevation-contours")
            kinds.add(feature.attributes["data-contour-kind"])
            points=np.asarray(feature.geometry.coords)
            heights=field.sample_points(points[:,0],points[:,1])
            self.assertLess(float(np.max(np.abs(heights-float(feature.attributes["data-height-m"])))),1e-9)
            self.assertNotIn("data-sampling-cells",feature.attributes)
            self.assertEqual(feature.attributes["data-source-field"],"accepted-continuous-ground")
        self.assertEqual(kinds,{"minor"})
        np.testing.assert_array_equal(field.native_m,original)
        self.assertFalse(field.native_m.flags.writeable)
        self.assertEqual(relief.diagnostics["nativeHeightsChanged"],0)

    def test_sparse_minor_payload_reuses_detail_land_and_one_support_mask(self):
        y,x=np.indices((12,24));grid,field,city,levels=fixture(100+x*180+y*115)
        relief=derive_city_relief(grid,field,[city],levels,settlement_locations={city.identifier:(4.5,4.5)})
        with tempfile.TemporaryDirectory() as directory:
            write_atlas_tiles(directory,24,12,[TileLevel(name,scale,shapely.box(0,0,24,12),[])
                for name,scale in (("regional",4),("local",16),("detail",64))])
            manifest=write_city_relief_tiles(directory,relief)
            self.assertEqual(manifest["levels"][-1]["publishedTiles"],["0-0"])
            payload=json.loads((Path(directory)/"tiles/city-detail/0-0.json").read_text())
            self.assertEqual(set(payload),{"bounds","surface","ink"})
            root=ET.fromstring("<svg>"+payload["surface"]+payload["ink"]+"</svg>")
            self.assertEqual(len(root.findall(".//clipPath")),0)
            masks=root.findall(".//mask")
            self.assertEqual(len(masks),1)
            self.assertEqual(masks[0].attrib["style"],"mask-type:alpha")
            surface=ET.fromstring("<svg>"+payload["surface"]+"</svg>")
            self.assertEqual([node.tag for node in surface], ["defs"])
            self.assertTrue(all(node.attrib["clip-path"]=="url(#tile-land-detail-0-0)"
                for node in root.iter() if "clip-path" in node.attrib))

    def test_barely_superthreshold_city_peak_retains_true_100_metre_curve(self):
        values=np.full((9,11),99.)
        values[4,4]=100.0001
        grid,field,city,_=fixture(values)
        relief=derive_city_relief(grid,field,[city],np.array((0.,1.)),
            settlement_locations={city.identifier:(4.5,4.5)})
        self.assertGreater(len(relief.features),0)
        lines=shapely.union_all([feature.geometry for feature in relief.features])
        points=shapely.get_coordinates(lines)
        self.assertGreaterEqual(len(points),4)
        self.assertEqual(relief.diagnostics["cartographicContract"]["subdivisionPerNativeCell"],4)
        self.assertLess(float(np.max(abs(field.sample_points(points[:,0],points[:,1])-100.))),1e-10)
        self.assertTrue(all(feature.attributes["data-height-m"]=="100.0" for feature in relief.features))
        self.assertEqual(relief.published_tiles,((0,0),))


if __name__=="__main__":unittest.main()
