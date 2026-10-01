"""The overview retains the actual supported field contours used by detail."""

from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import shapely

from world_atlas.core.cartographic_features import overview_markup
from world_atlas.core.cartographic_tiles import feature_markup, geometry_path_data
from world_atlas.core.landform_render import landform_features


class LandformOverviewContourTests(unittest.TestCase):
    def test_real_near_threshold_peak_keeps_its_field_boundary_in_overview(self):
        # Preserved v91 wetland support [731:738,870:877]. The peak is only
        # .00028479 above .5; native-centre protection alone accepts a triangle.
        support = np.zeros((7, 7), dtype=np.float64)
        support[3, 3] = .5002847909927368
        support[4, 4] = .358524888753891
        support[5, 5] = .36773616075515747
        support[6, 6] = .48995158076286316
        grid = SimpleNamespace(shape=support.shape, water=np.zeros(support.shape, dtype=np.uint8))
        masks = {kind: np.zeros(support.shape, dtype=bool)
                 for kind in ('wetland', 'snow-mountain', 'arid-upland', 'steep-slope')}
        masks['wetland'] = support >= .5
        inventory = SimpleNamespace(masks=masks, wetland_support=support,
                                    direction_east=np.zeros_like(support),
                                    direction_north=np.zeros_like(support))
        features = landform_features(grid, inventory)
        tint = next(feature for feature in features
                    if feature.attributes.get('data-landform-type') == 'wetland'
                    and feature.attributes.get('data-landform-texture') != 'true')
        original = tint.geometry
        self.assertGreater(len(original.exterior.coords), 20)
        self.assertLess(original.bounds[2] - original.bounds[0], .03)
        simplified = shapely.simplify(original, .12, preserve_topology=True)
        self.assertEqual(len(simplified.exterior.coords), 4)
        self.assertTrue(simplified.covers(shapely.Point(3.5, 3.5)))

        surface = shapely.box(0, 0, 7, 7)
        markup = overview_markup(features, surface, 7, 7, section='ink')
        overview = ET.fromstring('<svg>' + markup + '</svg>')
        outline = overview.find('.//path[@data-landform-type="wetland"]')
        detail = ET.fromstring('<svg>' + feature_markup(tint, clip_ids={'land': 'land-clip'}) + '</svg>')
        detail_outline = detail.find('.//path')
        self.assertEqual(outline.attrib['d'], geometry_path_data(original))
        self.assertEqual(outline.attrib['d'], detail_outline.attrib['d'])
        self.assertNotIn('data-landform-texture', markup)


if __name__ == '__main__':
    unittest.main()
