"""Display topology is sampled; emitted roots still belong to real ground."""
import unittest

import numpy as np
import shapely

from tests.test_implicit_terrain_bounds import refined_fixture
from world_atlas.core.cartographic_contours import (
    CARTOGRAPHIC_CONTOUR_CONTRACT,cartographic_level_curves,
)
from world_atlas.core.continuous_terrain import PhysicalTerrainField


def field(values):
    return PhysicalTerrainField(values,land_mask=values>0,sea_level_m=0,
                                elevation_scale_m=4000,elevation_exponent=1)


class CartographicContoursTests(unittest.TestCase):
    def test_refined_vertices_solve_original_ground_not_linear_sample_heights(self):
        base,source=refined_fixture()
        paths=cartographic_level_curves(source,[500.,1000.,1500.])
        self.assertTrue(all(paths))
        for level,graphs in zip((500.,1000.,1500.),paths,strict=True):
            points=np.concatenate(graphs)
            actual=source.sample_points(points[:,0],points[:,1])
            self.assertLess(float(abs(actual-level).max()),2e-8)
        points=np.concatenate(paths[1])
        self.assertGreater(float(abs(base.sample_points(points[:,0],points[:,1])-1000.).max()),1.)

    def test_adjacent_local_patches_share_byte_identical_boundary_roots(self):
        rows,columns=np.indices((12,14))
        source=field(100+columns*35+rows*81)
        left=cartographic_level_curves(source,[700.],query_bounds=[[2.,2.,6.,10.]])[0]
        right=cartographic_level_curves(source,[700.],query_bounds=[[6.,2.,10.,10.]])[0]
        a=np.concatenate(left);b=np.concatenate(right)
        np.testing.assert_array_equal(a[a[:,0]==6.].view(np.uint64),b[b[:,0]==6.].view(np.uint64))
        whole=cartographic_level_curves(source,[700.],query_bounds=[[2.,2.,6.,10.],[6.,2.,10.,10.]])[0]
        self.assertEqual(len(whole),1)
        self.assertTrue(all(len(path)>=2 for path in whole))

    def test_plateaus_empty_levels_and_closed_native_peak_remain_valid(self):
        source=field(np.full((7,7),100.))
        self.assertEqual(cartographic_level_curves(source,[100.,101.]),[[],[]])
        values=np.full((7,7),99.)
        values[3,3]=100.0001
        source=field(values)
        paths=cartographic_level_curves(source,[100.])[0]
        self.assertEqual(len(paths),1)
        np.testing.assert_array_equal(paths[0][0],paths[0][-1])
        self.assertLess(float(abs(source.sample_points(paths[0][:,0],paths[0][:,1])-100.).max()),1e-10)
        self.assertTrue(shapely.LineString(paths[0]).is_simple)

    def test_hidden_subsample_peak_is_explicitly_outside_display_topology_contract(self):
        _,source=refined_fixture(protect_rivers=False)
        peak=(4.516311,4.50029469)
        self.assertGreater(float(source.sample_points(*peak)),1900.25)
        self.assertEqual(cartographic_level_curves(source,[1900.25]),[[]])
        self.assertIn("not certified",CARTOGRAPHIC_CONTOUR_CONTRACT["hiddenFeatures"])


if __name__=="__main__":unittest.main()
