"""The implicit oracle bounds the real ground, including unseen subgrid peaks."""
import unittest
import json
from pathlib import Path

import numpy as np

from world_atlas.core.continuous_terrain import PhysicalTerrainField
from world_atlas.core.implicit_terrain_bounds import (
    directional_bounds, field_bounds, field_range_gradient_bounds,
    warped_pchip_event_bounds,
)
from world_atlas.core.terrain_refinement import RefinedTerrainField


def refined_fixture(*, protect_rivers=True):
    values = np.full((9, 9), 100.)
    values[3:6, 3:6] = ((700., 1100., 800.),
                       (1200., 1900., 900.), (600., 800., 500.))
    base = PhysicalTerrainField(values, land_mask=values > 0, sea_level_m=0,
                               elevation_scale_m=4000, elevation_exponent=1)
    flow = np.full(values.shape, -1, dtype=np.int32)
    flow[4, 4], flow[4, 5] = 4*9+5, 5*9+5
    discharge = np.zeros(values.shape)
    discharge[4, 4], discharge[4, 5] = 64., 128.
    segments = (np.array((((4.5, 4.5), (5.5, 4.5)),
                          ((5.5, 4.5), (5.5, 5.5))))
                if protect_rivers else np.empty((0, 2, 2)))
    field = RefinedTerrainField(base, seed=1, radius_km=12,
        flow_to=flow, discharge=discharge,
        river_segments=segments,
        river_anchors=np.array(((3.71, 3.89),)))
    return base, field


def coastal_fixture():
    rows,columns = np.indices((16,24))
    values = (columns-11.3)*70+15*np.sin(rows*.7)
    base = PhysicalTerrainField(values,land_mask=values>0,sea_level_m=0,
                                elevation_scale_m=4000,elevation_exponent=1)
    return RefinedTerrainField(base,seed=91,radius_km=100,
        flow_to=np.full(values.shape,-1,dtype=np.int32),discharge=np.zeros(values.shape),
        river_segments=np.empty((0,2,2)),river_anchors=np.empty((0,2)))


class ImplicitTerrainBoundsTests(unittest.TestCase):
    def test_real_refined_values_and_piecewise_gradients_are_enclosed(self):
        base, field = refined_fixture()
        rng = np.random.default_rng(8319)
        centres = rng.uniform(.6, 8.4, (47, 2))
        lower, upper = centres-.08, centres+.08
        low, high, grad_low, grad_high = field_range_gradient_bounds(field, lower, upper)
        points = lower[:, None, :]+(upper-lower)[:, None, :]*rng.random((47, 53, 2))
        actual = field.sample_points(points[:, :, 0], points[:, :, 1])
        self.assertTrue(np.all(low[:, None] <= actual))
        self.assertTrue(np.all(actual <= high[:, None]))
        self.assertGreater(float(np.max(abs(actual-base.sample_points(
            points[:, :, 0], points[:, :, 1])))), 1.)
        # Numerical derivatives are independent witnesses here; the oracle
        # itself uses analytic interval derivatives of the actual formula.
        for axis in (0, 1):
            step = np.eye(2)[axis]*1e-6
            first, last = points-step, points+step
            gradient = (field.sample_points(last[:, :, 0], last[:, :, 1])
                        -field.sample_points(first[:, :, 0], first[:, :, 1]))/2e-6
            self.assertTrue(np.all(grad_low[:, None, axis]-2e-7 <= gradient))
            self.assertTrue(np.all(gradient <= grad_high[:, None, axis]+2e-7))

    def test_same_sign_corners_do_not_discard_a_hidden_peak(self):
        _, field = refined_fixture()
        centre = np.array((4.5, 4.5))
        lower, upper = centre-.1, centre+.1
        corners = np.array((lower, (upper[0], lower[1]), upper,
                            (lower[0], upper[1])))
        corner_heights = field.sample_points(corners[:, 0], corners[:, 1])
        peak = float(field.sample_points(*centre))
        level = (float(corner_heights.max())+peak)/2
        self.assertTrue(np.all(corner_heights < level))
        self.assertGreater(peak, level)
        low, high = field_bounds(field, lower[None], upper[None])
        self.assertLess(low[0], level)
        self.assertGreaterEqual(high[0], peak)

    def test_true_non_native_peak_is_retained_above_every_native_sample(self):
        _, field = refined_fixture(protect_rivers=False)
        level = 1900.25
        self.assertLess(float(field.native_m.max()), level)
        peak = np.array((4.516311, 4.50029469))
        self.assertGreater(float(field.sample_points(*peak)), level)
        lower, upper = np.array(((4.5, 4.5),)), np.array(((5.5, 5.5),))
        corners = np.array(((4.5, 4.5), (5.5, 4.5), (5.5, 5.5), (4.5, 5.5)))
        self.assertTrue(np.all(field.sample_points(corners[:, 0], corners[:, 1]) < level))
        low, high = field_bounds(field, lower, upper)
        self.assertLess(low[0], level)
        self.assertGreater(high[0], level)

    def test_ranges_converge_without_a_fixed_gain_envelope(self):
        _, field = refined_fixture()
        point = np.array(((3.171, 4.239), (4.173, 3.281), (5.172, 5.191)))
        widths = []
        for radius in (.01, 1e-4, 1e-7):
            low, high = field_bounds(field, point-radius, point+radius)
            widths.append(high-low)
        self.assertTrue(np.all(widths[1] < widths[0]*.05))
        self.assertTrue(np.all(widths[2] < widths[1]*.005))
        self.assertLess(float(np.max(widths[-1])), .02)

    def test_zero_pchip_slope_switch_keeps_convergent_derivative_bounds(self):
        _, field = refined_fixture()
        point = np.array(((4.375, 3.375),))
        widths = []
        for radius in (.004, .002, 1e-7):
            _, _, low, high = field_range_gradient_bounds(field, point-radius, point+radius)
            widths.append(high-low)
        self.assertTrue(np.all(widths[1] < widths[0]*.6))
        self.assertTrue(np.all(widths[2] < widths[1]*1e-4))

    def test_mean_value_bound_excludes_a_thin_tangent_edge(self):
        base, _ = refined_fixture()
        lower = np.array(((4.4999999, 4.499),))
        upper = np.array(((4.5000001, 4.499),))
        low, high = field_bounds(base, lower, upper)
        actual = float(base.sample_points(4.5, 4.499))
        self.assertLess(float(high[0]-low[0]), 1e-8)
        self.assertLessEqual(low[0], actual)
        self.assertGreaterEqual(high[0], actual)

    def test_actual_constant_chart_keeps_exact_height_and_zero_derivatives(self):
        _, field = refined_fixture()
        lower = np.array(((0., 0.), (.5, .5)))
        upper = lower+.5
        low, high, grad_low, grad_high = field_range_gradient_bounds(field, lower, upper)
        np.testing.assert_array_equal(low, (100., 100.))
        np.testing.assert_array_equal(high, low)
        np.testing.assert_array_equal(grad_low, np.zeros((2, 2)))
        np.testing.assert_array_equal(grad_high, grad_low)

    def test_actual_pchip_switch_keeps_the_shared_hermite_slope_dependency(self):
        fixture = json.loads((Path(__file__).parent/'fixtures'
                              /'implicit-terrain-pchip-switch-v97.json').read_text(encoding='utf-8'))
        values = np.asarray(fixture['nativeRelativeElevationM'])
        field = PhysicalTerrainField(values, land_mask=values > 0, sea_level_m=0,
                                     elevation_scale_m=4000, elevation_exponent=1)
        lower, upper = np.array((fixture['lower'],)), np.array((fixture['upper'],))
        low, high, grad_low, grad_high = field_range_gradient_bounds(field, lower, upper)
        # This real box straddles a harmonic-slope switch. The true collected
        # Hermite weight is positive; independent +m/-2m/+m appearances used
        # to invent x gradients on both sides of zero even after 40 levels.
        self.assertLess(grad_high[0, 0], -80.)
        self.assertGreater(grad_low[0, 1], 190.)
        self.assertLess(float(high[0]-low[0]), 1e-9)
        points = lower+(upper-lower)*np.linspace(0, 1, 17)[:, None]
        actual = field.sample_points(points[:, 0], points[:, 1])
        self.assertTrue(np.all(actual >= low[0]))
        self.assertTrue(np.all(actual <= high[0]))

    def test_native_knots_polar_frame_and_longitude_cut_are_all_enclosed(self):
        _, field = refined_fixture()
        lower = np.array(((3.4, 3.4), (0., 0.), (8.8, 8.8), (4.4, 4.4)))
        upper = np.array(((3.6, 3.6), (.2, .2), (9., 9.), (4.6, 4.6)))
        low, high = field_bounds(field, lower, upper)
        fraction = np.linspace(0, 1, 17)
        points = lower[:, None, None, :]+(upper-lower)[:, None, None, :]*np.stack(
            np.meshgrid(fraction, fraction), axis=-1)[None]
        actual = field.sample_points(points[..., 0], points[..., 1])
        self.assertTrue(np.all(low[:, None, None] <= actual))
        self.assertTrue(np.all(actual <= high[:, None, None]))

    def test_actual_coastal_warp_and_crossed_source_patches_are_enclosed(self):
        field = coastal_fixture()
        self.assertGreater(field._landforms.count, 0)
        centres = field._landforms.centres[field._landforms.count:2*field._landforms.count]
        lower, upper = centres-.1, centres+.1
        low, high = field_bounds(field, lower, upper)
        rng = np.random.default_rng(110)
        points = lower[:, None]+(upper-lower)[:, None]*rng.random((len(lower), 131, 2))
        actual = field.sample_points(points[..., 0], points[..., 1])
        self.assertTrue(np.all(low[:, None] <= actual))
        self.assertTrue(np.all(actual <= high[:, None]))

    def test_batching_preserves_bounds_and_does_not_mutate_inputs(self):
        _, field = refined_fixture()
        rng = np.random.default_rng(513)
        lower = rng.uniform(.6, 8.2, (4099, 2))
        upper = lower+.1
        original_lower, original_upper = lower.copy(), upper.copy()
        whole = field_range_gradient_bounds(field, lower, upper)
        first = field_range_gradient_bounds(field, lower[:4096], upper[:4096])
        last = field_range_gradient_bounds(field, lower[4096:], upper[4096:])
        for actual, a, b in zip(whole, first, last, strict=True):
            np.testing.assert_array_equal(actual, np.concatenate((a, b)))
        np.testing.assert_array_equal(lower, original_lower)
        np.testing.assert_array_equal(upper, original_upper)

    def test_invalid_boxes_are_rejected(self):
        _, field = refined_fixture()
        for lower, upper in (((1., 1.), (2., 2.)),
                             (((1., -1.),), ((2., 0.),)),
                             (((1., 1.),), ((2.01, 2.),)),
                             (((2., 2.),), ((1., 3.),))):
            with self.assertRaises(ValueError):
                field_bounds(field, lower, upper)

    def test_direction_matrix_columns_are_native_vectors_and_inputs_are_immutable(self):
        rows,columns=np.indices((9,9))
        values=100+20*(columns+.5)-30*(rows+.5)
        field=PhysicalTerrainField(values,land_mask=values>0,sea_level_m=0,
                                   elevation_scale_m=4000,elevation_exponent=1)
        lower=np.array(((3.12,4.17),(2.31,3.19)))
        upper=lower.copy()
        directions=np.array(((1.,-2.),(1.,.5)))
        original=(lower.copy(),upper.copy(),directions.copy())
        low,high=directional_bounds(field,lower,upper,directions)
        expected=np.broadcast_to((-10.,-55.),low.shape)
        self.assertTrue(np.all(low<=expected));self.assertTrue(np.all(expected<=high))
        self.assertLess(float(np.max(high-low)),1e-9)
        batches=directional_bounds(field,lower,upper,np.broadcast_to(directions,(2,2,2)))
        for first,last in zip((low,high),batches,strict=True):np.testing.assert_array_equal(first,last)
        for first,last in zip((lower,upper,directions),original,strict=True):np.testing.assert_array_equal(first,last)

    def test_seeded_direction_retains_nearest_river_branch_correlation(self):
        base,_=refined_fixture()
        flow=np.full((9,9),-1,dtype=np.int32)
        flow[4,4],flow[4,5]=41,50
        discharge=np.zeros((9,9));discharge[4,4],discharge[4,5]=64,128
        field=RefinedTerrainField(base,seed=1,radius_km=12,flow_to=flow,discharge=discharge,
            river_segments=np.array((((4.5,3.5),(5.5,4.5)),((4.5,4.5),(5.5,5.5)))),
            river_anchors=np.empty((0,2)))
        point=np.array(((4.8,4.3),));lower,upper=point-1e-7,point+1e-7
        _,_,gx,gy=field_range_gradient_bounds(field,lower,upper)
        low,high=directional_bounds(field,lower,upper,((1,1),(1,-1)))
        # The two nearest native channels tie here. Their distance gradients
        # are opposite normals; both directional sums vanish along the fan.
        # Summing the final coordinate hulls invents a 228-wide interval.
        self.assertGreater(float((gy-gx).sum()),200)
        self.assertLess(float(high[0,0]-low[0,0]),.01)
        self.assertLess(high[0,0],-400)
        epsilon=1e-6
        slope=float((field.sample_points(point[:,0]+epsilon,point[:,1]+epsilon)
                    -field.sample_points(point[:,0]-epsilon,point[:,1]-epsilon))[0]/(2*epsilon))
        self.assertLessEqual(low[0,0]-1e-5,slope)
        self.assertGreaterEqual(high[0,0]+1e-5,slope)

    def test_directional_api_rejects_nonfinite_zero_or_misaligned_vectors(self):
        _,field=refined_fixture()
        lower=np.array(((3.,3.),));upper=lower+.1
        for directions in ((1,1),((0,1),(0,1)),((1,np.nan),(1,-1)),np.ones((2,2,2))):
            with self.assertRaises(ValueError):directional_bounds(field,lower,upper,directions)

    def test_inverse_event_encloses_true_arc_and_its_parameter_derivative(self):
        field = coastal_fixture()
        count = field._landforms.count
        centres = field._landforms.centres[count:2*count]
        c = centres[:,0].copy()
        lower,upper = centres[:,1]-.01,centres[:,1]+.01
        original = c.copy(),lower.copy(),upper.copy(),field.native_m.copy()
        a,b,low,high,dl,dh = warped_pchip_event_bounds(field,lower,upper,c)
        parameter = lower[:,None]+(upper-lower)[:,None]*np.linspace(0,1,41)
        x,y = field.inverse_ground_coordinates(c[:,None],parameter)
        points = np.stack((x,y),axis=-1)
        self.assertTrue(np.all(a[:,None]<=points));self.assertTrue(np.all(points<=b[:,None]))
        heights = field.sample_points(x,y)
        self.assertTrue(np.all(low[:,None]<=heights));self.assertTrue(np.all(heights<=high[:,None]))
        epsilon = 1e-6
        x0,y0 = field.inverse_ground_coordinates(c[:,None],parameter-epsilon)
        x1,y1 = field.inverse_ground_coordinates(c[:,None],parameter+epsilon)
        derivative = (field.sample_points(x1,y1)-field.sample_points(x0,y0))/(2*epsilon)
        self.assertTrue(np.all(dl[:,None]-1e-7<=derivative))
        self.assertTrue(np.all(derivative<=dh[:,None]+1e-7))
        # With no drainage detail F=B(W). Along inverse_W(c,t), the derivative
        # is the source B_y. The physical F_y alone omits the x motion of gamma.
        gy = (field.sample_points(x,y+epsilon)-field.sample_points(x,y-epsilon))/(2*epsilon)
        self.assertGreater(float(np.max(abs(gy-derivative))),.05)
        for actual,expected in zip((c,lower,upper,field.native_m),original,strict=True):
            np.testing.assert_array_equal(actual,expected)

    def test_inverse_event_ranges_contract_and_keep_unwrapped_longitudes(self):
        field = coastal_fixture()
        index = field._landforms.count
        centre = field._landforms.centres[index]
        widths = []
        for radius in (.01,1e-4,1e-7):
            lower,upper = np.array((centre[1]-radius,)),np.array((centre[1]+radius,))
            a,b,low,high,dl,dh = warped_pchip_event_bounds(field,lower,upper,centre[0]+field.width)
            widths.append((b-a)[0])
            x,y = field.inverse_ground_coordinates(np.array((centre[0]+field.width,)),np.array((centre[1],)))
            self.assertTrue(np.all(a[0]<=np.array((x[0],y[0]))))
            self.assertTrue(np.all(np.array((x[0],y[0]))<=b[0]))
            height = float(field.sample_points(x,y)[0])
            self.assertLessEqual(low[0],height);self.assertGreaterEqual(high[0],height)
            self.assertGreater(a[0,0],field.width)
        self.assertTrue(np.all(widths[1]<widths[0]*.02))
        self.assertTrue(np.all(widths[2]<widths[1]*.002))

    def test_inverse_event_accounts_for_numerical_inverse_residual(self):
        field = coastal_fixture()
        centre = field._landforms.centres[field._landforms.count]
        source_x,source_y = np.array((centre[0],)),np.array((centre[1],))
        original = field.inverse_ground_coordinates
        exact = np.array(original(source_x,source_y)).T
        def perturbed(x,y):
            px,py = original(x,y)
            return px+2e-5,py-3e-5
        field.inverse_ground_coordinates = perturbed
        lower,upper,low,high,_dl,_dh = warped_pchip_event_bounds(field,source_y,source_y,source_x)
        self.assertTrue(np.all(lower<=exact));self.assertTrue(np.all(exact<=upper))
        height = float(field.sample_points(exact[:,0],exact[:,1])[0])
        self.assertLessEqual(low[0],height);self.assertGreaterEqual(high[0],height)

    def test_inverse_event_identity_plateau_and_frame_remain_exact(self):
        _,field = refined_fixture()
        lower,upper = np.array((0.,.1)),np.array((.5,.6))
        a,b,low,high,dl,dh = warped_pchip_event_bounds(field,lower,upper,np.array((0.,1.2)))
        np.testing.assert_array_equal(low,(100.,100.));np.testing.assert_array_equal(high,low)
        np.testing.assert_array_equal(dl,(0.,0.));np.testing.assert_array_equal(dh,dl)
        self.assertEqual(a[0,1],0.);self.assertEqual(b[0,1],.5)

    def test_inverse_event_rejects_foreign_model_and_invalid_source_intervals(self):
        base,field = refined_fixture()
        with self.assertRaises(TypeError):warped_pchip_event_bounds(base,np.array((1.,)),np.array((2.,)),1.)
        for lower,upper,c in (([1.],[2.01],1.),([-1.],[0.],1.),([2.],[1.],1.),
                              ([1.],[2.],np.nan),([[1.]],[[2.]],1.),([1.,2.],[2.,3.],[1.])):
            with self.assertRaises(ValueError):warped_pchip_event_bounds(field,lower,upper,c)


if __name__ == '__main__':
    unittest.main()
