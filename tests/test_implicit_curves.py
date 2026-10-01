import unittest

import numpy as np

from world_atlas.core.implicit_curves import adaptive_curve_paths


class ImplicitCurveTests(unittest.TestCase):
    def test_tiny_true_circular_branch_keeps_curvature_and_original_ports(self):
        radius = .00001
        starts = np.array(((radius, 0.),))
        ends = np.array(((0., radius),))
        before = starts.copy(), ends.copy()

        def sections(first, last, owner):
            a = np.arctan2(first[:, 1], first[:, 0])
            b = np.arctan2(last[:, 1], last[:, 0])
            angle = a[:, None]+(b-a)[:, None]*np.array((.25,.5,.75))
            return radius*np.stack((np.cos(angle), np.sin(angle)), axis=2)

        points, = adaptive_curve_paths(starts, ends, sections)
        self.assertGreater(len(points), 16)
        np.testing.assert_array_equal(points[0], starts[0])
        np.testing.assert_array_equal(points[-1], ends[0])
        np.testing.assert_allclose(np.linalg.norm(points, axis=1), radius, atol=1e-20)
        np.testing.assert_array_equal(starts, before[0])
        np.testing.assert_array_equal(ends, before[1])
        chord = np.linalg.norm(np.diff(points, axis=0), axis=1)
        radial_error = radius-np.linalg.norm((points[:-1]+points[1:])*.5, axis=1)
        self.assertLess(float(np.max(radial_error/chord)), .02)

    def test_parametrization_along_straight_chord_does_not_invent_vertices(self):
        def sections(first,last,owner):
            return first[:,None,:]+(last-first)[:,None,:]*np.array((.01,.3,.99))[None,:,None]
        first,last=np.array(((1.,2.),(3.,4.))),np.array(((2.,5.),(6.,4.)))
        paths=adaptive_curve_paths(first,last,sections)
        self.assertEqual([len(p)for p in paths],[2,2])
        for p,a,b in zip(paths,first,last,strict=True):
            np.testing.assert_array_equal(p,np.array((a,b)))

    def test_branch_owner_survives_adaptive_subdivision(self):
        curvatures=np.array((1.,7.))
        first=np.array(((0.,0.),(0.,0.)))
        last=np.column_stack((np.ones(2),curvatures))
        def sections(a,b,owner):
            x=a[:,None,0]+(b-a)[:,None,0]*np.array((.25,.5,.75))
            return np.stack((x,curvatures[owner,None]*x*x),axis=2)
        paths=adaptive_curve_paths(first,last,sections)
        for i,p in enumerate(paths):
            np.testing.assert_allclose(p[:,1],curvatures[i]*p[:,0]**2,atol=1e-14)

    def test_invalid_or_unresolved_true_sections_fail(self):
        first,last=np.array(((0.,0.),)),np.array(((1.,0.),))
        with self.assertRaisesRegex(ValueError,'three finite'):
            adaptive_curve_paths(first,last,lambda a,b,o:np.zeros((len(a),2,2)))
        with self.assertRaisesRegex(ValueError,'matching finite'):
            adaptive_curve_paths(first,np.array(((np.nan,0.),)),lambda a,b,o:None)


if __name__=='__main__':
    unittest.main()
