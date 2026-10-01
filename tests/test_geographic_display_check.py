"""Actual delivered text and source summit counterexamples for the independent gate."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


spec = importlib.util.spec_from_file_location(
    "geographic_consumer",Path(__file__).resolve().parents[1]/"scripts/check_geographic_display.py")
consumer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(consumer)


def fixture():
    shape = (21,35)
    grid = SimpleNamespace(shape=shape,elevation=np.full(shape,.2),water=np.zeros(shape,dtype=np.uint8),
                           snow=np.zeros(shape,dtype=bool))
    grid.elevation[3:17,5:25] = .8
    raw = np.full(shape,100.)
    raw[3:17,5:25] = 500
    raw[10,15] = 2000
    features = tuple(SimpleNamespace(identifier=identifier,feature_type=kind,name=name,row=row,column=column,tier=tier)
                     for identifier,kind,name,row,column,tier in
                     (("mountain-01","mountain","清风山系",10,10,"major"),
                      ("peak-01","peak","真峰",10,15,"detail"),("plain-01","plain","平原",18,30,"secondary")))
    scene = ('<svg><g id="geographic-labels"><defs><path id="geographic-path-mountain-01" '
             'd="M8.5,10.5 Q10.5,10.5 12.5,10.5"/></defs>'
             '<text data-feature-type="mountain" data-feature-tier="major">'
             '<textPath href="#geographic-path-mountain-01">清风山系</textPath></text>'
             '<text data-feature-type="peak" data-feature-tier="detail" x="15.5" y="10.5">真峰</text>'
             '<text data-feature-type="plain" data-feature-tier="secondary" x="30.5" y="18.5">平原</text></g>'
             '<g data-geographic-symbol="peak" data-map-x="15.5" data-map-y="10.5">'
             '<use href="#mark-peak"/><text data-height-m="2000" data-height-datum="model-sea-level">2000</text></g></svg>')
    return grid,raw,SimpleNamespace(geographic_features=features),scene


class GeographicDisplayCheckTests(unittest.TestCase):
    def test_actual_scene_and_unclipped_summit_support_pass(self):
        grid,raw,society,scene = fixture()
        report = consumer.check_scene(grid,raw,society,scene)
        self.assertEqual(report["status"],"ok")
        self.assertTrue(report["peaks"][0]["strictRawLocalMaximum"])
        self.assertEqual(report["peaks"][0]["elevationMeters"],2000)
        self.assertEqual(report["mountains"][0]["outsideOwnComponent"],0)

    def test_clipped_normalized_rim_cannot_mask_wrong_raw_summit(self):
        grid,raw,society,scene = fixture()
        grid.elevation[3:17,5:25] = 1
        raw[10,16] = 2200
        report = consumer.check_scene(grid,raw,society,scene)
        self.assertEqual(report["status"],"failed")
        self.assertFalse(report["peaks"][0]["strictRawLocalMaximum"])

    def test_peak_number_and_model_datum_are_verified_against_unclipped_metres(self):
        grid,raw,society,scene=fixture()
        for old,new in (('data-height-m="2000"','data-height-m=".94"'),
                        ('>2000</text>','>7000</text>'),('model-sea-level','unknown-datum')):
            with self.subTest(new=new):
                report=consumer.check_scene(grid,raw,society,scene.replace(old,new))
                self.assertEqual(report["status"],"failed")
                self.assertTrue(report["peakHeightNumberErrors"])

    def test_curve_on_a_neighboring_highland_fails_own_component(self):
        grid,raw,society,scene = fixture()
        grid.elevation[3:17,27:34] = .9
        scene = scene.replace("M8.5,10.5 Q10.5,10.5 12.5,10.5","M28.5,10.5 Q30.5,10.5 32.5,10.5")
        report = consumer.check_scene(grid,raw,society,scene)
        self.assertEqual(report["status"],"failed")
        self.assertGreater(report["mountains"][0]["outsideOwnComponent"],0)

    def test_compact_label_preserves_anchor_and_natural_glyphs(self):
        grid,raw,society,scene = fixture()
        scene = scene.replace('<text data-feature-type="mountain" data-feature-tier="major">'
                              '<textPath href="#geographic-path-mountain-01">清风山系</textPath></text>',
                              '<text data-feature-type="mountain" data-feature-tier="major" x="10.5" y="10.5">清风山系</text>')
        self.assertEqual(consumer.check_scene(grid,raw,society,scene)["status"],"ok")
        for bad in (scene.replace('x="10.5"','x="11.5"'),scene.replace('x="10.5"','textLength="1" x="10.5"')):
            self.assertEqual(consumer.check_scene(grid,raw,society,bad)["status"],"failed")

    def test_compressed_curve_missing_name_and_wrong_summit_marker_fail(self):
        for mutate in (lambda scene:scene.replace('<textPath href=','<textPath lengthAdjust="spacingAndGlyphs" textLength="1" href='),
                       lambda scene:scene.replace(">平原</text>",">伪名</text>"),
                       lambda scene:scene.replace('data-map-x="15.5"','data-map-x="14.5"')):
            grid,raw,society,scene = fixture()
            self.assertEqual(consumer.check_scene(grid,raw,society,mutate(scene))["status"],"failed")

    def test_longitude_seam_is_one_physical_component(self):
        grid,raw,_,_ = fixture()
        grid.elevation[:] = .2
        grid.elevation[3:17,0] = .8
        grid.elevation[3:17,-1] = .8
        feature = SimpleNamespace(identifier="mountain-01",feature_type="mountain",name="跨线山系",
                                  row=10,column=0,tier="major")
        scene = ('<svg><g id="geographic-labels"><defs><path id="geographic-path-mountain-01" '
                 'd="M-.5,10.5 Q.5,10.5 .5,11.5"/></defs>'
                 '<text data-feature-type="mountain" data-feature-tier="major">'
                 '<textPath href="#geographic-path-mountain-01">跨线山系</textPath></text></g></svg>')
        report = consumer.check_scene(grid,raw,SimpleNamespace(geographic_features=(feature,)),scene)
        self.assertEqual(report["status"],"ok")


if __name__ == "__main__":
    unittest.main()
