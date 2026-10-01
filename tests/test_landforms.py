"""Resolved physical evidence, coastal exclusions and bounded saved identities."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from xml.etree import ElementTree

import numpy as np

from world_atlas.core.landforms import LANDFORM_TYPES, derive_landform_inventory, _isthmus_support
from world_atlas.core.cartographic_symbols import AREA_PATTERNS, LANDFORM_STYLES, geographic_legend, pattern_markup
from world_atlas.core.society.geographic_features import enrich_saved_geographic_features
from world_atlas.core.society.model import GeographicFeature
from world_atlas.core.society.names import extend_geographic_features
from world_atlas.core.society.onomastics import geographic_name_candidates
from world_atlas.core.society.world_identity import assign_world_identity
from world_atlas.core.render import _geographic_symbol_overlay, _toponymy_overlay
from test_names import _lexicon
from test_presentation import _fixture
from test_world_identity import _society


def fixture(shape=(31,61)):
    grid = SimpleNamespace(shape=shape,water=np.zeros(shape,dtype=np.uint8),
        elevation=np.full(shape,.3),snow=np.zeros(shape,dtype=bool),
        river_order=np.zeros(shape,dtype=np.uint8),
        metadata={"planet":{"radiusKm":6371.0},"extents":{"north":2.5,"south":-2.5,"west":-5.,"east":5.}})
    thematic = SimpleNamespace(land_potential=np.full(shape,.5),physiography=SimpleNamespace(
        wetland=np.zeros(shape,dtype=bool),wetland_support=np.zeros(shape,dtype=np.float32),desert=np.zeros(shape,dtype=bool)))
    return grid,thematic,np.full(shape,500.)


class LandformTests(unittest.TestCase):
    def test_sea_zero_is_not_a_cliff_or_false_foothill(self):
        grid,thematic,raw = fixture()
        grid.water[:,:12]=1
        raw[:,:12]=-6000
        inventory = derive_landform_inventory(grid,thematic,raw_elevation_m=raw)
        for kind in ("ridge","hills","gorge","foothills","steep-slope"):
            self.assertFalse(inventory.masks[kind].any())
        self.assertFalse(inventory.direction_east.any())
        self.assertFalse(inventory.direction_north.any())
        self.assertTrue(all(not mask[grid.water!=0].any() for mask in inventory.masks.values()))

    def test_deep_valley_requires_actual_river_context_and_side_relief(self):
        grid,thematic,raw = fixture()
        columns = np.arange(grid.shape[1])
        raw[:]=200+400*np.minimum(np.abs(columns-30),5)
        inventory = derive_landform_inventory(grid,thematic,raw_elevation_m=raw)
        self.assertFalse(inventory.masks["gorge"].any())
        grid.river_order[:,30]=3
        inventory = derive_landform_inventory(grid,thematic,raw_elevation_m=raw)
        self.assertTrue(inventory.masks["gorge"][15,30])
        self.assertFalse(inventory.masks["gorge"][15,20])
        # A flat valley floor has zero center slope; its measured side walls
        # still give evidence of a deep regional valley.
        self.assertEqual(inventory.direction_east[15,30],0)

    def test_snow_and_aridity_need_physical_relief_not_only_climate_flags(self):
        grid,thematic,raw = fixture()
        grid.elevation[:]=.8
        grid.snow[:]=True
        thematic.physiography.desert[:]=True
        flat = derive_landform_inventory(grid,thematic,raw_elevation_m=raw)
        self.assertFalse(flat.masks["snow-mountain"].any())
        self.assertFalse(flat.masks["arid-upland"].any())
        raw[12:19,26:35]=3000
        supported = derive_landform_inventory(grid,thematic,raw_elevation_m=raw)
        self.assertTrue(supported.masks["snow-mountain"].any())
        self.assertTrue(supported.masks["arid-upland"].any())
        grid.snow[:]=False
        self.assertFalse(derive_landform_inventory(grid,thematic,raw_elevation_m=raw).masks["snow-mountain"].any())

    def test_lake_shores_and_off_grid_edges_do_not_become_sea_capes(self):
        grid,thematic,raw = fixture()
        grid.water[:]=2
        grid.water[5:26,15:46]=0
        grid.water[14:17,46:54]=0
        raw[:]=np.where(grid.water==0,500,-10)
        inventory = derive_landform_inventory(grid,thematic,raw_elevation_m=raw)
        for kind in ("cape","peninsula","isthmus"):
            self.assertFalse(inventory.masks[kind].any())
        grid.water[:]=0
        raw[:]=500
        inventory = derive_landform_inventory(grid,thematic,raw_elevation_m=raw)
        self.assertFalse(inventory.masks["cape"].any())
        self.assertFalse(inventory.masks["peninsula"].any())

    def test_isthmus_cut_really_separates_two_connected_land_arms(self):
        land=np.zeros((35,45),dtype=bool)
        land[3:13,13:32]=True
        land[22:32,13:32]=True
        land[13:22,21:24]=True
        necks=_isthmus_support(land,~land)
        self.assertTrue(necks[17,22])
        # A parallel second connection invalidates the supposed local neck.
        land[10:25,27:29]=True
        self.assertFalse(_isthmus_support(land,~land)[17,22])
        island=np.zeros_like(land)
        island[16:19,21:24]=True
        self.assertFalse(_isthmus_support(island,~island).any())

    def test_downslope_points_physically_down_and_never_across_false_sea_gradient(self):
        grid,thematic,raw=fixture()
        raw[:]=1000+10*np.arange(grid.shape[1])[None,:]+5*np.arange(grid.shape[0])[:,None]
        inventory=derive_landform_inventory(grid,thematic,raw_elevation_m=raw)
        self.assertLess(inventory.direction_east[15,30],0)
        self.assertGreater(inventory.direction_north[15,30],0)
        self.assertAlmostEqual(float(np.hypot(inventory.direction_east[15,30],inventory.direction_north[15,30])),1,places=6)

    def test_raw_metres_required(self):
        grid,thematic,raw=fixture()
        for invalid in (np.zeros(grid.shape),np.full(grid.shape,np.nan),raw[:-1]):
            with self.assertRaises(ValueError):
                derive_landform_inventory(grid,thematic,raw_elevation_m=invalid)

    def test_all_new_types_have_model_naming_and_shared_style(self):
        self.assertEqual(set(LANDFORM_TYPES),set(LANDFORM_STYLES))
        for kind in LANDFORM_TYPES:
            GeographicFeature(f"{kind}-01",kind,"真实地貌",1,1,1,"detail")
            self.assertTrue(geographic_name_candidates("lineage-s01-01",kind,seed=11))
            self.assertEqual(set(LANDFORM_STYLES[kind]),{"label","color","representation","min_scale","max_scale"})
        for kind in AREA_PATTERNS:
            node=ElementTree.fromstring(pattern_markup(kind,"actual-pattern"))
            self.assertEqual(node.get("patternUnits"),"userSpaceOnUse")
            self.assertEqual(node.get("data-landform-pattern"),kind)
        self.assertNotIn("沙地",geographic_legend())

    def test_enrichment_preserves_existing_records_all_other_layers_and_is_idempotent(self):
        grid,society=_fixture()
        physical,thematic,raw=fixture(grid.shape)
        original=GeographicFeature("mountain-01","mountain","原有山系",4,6,1,"secondary")
        society=replace(society,geographic_features=(original,))
        # A supplied, independently classed native component makes this test
        # about identity preservation rather than mirroring a classifier.
        inventory=SimpleNamespace(masks={kind:np.zeros(grid.shape,dtype=bool) for kind in LANDFORM_TYPES})
        inventory.masks["hills"][2:6,3:9]=True
        records,evidence=extend_geographic_features(physical,society.cultures,society.geographic_features,
                                                   _lexicon(),inventory=inventory)
        self.assertIs(records[0],original)
        self.assertEqual(len(evidence),1)
        again,more=extend_geographic_features(physical,society.cultures,records,_lexicon(),inventory=inventory)
        self.assertEqual(records,again)
        self.assertEqual(more,[])
        enriched,report=enrich_saved_geographic_features(physical,society,thematic=thematic,
                                                        raw_elevation_m=raw,lexicon=_lexicon())
        self.assertIs(enriched.geographic_features[0],original)
        for field in ("population","settlements","transport","cultures","religions","politics","provinces"):
            self.assertIs(getattr(enriched,field),getattr(society,field))
        self.assertTrue(report["originalRecordsPreserved"])

    def test_peak_point_numbers_use_raw_metres_and_new_labels_render(self):
        grid,society=_fixture()
        society=replace(society,geographic_features=(GeographicFeature("peak-01","peak","真峰",4,6,1,"detail"),
            *(GeographicFeature(f"{kind}-01",kind,kind,4,7,1,"detail") for kind in LANDFORM_TYPES)))
        raw=np.where(grid.water==0,1247.6,-5)
        markup=_geographic_symbol_overlay(grid,None,society,raw_elevation_m=raw)
        document=ElementTree.fromstring(markup)
        markers=[node for node in document.iter() if node.get("data-geographic-symbol")]
        self.assertEqual(len(markers),1)
        height=next(node for node in document.iter() if node.get("data-height-m"))
        self.assertEqual(float(height.get("data-height-m")),1247.6)
        self.assertEqual(height.text,"1248")
        self.assertNotIn("mark-desert",markup)
        labels=ElementTree.fromstring(_toponymy_overlay(grid,society))
        self.assertEqual(len([node for node in labels.iter() if node.get("data-feature-type")]),13)

    def test_saved_additions_use_canonical_naming_replay_without_renaming_prior_records(self):
        society=_society()
        physical,thematic,raw=fixture(society.population.population_weight.shape)
        physical.metadata["societyGeneration"]={"namingSeed":123,"forbiddenNames":[]}
        prior=GeographicFeature("plain-01","plain","原有平原",2,1,1,"secondary")
        society=assign_world_identity(replace(society,geographic_features=(prior,)),seed=123)
        raw[:,2:6]=1100
        enriched,report=enrich_saved_geographic_features(physical,society,thematic=thematic,
                                                        raw_elevation_m=raw,lexicon=_lexicon())
        self.assertGreater(len(report["addedFeatures"]),0)
        self.assertEqual(enriched.geographic_features[0],society.geographic_features[0])
        self.assertEqual(assign_world_identity(enriched,seed=123).geographic_features,enriched.geographic_features)
        self.assertEqual(enriched.settlements,society.settlements)
        self.assertEqual(enriched.politics.states,society.politics.states)


if __name__=="__main__":
    unittest.main()
