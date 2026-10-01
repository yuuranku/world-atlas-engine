"""Place cards are projections of saved records, not new simulated facts."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from test_world_identity import _society
from world_atlas.core.atlas_ui import build_place_index, ui_markup, write_map_app_assets, THEMES
from world_atlas.core.society.model import GovernmentForm


def saved_fixture():
    society = _society()
    society = replace(society, politics=replace(society.politics,
        government_forms=(GovernmentForm(1, "city-republic", "城邦共和制", "fixture local government"),)))
    grid = SimpleNamespace(shape=(6, 8), metadata={"extents": {"west": -180., "east": 180., "north": 90., "south": -90.},
        "worldProfile": {"technologyEra": "ancient"}, "planet": {"radiusKm": 6400.}}, content_digest=lambda: "test-grid")
    return grid, society


def saved_locations(society):
    return {city.identifier: (city.row+.5, city.column+.5) for city in society.settlements}


class AtlasUITests(unittest.TestCase):
    def test_place_fields_use_saved_ranges_centres_and_native_ownership(self):
        grid, society = saved_fixture()
        places = {place["id"]: place for place in build_place_index(grid, society, settlement_locations=saved_locations(society))}
        city = places["city:east-port"]
        self.assertEqual(city["native"], [1.5, 2.5])
        self.assertEqual((city["longitude"], city["latitude"]), (-112.5, 15.))
        self.assertEqual(city["population"], [20_000, 40_000])
        self.assertEqual(city["state"], {"id": 1, "name": "旧东"})
        self.assertEqual(city["province"], {"id": 1, "name": "旧东府"})
        self.assertTrue(city["isCapital"])
        self.assertEqual(city["holyReligion"], "旧日教")
        self.assertEqual(places["state:2"]["bounds"], [4, 0, 8, 6])
        self.assertEqual(places["province:2"]["capitalName"], "Old Gate")
        self.assertNotIn("population", places["province:2"])
        self.assertEqual(places["geography:west-river"]["typeLabel"], "河流")
        self.assertEqual(places["geography:west-river"]["ownerLabel"], "标注位置")

    def test_inhabited_frontier_is_not_renamed_uninhabited_or_given_an_owner(self):
        grid, society = saved_fixture()
        states = society.politics.state_id.copy()
        provinces = society.provinces.province_id.copy()
        frontier = society.politics.frontier.copy()
        states[4, 2] = provinces[4, 2] = 0
        frontier[4, 2] = True
        modified = replace(society, politics=replace(society.politics, state_id=states, frontier=frontier),
                           provinces=replace(society.provinces, province_id=provinces))
        city = next(place for place in build_place_index(grid, modified, settlement_locations=saved_locations(modified)) if place["id"] == "city:east-market")
        self.assertTrue(city["frontier"])
        self.assertIsNone(city["state"])
        self.assertIsNone(city["province"])
        self.assertEqual(city["population"], [4_000, 8_000])
        self.assertEqual(int(society.politics.state_id[4, 2]), 1)

    def test_markup_has_real_controls_and_escaped_world_text_without_old_select(self):
        markup = ui_markup(world_name='<罗特>', summary='80国 & 454省', legends={"vegetation":'<p>真实覆盖图例</p>'})
        self.assertIn('&lt;罗特&gt;', markup)
        self.assertIn('80国 &amp; 454省', markup)
        self.assertEqual(markup.count('data-theme-button='), len(THEMES))
        self.assertEqual(markup.count('data-season-button='), 4)
        self.assertEqual(markup.count('type="checkbox" data-layer='), 19)
        self.assertEqual(markup.count('<select'),1)
        self.assertIn('<select id="navigation-mode"',markup)
        self.assertNotIn('city-focus', markup)
        self.assertIn('id="map-status"', markup)
        self.assertIn('data-legend-key="vegetation"', markup)
        self.assertNotIn('当前位置', markup)

    def test_claimed_frontier_retains_final_country_without_stateless_label(self):
        grid, society = saved_fixture()
        frontier = society.politics.frontier.copy()
        frontier[4, 2] = True
        modified = replace(society, politics=replace(society.politics, frontier=frontier))
        city = next(place for place in build_place_index(grid, modified, settlement_locations=saved_locations(modified))
                    if place['id'] == 'city:east-market')
        self.assertEqual(city['state'], {'id': 1, 'name': '旧东'})
        self.assertEqual(city['population'], [4_000, 8_000])
        self.assertFalse(city['frontier'])

    def test_stateless_city_does_not_depend_on_original_frontier_mask(self):
        grid, society = saved_fixture()
        states = society.politics.state_id.copy()
        provinces = society.provinces.province_id.copy()
        frontier = society.politics.frontier.copy()
        states[4, 2] = provinces[4, 2] = 0
        frontier[4, 2] = False
        modified = replace(society, politics=replace(society.politics, state_id=states, frontier=frontier),
                           provinces=replace(society.provinces, province_id=provinces))
        city = next(place for place in build_place_index(grid, modified, settlement_locations=saved_locations(modified))
                    if place['id'] == 'city:east-market')
        self.assertTrue(city['frontier'])
        self.assertIsNone(city['state'])

    def test_written_assets_and_place_index_are_the_actual_source_and_display_anchors(self):
        grid, society = saved_fixture()
        locations = saved_locations(society)
        locations['east-port'] = (2.123, 1.987)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_map_app_assets(output, grid=grid, society=society, settlement_locations=locations)
            places = json.loads((output/'place-index.json').read_text(encoding='utf-8'))
            city = next(place for place in places if place['id'] == 'city:east-port')
            self.assertEqual(city['native'], [1.987, 2.123])
            self.assertEqual(city['name'], '旧东港')
            self.assertEqual(city['eraLabel'], '古典时代')
            self.assertEqual(city['administrationLabel'], '中央常设治理')
            self.assertTrue((output/'governance.json').exists())
            source = Path(__file__).resolve().parents[1]/'src/world_atlas/core/web'
            for filename in ('atlas-ui.css', 'atlas-ui.js', 'atlas-tiles.js', 'atlas-overview.js', 'atlas-interaction.js', 'atlas-ruler.js',
                             'city-detail.js', 'city-site.js', 'city-map.js'):
                self.assertEqual((output/filename).read_bytes(), (source/filename).read_bytes())
            self.assertFalse((output/'settlement-index.js').exists())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaises(KeyError):
                write_map_app_assets(output, grid=grid, society=society, settlement_locations={})
            self.assertEqual(list(output.iterdir()), [])

    def test_city_region_anchors_use_delivered_positions_but_owners_use_canonical_cells(self):
        grid, society = saved_fixture()
        locations = saved_locations(society)
        locations['east-port'] = (2.123, 1.987)
        places = {place['id']: place for place in build_place_index(grid, society, settlement_locations=locations)}
        self.assertEqual(places['city:east-port']['native'], [1.987, 2.123])
        self.assertEqual(places['state:1']['native'], [1.987, 2.123])
        self.assertEqual(places['province:1']['native'], [1.987, 2.123])
        self.assertEqual(places['city:east-port']['state']['id'], 1)
        with self.assertRaises(KeyError):
            build_place_index(grid, society, settlement_locations={})


if __name__ == '__main__':
    unittest.main()
