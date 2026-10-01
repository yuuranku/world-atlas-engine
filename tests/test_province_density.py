"""Province detail follows real administrative seats and governing workload."""
from types import SimpleNamespace
import unittest

from world_atlas.core.society.provinces import _province_core_targets, _select_province_cores


class ProvinceDensityTests(unittest.TestCase):
    def targets(self, areas, capacities, densities=None):
        return _province_core_targets(
            areas, densities or dict.fromkeys(areas, 1.0),
            dict.fromkeys(areas, "central-bureaucracy"), capacities,
        )

    def test_eighty_countries_support_the_requested_atlas_detail(self):
        areas = dict.fromkeys(range(1, 81), 1_000)
        targets = self.targets(areas, dict.fromkeys(areas, 10))
        self.assertEqual(sum(targets.values()), 448)
        self.assertLessEqual(max(targets.values()) - min(targets.values()), 1)

    def test_single_town_states_release_capacity_to_large_settled_countries(self):
        areas = {**dict.fromkeys(range(1, 10), 20), 10: 8_000}
        capacities = {**dict.fromkeys(range(1, 10), 1), 10: 80}
        targets = self.targets(areas, capacities)
        self.assertTrue(all(targets[identifier] == 1 for identifier in range(1, 10)))
        self.assertGreater(targets[10], 18)
        self.assertEqual(sum(targets.values()), 56)

    def test_no_province_is_invented_without_a_seat(self):
        targets = self.targets({1: 1_000, 2: 4_000}, {1: 1, 2: 2})
        self.assertEqual(targets, {1: 1, 2: 2})

    def test_dense_equal_area_state_gets_more_local_administration(self):
        targets = self.targets({1: 1_000, 2: 1_000}, {1: 30, 2: 30}, {1: 0.3, 2: 2.0})
        self.assertGreater(targets[2], targets[1])

    def test_core_selection_does_not_reintroduce_the_old_eighteen_seat_ceiling(self):
        cities = tuple(SimpleNamespace(identifier=f"town-{i}", row=2, column=i * 5,
            score=1.0, tier="town") for i in range(32))
        chosen = _select_province_cores(cities, capital_identifier="town-0", count=25, width=200)
        self.assertEqual(len(chosen), 25)
        self.assertEqual(chosen[0].identifier, "town-0")
        self.assertEqual(len({item.identifier for item in chosen}), 25)


if __name__ == "__main__":
    unittest.main()
