"""Consumer regressions for stale population summaries after owner changes."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


_spec = importlib.util.spec_from_file_location(
    "administrative_population_consumer",
    Path(__file__).resolve().parents[1] / "scripts/check_administrative_population.py",
)
_consumer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_consumer)


class AdministrativePopulationConsumerTests(unittest.TestCase):
    def fixture(self):
        grid = SimpleNamespace(shape=(5, 2), metadata={
            "extents": {"north": 90, "south": -90, "east": 180, "west": -180},
            "planet": {"radiusKm": 6400},
        })
        owners = np.full(grid.shape, 2, dtype=np.int16)
        owners[[0, -1]] = 1
        states = [
            SimpleNamespace(identifier=1, name="极地国", population_min=40_000, population_max=60_000),
            SimpleNamespace(identifier=2, name="中纬国", population_min=60_000, population_max=100_000),
        ]
        provinces = [
            SimpleNamespace(identifier=1, name="高纬省", area_cells=4, population_density_class="dense"),
            SimpleNamespace(identifier=2, name="中纬省", area_cells=6, population_density_class="settled"),
        ]
        society = SimpleNamespace(
            population=SimpleNamespace(population_weight=np.full(grid.shape, .1),
                                       population_min=100_000, population_max=160_000),
            politics=SimpleNamespace(state_id=owners, states=states),
            provinces=SimpleNamespace(province_id=owners.copy(), provinces=provinces),
        )
        return grid, society

    def test_equal_native_headcounts_have_different_spherical_density(self):
        grid, society = self.fixture()
        report = _consumer.check_population_records(grid, society)
        self.assertFalse(report["statePopulationMismatches"])
        self.assertFalse(report["provinceAreaOrDensityMismatches"])
        self.assertAlmostEqual(report["sphericalArea"]["controlledAreaKm2"],
                               4 * np.pi * 6400 ** 2, places=6)
        self.assertEqual([row["expectedDensityClass"] for row in report["provinceChecks"]],
                         ["dense", "settled"])
        self.assertGreater(report["provinceChecks"][0]["densityPersonsPerKm2"],
                           report["provinceChecks"][1]["densityPersonsPerKm2"])

    def test_stale_country_interval_is_reported_with_exact_expected_values(self):
        grid, society = self.fixture()
        society.politics.states[0].population_min = 80_000
        report = _consumer.check_population_records(grid, society)
        self.assertEqual(len(report["statePopulationMismatches"]), 1)
        self.assertEqual(report["statePopulationMismatches"][0]["expectedRange"], [40_000, 60_000])
        self.assertEqual(report["statePopulationMismatches"][0]["savedRange"], [80_000, 60_000])

    def test_stale_province_area_and_density_are_reported(self):
        grid, society = self.fixture()
        society.provinces.provinces[0].area_cells = 5
        society.provinces.provinces[0].population_density_class = "settled"
        mismatch = _consumer.check_population_records(grid, society)["provinceAreaOrDensityMismatches"]
        self.assertEqual(len(mismatch), 1)
        self.assertEqual(mismatch[0]["expectedAreaCells"], 4)
        self.assertEqual(mismatch[0]["expectedDensityClass"], "dense")

    def test_country_rounding_keeps_original_round_to_even(self):
        grid, society = self.fixture()
        society.population.population_weight[:] = 0
        society.population.population_weight[0, 0] = .25
        society.population.population_weight[2, 0] = .75
        society.population.population_max = 200_000
        society.politics.states[0].population_min = 20_000
        society.politics.states[0].population_max = 50_000
        society.politics.states[1].population_min = 80_000
        society.politics.states[1].population_max = 150_000
        checks = _consumer.check_population_records(grid, society)["stateChecks"]
        self.assertTrue(all(row["matches"] for row in checks))
        self.assertEqual(checks[0]["expectedRange"], [20_000, 50_000])


if __name__ == "__main__":
    unittest.main()
