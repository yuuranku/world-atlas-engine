"""Regression contracts for connected, core-rooted tectonic plates."""

from __future__ import annotations

import unittest

import numpy as np

from world_atlas.core.planet_morphology import sample_world_morphology
from world_atlas.core.procedural_planet import (
    _curved_boundary_corridors,
    _derive_seed,
    _plate_definitions,
)
from world_atlas.physical.planetary_grid import build_lat_lon_grid, lon_lat_to_unit_vector
from world_atlas.physical.tectonics import (
    _components,
    build_plate_fields,
    clean_orphan_components,
)


_RADIUS_KM = 6400.0


class PlateTopologyTests(unittest.TestCase):
    def test_declared_core_is_the_component_root_even_when_a_larger_orphan_exists(self):
        grid = build_lat_lon_grid(8, 16)
        labels = np.full(grid.shape, "other", dtype=object)
        # The plate's real core is a small connected component.  The larger
        # detached patch is an invalid assignment artifact and must be merged,
        # not allowed to replace the plate's generating site.
        labels[2, 2:4] = "rooted"
        labels[5, 9:13] = "rooted"
        core = 2 * grid.shape[1] + 2
        cleaned, merge_count = clean_orphan_components(
            grid,
            labels,
            ("other", "rooted"),
            protected_cells={"rooted": {core}},
        )
        self.assertEqual(merge_count, 1)
        self.assertEqual(cleaned.reshape(-1)[core], "rooted")
        self.assertEqual(len(_components(grid, cleaned, "rooted")), 1)

    def test_corridor_deformation_keeps_every_generated_plate_core_connected(self):
        # This seed formerly produced a one-cell plate-micro-01 core plus a
        # detached 26,560-cell component because a curved corridor crossed the
        # high-latitude microplate's interior.  It is deliberately evaluated
        # on the reference grid, where that topology error first manifested.
        seed = 627677218
        stage_seed = _derive_seed(seed, "plates")
        morphology = sample_world_morphology(_derive_seed(seed, "morphology"))
        grid = build_lat_lon_grid(360, 720)
        definitions = _plate_definitions(12, morphology, stage_seed)
        first_pass = build_plate_fields(
            grid,
            plate_count=12,
            anchors=definitions,
            stage_seed=stage_seed,
            radius_km=_RADIUS_KM,
            boundary_corridors=(),
        )
        corridors = _curved_boundary_corridors(first_pass, morphology, stage_seed)
        fields = build_plate_fields(
            grid,
            plate_count=12,
            anchors=definitions,
            stage_seed=stage_seed,
            radius_km=_RADIUS_KM,
            boundary_corridors=corridors,
        )
        flat = fields.plate_grid.reshape(-1)
        vectors = grid.unit_vectors.reshape((-1, 3))
        for definition in definitions:
            core = int(np.argmax(vectors @ lon_lat_to_unit_vector(
                definition.anchor_lon, definition.anchor_lat,
            )))
            self.assertEqual(flat[core], definition.plate_id)
            self.assertEqual(len(_components(grid, fields.plate_grid, definition.plate_id)), 1)
        self.assertEqual(fields.diagnostics["topology"]["connected_plate_count"], 12)
        self.assertEqual(fields.diagnostics["topology"]["maximum_component_count"], 1)


if __name__ == "__main__":
    unittest.main()
