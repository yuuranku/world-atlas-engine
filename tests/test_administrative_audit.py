"""Regression coverage for read-only administrative topology review."""

from types import SimpleNamespace
import unittest

import numpy as np

from world_atlas.core.society.administrative_audit import (
    audit_administrative_topology,
)


def _record(identifier: int, core_settlement_id: str, state_identifier: int | None = None):
    values = {
        "identifier": identifier,
        "core_settlement_id": core_settlement_id,
    }
    if state_identifier is not None:
        values["state_identifier"] = state_identifier
    return SimpleNamespace(**values)


def _society(
    state_id: np.ndarray,
    province_id: np.ndarray,
    settlements: tuple[SimpleNamespace, ...],
    *,
    state_records: tuple[SimpleNamespace, ...],
    province_records: tuple[SimpleNamespace, ...],
) -> SimpleNamespace:
    return SimpleNamespace(
        settlements=settlements,
        transport=SimpleNamespace(routes=()),
        politics=SimpleNamespace(state_id=state_id, states=state_records),
        provinces=SimpleNamespace(
            province_id=province_id,
            provinces=province_records,
        ),
    )


class AdministrativeTopologyAuditTests(unittest.TestCase):
    def test_same_land_exclave_requires_review_and_wrap_is_one_core_component(self):
        water = np.zeros((5, 7), dtype=np.uint8)
        state_id = np.zeros((5, 7), dtype=np.int16)
        province_id = np.zeros((5, 7), dtype=np.int32)
        # The first two cells join across the longitude seam.  The third one
        # belongs to the same country but is isolated on the same landmass.
        state_id[2, (0, 6, 3)] = 1
        province_id[2, (0, 6, 3)] = 1
        state_id[1, 1] = 2
        province_id[1, 1] = 2
        society = _society(
            state_id,
            province_id,
            (
                SimpleNamespace(identifier="state-one", row=2, column=0),
                SimpleNamespace(identifier="state-two", row=1, column=1),
            ),
            state_records=(
                _record(1, "state-one"),
                _record(2, "state-two"),
            ),
            province_records=(
                _record(1, "state-one", 1),
                _record(2, "state-two", 2),
            ),
        )

        audit = audit_administrative_topology(
            SimpleNamespace(water=water, shape=water.shape),
            society,
        )

        state_one = audit["states"]["owners"][0]
        province_one = audit["provinces"]["owners"][0]
        self.assertEqual(state_one["coreComponentAreaCells"], 2)
        self.assertEqual(state_one["sameLandNonCoreComponentCount"], 1)
        self.assertEqual(state_one["sameLandNonCoreAreaCells"], 1)
        self.assertEqual(state_one["status"], "requires_review")
        self.assertTrue(state_one["seatIsLargestComponent"])
        self.assertEqual(province_one["sameLandNonCoreComponentCount"], 1)
        self.assertEqual(
            audit["states"]["summary"]["requiresReviewOwnerCount"],
            1,
        )

    def test_water_separated_component_is_reported_without_false_exclave_failure(self):
        water = np.ones((5, 7), dtype=np.uint8)
        water[1, 1] = 0
        water[3, 5] = 0
        state_id = np.zeros((5, 7), dtype=np.int16)
        province_id = np.zeros((5, 7), dtype=np.int32)
        state_id[1, 1] = state_id[3, 5] = 1
        province_id[1, 1] = province_id[3, 5] = 1
        society = _society(
            state_id,
            province_id,
            (SimpleNamespace(identifier="island-seat", row=1, column=1),),
            state_records=(_record(1, "island-seat"),),
            province_records=(_record(1, "island-seat", 1),),
        )

        audit = audit_administrative_topology(
            SimpleNamespace(water=water, shape=water.shape),
            society,
        )

        state_one = audit["states"]["owners"][0]
        self.assertEqual(state_one["sameLandNonCoreComponentCount"], 0)
        self.assertEqual(state_one["waterSeparatedNonCoreComponentCount"], 1)
        self.assertEqual(state_one["waterSeparatedNonCoreAreaCells"], 1)
        self.assertEqual(state_one["status"], "ok")

    def test_seat_outside_largest_component_is_reported(self):
        water = np.zeros((4, 6), dtype=np.uint8)
        state_id = np.zeros((4, 6), dtype=np.int16)
        province_id = np.zeros((4, 6), dtype=np.int32)
        state_id[0, 0] = 1
        province_id[0, 0] = 1
        state_id[2:4, 3:6] = 1
        province_id[2:4, 3:6] = 1
        society = _society(
            state_id,
            province_id,
            (SimpleNamespace(identifier="small-seat", row=0, column=0),),
            state_records=(_record(1, "small-seat"),),
            province_records=(_record(1, "small-seat", 1),),
        )

        audit = audit_administrative_topology(
            SimpleNamespace(water=water, shape=water.shape),
            society,
        )

        state_one = audit["states"]["owners"][0]
        self.assertEqual(state_one["coreComponentAreaCells"], 1)
        self.assertEqual(state_one["largestComponentAreaCells"], 6)
        self.assertFalse(state_one["seatIsLargestComponent"])


if __name__ == "__main__":
    unittest.main()
