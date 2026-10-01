"""Final ownership contracts apply after every administrative repair pass."""
from types import SimpleNamespace
import unittest

import numpy as np

from world_atlas.core.society.model import SocietyLayers


def society_arguments():
    shape = (3, 6)
    states = np.tile(np.array([1, 1, 1, 2, 2, 0], dtype=np.int16), (3, 1))
    provinces = states.astype(np.int32)
    settlements = (
        SimpleNamespace(identifier="first-seat", row=1, column=1),
        SimpleNamespace(identifier="second-seat", row=1, column=4),
    )
    return dict(
        population=SimpleNamespace(population_weight=np.ones(shape)),
        settlements=settlements,
        transport=SimpleNamespace(accessibility=np.ones(shape)),
        cultures=SimpleNamespace(civilization_id=np.ones(shape)),
        religions=SimpleNamespace(religion_id=np.ones(shape)),
        geographic_features=(),
        politics=SimpleNamespace(state_id=states, states=tuple(
            SimpleNamespace(identifier=i, core_settlement_id=seat.identifier)
            for i, seat in enumerate(settlements, start=1)
        )),
        provinces=SimpleNamespace(province_id=provinces, provinces=tuple(
            SimpleNamespace(identifier=i, state_identifier=i,
                            core_settlement_id=seat.identifier)
            for i, seat in enumerate(settlements, start=1)
        )),
    )


class AdministrativeIntegrityTests(unittest.TestCase):
    def test_valid_nested_ownership_does_not_change_arrays(self):
        arguments = society_arguments()
        before = arguments["provinces"].province_id.copy()
        SocietyLayers(**arguments)
        np.testing.assert_array_equal(arguments["provinces"].province_id, before)

    def test_every_controlled_cell_has_a_province(self):
        arguments = society_arguments()
        arguments["provinces"].province_id[0, 0] = 0
        with self.assertRaisesRegex(ValueError, "every country cell"):
            SocietyLayers(**arguments)

    def test_province_cannot_cross_parent_country(self):
        arguments = society_arguments()
        arguments["provinces"].province_id[0, 0] = 2
        with self.assertRaisesRegex(ValueError, "parent country"):
            SocietyLayers(**arguments)

    def test_capital_must_be_owned_by_its_country(self):
        arguments = society_arguments()
        arguments["politics"].states[0].core_settlement_id = "second-seat"
        with self.assertRaisesRegex(ValueError, "country core"):
            SocietyLayers(**arguments)

    def test_province_seat_must_be_owned_by_its_province(self):
        arguments = society_arguments()
        arguments["provinces"].provinces[0].core_settlement_id = "second-seat"
        with self.assertRaisesRegex(ValueError, "province core"):
            SocietyLayers(**arguments)

    def test_missing_seat_reference_is_rejected(self):
        arguments = society_arguments()
        arguments["provinces"].provinces[0].core_settlement_id = "missing-seat"
        with self.assertRaisesRegex(ValueError, "province core"):
            SocietyLayers(**arguments)


if __name__ == "__main__":
    unittest.main()
