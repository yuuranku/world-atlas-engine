"""Tests for final-raster geography acceptance evidence."""

from __future__ import annotations

import unittest

import numpy as np

from world_atlas.core.geography_audit import audit_final_geography


class GeographyAuditTests(unittest.TestCase):
    def test_periodic_four_and_eight_connectivity_are_reported_separately(self):
        land = np.zeros((5, 8), dtype=bool)
        # Owner 1 crosses the -180/180 seam; owner 2 is diagonal-only.
        land[2, 0] = True
        land[2, -1] = True
        land[0, 3] = True
        land[1, 4] = True
        owners = np.zeros_like(land, dtype=np.int16)
        owners[2, 0] = owners[2, -1] = 1
        owners[0, 3] = owners[1, 4] = 2

        review = audit_final_geography(
            land,
            owners,
            np.zeros(land.shape[0]),
            recipe_continent_count=2,
            major_land_share=0.30,
        )

        four = review["land"]["fourConnected"]
        eight = review["land"]["eightConnected"]
        self.assertEqual(four["componentCount"], 3)
        self.assertEqual(eight["componentCount"], 2)
        self.assertEqual(len(four["components"]), 3)
        self.assertEqual(len(eight["components"]), 2)
        # Four-neighbour topology has only the seam component above 30%; the
        # diagonal pair becomes a second major landmass under eight-neighbours.
        acceptance = review["continentAcceptance"]
        self.assertEqual(acceptance["fourConnectedMajorLandmassCount"], 1)
        self.assertEqual(acceptance["eightConnectedMajorLandmassCount"], 2)
        self.assertFalse(acceptance["fourConnectedMatchesRecipe"])
        self.assertTrue(acceptance["eightConnectedMatchesRecipe"])
        self.assertEqual(acceptance["status"], "not_met")

        fragments = {item["ownerId"]: item for item in review["ownerFragments"]}
        self.assertEqual(fragments[1]["fourConnectedFragmentCount"], 1)
        self.assertEqual(fragments[2]["fourConnectedFragmentCount"], 2)
        self.assertEqual(fragments[2]["eightConnectedFragmentCount"], 1)

    def test_requested_continent_count_is_not_satisfied_by_owner_labels_alone(self):
        land = np.zeros((6, 12), dtype=bool)
        land[1:5, 1:5] = True
        land[1:5, 7:11] = True
        owners = np.zeros_like(land, dtype=np.int16)
        # Three owner IDs are deliberately spread across only two real blocks.
        owners[land] = 1
        owners[1:5, 7:9] = 2
        owners[1:5, 9:11] = 3

        review = audit_final_geography(
            land,
            owners,
            np.zeros(land.shape[0]),
            recipe_continent_count=3,
        )

        acceptance = review["continentAcceptance"]
        self.assertEqual(acceptance["fourConnectedMajorLandmassCount"], 2)
        self.assertEqual(acceptance["eightConnectedMajorLandmassCount"], 2)
        self.assertEqual(acceptance["status"], "not_met")
        self.assertEqual(len(review["ownerFragments"]), 3)
        self.assertEqual(
            sum(item["weightedArea"] for item in review["ownerFragments"]),
            review["land"]["fourConnected"]["weightedArea"],
        )


if __name__ == "__main__":
    unittest.main()
