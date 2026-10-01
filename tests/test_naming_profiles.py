"""Naming traditions stay coherent across a world's linked place records."""

from collections import Counter
import unittest

import numpy as np

from world_atlas.core.society.naming_profiles import (
    PROFILE_SETS,
    naming_profile,
    profile_lineage,
)
from world_atlas.core.society.onomastics import (
    geographic_name_candidates,
    lineage_branch,
    lineage_roots,
    lineage_style_index,
    settlement_name_candidates,
)
from world_atlas.core.society.world_identity import assign_world_identity, naming_audit

from test_world_identity import _society


class NamingProfileTests(unittest.TestCase):
    def test_daughter_languages_and_local_branches_inherit_one_complete_profile(self):
        for profile_set in PROFILE_SETS:
            for style in range(12):
                base = f"lineage-s{style:02d}-17"
                selected = profile_lineage(base, seed=617, profile_set=profile_set)
                profile = naming_profile(selected, style=style)
                self.assertEqual(profile.style, style)
                self.assertEqual(lineage_style_index(selected), style)
                for branch in ("language-2", "state-4", "city-53", "feature-river-7"):
                    daughter = lineage_branch(selected, branch)
                    self.assertEqual(naming_profile(daughter, style=style), profile)
                    self.assertEqual(
                        profile_lineage(daughter, seed=617, profile_set=profile_set),
                        selected,
                    )

    def test_explicit_sets_change_inventory_without_changing_sound_family(self):
        for style in range(12):
            profiles = [
                naming_profile(
                    profile_lineage(f"lineage-s{style:02d}-17", seed=617, profile_set=kind),
                    style=style,
                )
                for kind in PROFILE_SETS
            ]
            self.assertEqual(len({profile.key for profile in profiles}), 3)
            self.assertEqual(len({(profile.stems, profile.endings) for profile in profiles}), 3)

    def test_mixed_selection_is_seeded_and_uses_all_available_sets(self):
        reached = set()
        for seed in range(24):
            for style in range(12):
                lineage = f"lineage-s{style:02d}-17"
                selected = profile_lineage(lineage, seed=seed)
                self.assertEqual(selected, profile_lineage(lineage, seed=seed))
                profile = naming_profile(selected, style=style)
                reached.add(profile.key.split("-", 1)[0])
        self.assertEqual(reached, set(PROFILE_SETS))

    def test_unrelated_fantasy_sound_families_do_not_share_one_disguised_pool(self):
        themes = {}
        for profile in PROFILE_SETS["fantasy"]:
            theme = profile.key.rsplit("-", 1)[0]
            themes.setdefault(theme, []).append(profile)
        for profiles in themes.values():
            western = [profile for profile in profiles if profile.style != 0]
            self.assertEqual(
                len({(profile.stems, profile.endings) for profile in western}),
                len(western),
            )

    def test_unknown_sets_and_incompatible_profile_markers_are_rejected(self):
        with self.assertRaises(ValueError):
            profile_lineage("lineage-s00-01", seed=1, profile_set="unknown")
        with self.assertRaises(ValueError):
            naming_profile("lineage-s00-01:profile-unknown", style=0)
        with self.assertRaises(ValueError):
            naming_profile("lineage-s00-01:profile-historical-01", style=0)

    def test_compounds_do_not_repeat_the_same_topographic_morpheme(self):
        cases = (
            ("lineage-s09-01:profile-historical-09", "coast", "island", "凯尔凯尔"),
            ("lineage-s01-01:profile-realistic-01", "coast", "island", "霍尔霍尔姆"),
            ("lineage-s09-01:profile-fantasy-elder-09", "plain", "plain", "摩尔莫尔"),
        )
        for lineage, environment, feature, repetition in cases:
            with self.subTest(repetition=repetition):
                self.assertNotIn(repetition, settlement_name_candidates(lineage, environment, None, seed=617))
                self.assertNotIn(repetition, geographic_name_candidates(lineage, feature, seed=617))
                self.assertNotIn(repetition, lineage_roots(lineage, count=384))

    def test_historical_eastern_countries_use_compact_polity_stems(self):
        renamed = assign_world_identity(_society(), seed=617, profile_set="historical")
        culture = renamed.cultures.civilizations[0]
        profile = naming_profile(culture.name_family, style=0)
        country = renamed.politics.states[0]
        self.assertIn(country.name, profile.polities)
        self.assertLessEqual(len(country.name), 2)
        self.assertNotEqual(country.name, renamed.settlements[0].name)

    def test_full_identity_replay_preserves_linked_records_and_profile_selection(self):
        source = _society()
        for profile_set in PROFILE_SETS:
            first = assign_world_identity(source, seed=617, profile_set=profile_set)
            replay = assign_world_identity(first, seed=617, profile_set=profile_set)
            self.assertEqual(naming_audit(first, ()), naming_audit(replay, ()))
            for before, after in (
                (source.politics.state_id, first.politics.state_id),
                (source.provinces.province_id, first.provinces.province_id),
                (source.population.population_weight, first.population.population_weight),
            ):
                np.testing.assert_array_equal(before, after)
            self.assertEqual(
                [(item.identifier, item.row, item.column) for item in source.settlements],
                [(item.identifier, item.row, item.column) for item in first.settlements],
            )
            civilizations = {item.identifier: item for item in first.cultures.civilizations}
            for language in first.cultures.languages:
                parent = civilizations[language.family_identifier]
                self.assertEqual(
                    naming_profile(language.name_family, style=lineage_style_index(language.name_family)),
                    naming_profile(parent.name_family, style=lineage_style_index(parent.name_family)),
                )
            cities = {item.identifier: item.name for item in first.settlements}
            self.assertTrue(all(
                item.name.startswith(cities[item.core_settlement_id])
                for item in first.provinces.provinces
            ))
            names = (
                [item.name for item in first.settlements]
                + [item.name for item in first.politics.states]
                + [item.name for item in first.provinces.provinces]
                + [item.name for item in first.geographic_features]
                + [item.name for item in first.cultures.civilizations]
                + [item.name for item in first.cultures.languages]
                + [item.name for item in first.religions.religions]
            )
            self.assertFalse([name for name, count in Counter(names).items() if count > 1])


if __name__ == "__main__":
    unittest.main()
