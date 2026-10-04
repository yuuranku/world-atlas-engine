"""Canonical naming uses lineage and place semantics, never old labels."""

from dataclasses import replace
import re
import unittest

import numpy as np

from world_atlas.core.society.model import (
    Civilization,
    CultureLayers,
    GeographicFeature,
    GovernmentForm,
    Language,
    PoliticalEntity,
    PoliticalLayers,
    PopulationLayers,
    Province,
    ProvinceLayers,
    Religion,
    ReligionLayers,
    Settlement,
    SocietyLayers,
    State,
    TransportLayers,
)
from world_atlas.core.society.world_identity import NameRegistry, assign_world_identity, naming_audit
from world_atlas.core.society.naming_profiles import naming_profile
from world_atlas.core.society.onomastics import settlement_name_candidates


def _society() -> SocietyLayers:
    """A two-lineage world with stable records but deliberately stale labels."""

    shape = (6, 8)
    state_id = np.ones(shape, dtype=np.int16)
    state_id[:, 4:] = 2
    province_id = state_id.astype(np.int32)
    civilization_id = state_id.copy()
    language_id = state_id.copy()
    population = PopulationLayers(
        population_weight=np.ones(shape, dtype=np.float32),
        population_band=np.ones(shape, dtype=np.uint8),
        population_min=100_000,
        population_max=200_000,
    )
    settlements = (
        Settlement(
            identifier="east-port", name="旧东港", row=2, column=1,
            tier="city", site_type="port", score=1.0,
            population_min=20_000, population_max=40_000,
            holy_religion_identifier=1,
        ),
        Settlement(
            identifier="east-market", name="旧东市", row=4, column=2,
            tier="town", site_type="market", score=0.6,
            population_min=4_000, population_max=8_000,
        ),
        Settlement(
            identifier="west-river", name="Old West", row=2, column=5,
            tier="city", site_type="river-city", score=0.9,
            population_min=18_000, population_max=36_000,
        ),
        Settlement(
            identifier="west-pass", name="Old Gate", row=4, column=6,
            tier="town", site_type="pass", score=0.5,
            population_min=3_000, population_max=6_000,
        ),
    )
    civilizations = (
        Civilization(1, "旧东文明", "lineage-s00-01", "east-port", ("north", "south", "coast")),
        Civilization(2, "Old Western Civilization", "lineage-s01-02", "west-river", ("north", "south", "coast")),
    )
    languages = (
        Language(1, "旧东语", "lineage-s00-01", 1, "east-port"),
        Language(2, "Old Western Tongue", "lineage-s01-02", 2, "west-river"),
    )
    cultures = CultureLayers(
        civilization_id=civilization_id,
        civilization_influence=np.ones(shape, dtype=np.uint8),
        language_family_id=language_id,
        language_id=language_id,
        language_contact=np.zeros(shape, dtype=bool),
        civilizations=civilizations,
        languages=languages,
    )
    religions = ReligionLayers(
        religion_id=np.ones(shape, dtype=np.int16),
        religions=(Religion(1, "旧日教", "ancestral-rite", "east-port", 1, "world", "care", "temple"),),
    )
    states = (
        State(1, "旧东", "east-port", "medium", 50_000, 90_000, 1, 1),
        State(2, "Old West", "west-river", "medium", 50_000, 90_000, 2, 2),
    )
    politics = PoliticalLayers(
        state_id=state_id,
        frontier=np.zeros(shape, dtype=bool),
        states=states,
        government_forms=(GovernmentForm(1, "civic", "Civic", "civic government"),),
        political_entities=(
            PoliticalEntity(1, 1, 1, "旧东共和国"),
            PoliticalEntity(2, 2, 1, "Old West王国"),
        ),
        frontier_groups=(),
    )
    provinces = ProvinceLayers(
        province_id=province_id,
        provinces=(
            Province(1, "旧东府", 1, "east-port", "coastal", "central-bureaucracy", "capital", "capital-district", "dense", 24),
            Province(2, "Old West County", 2, "west-pass", "river", "civic-administration", "civic", "civic-district", "settled", 24),
        ),
    )
    return SocietyLayers(
        population=population,
        settlements=settlements,
        transport=TransportLayers(np.zeros(shape, dtype=np.float32), ()),
        cultures=cultures,
        religions=religions,
        geographic_features=(
            GeographicFeature("east-desert", "desert", "旧东沙漠", 3, 3, 1, "secondary"),
            GeographicFeature("west-river", "river", "Old West River", 1, 6, 2, "major"),
        ),
        politics=politics,
        provinces=provinces,
    )


class WorldIdentityTests(unittest.TestCase):
    def test_large_local_inventory_grows_without_serials_or_changing_culture(self):
        lineage = "lineage-s05-17:profile-fantasy-astral-05"
        profile = naming_profile(lineage, style=5)
        first_candidates = settlement_name_candidates(lineage, "plain", None, seed=31)
        # Keep old complete names and an embedded morpheme prohibited. The
        # registry must grow the grammar, rather than prefixing these names.
        forbidden = (*first_candidates[:80], profile.stems[0], "旧城")

        def allocate():
            registry = NameRegistry(9127, forbidden)
            names = [registry.name(f"city:large-{index}", 5, candidates=("旧城",),
                                  lineage=lineage, environment="plain") for index in range(1200)]
            return names

        names = allocate()
        self.assertEqual(names, allocate())
        self.assertEqual(len(set(names)), 1200)
        grammar = "(?:" + "|".join(map(re.escape, profile.stems)) + "){2,}(?:萨赫勒)?"
        self.assertTrue(all(re.fullmatch(grammar, name) for name in names))
        self.assertTrue(all(not any(character.isdigit() for character in name) for name in names))
        self.assertFalse(any(old in name for name in names for old in forbidden))
        self.assertFalse(any(a == b for name in names for a, b in zip(name, name[1:])))

    def test_allocated_compounds_can_still_form_distinct_longer_names(self):
        lineage = "lineage-s00-17:profile-fantasy-sylvan-00"
        profile = naming_profile(lineage, style=0)
        registry = NameRegistry(73)
        registry.used.update(first + second + ending for first in profile.stems
                             for second in profile.stems for ending in (*profile.endings, ""))
        name = registry.name("city:dense", 0, candidates=(), lineage=lineage)
        self.assertGreaterEqual(len(name), 4)
        self.assertTrue(registry.available(name + "新"))

    def test_unusable_morphemes_or_fixed_suffix_fail_without_retrying(self):
        lineage = "lineage-s05-17:profile-realistic-05"
        profile = naming_profile(lineage, style=5)
        with self.assertRaisesRegex(ValueError, "no unexcluded morpheme"):
            NameRegistry(73, profile.stems).name("city:blocked", 5, candidates=(), lineage=lineage)
        with self.assertRaisesRegex(ValueError, "excluded suffix"):
            NameRegistry(73, ("旧名",)).name("province:blocked", 5, suffix="旧名州", lineage=lineage)
        # The fixed suffix is safe in isolation, but every possible word
        # boundary is blocked. An endlessly extensible prefix cannot help.
        blocked_boundaries = tuple(morpheme[-1] + "城" for morpheme in (*profile.stems, *profile.endings))
        with self.assertRaisesRegex(ValueError, "no unexcluded morpheme"):
            NameRegistry(73, blocked_boundaries).name("province:blocked-boundary", 5,
                candidates=(), suffix="城", lineage=lineage)

    def test_indexed_exclusions_keep_exact_and_substring_semantics(self):
        forbidden = ("安", "阿尔", "旧地名", "", " ")
        registry = NameRegistry(73, forbidden)
        registry.used.add("新城")
        normalized = set(name.strip() for name in forbidden if name.strip())
        for name in ("安", "安川", "阿尔", "新阿尔城", "旧地名州", "新城", "新泉"):
            expected = (name not in registry.used and name not in normalized
                        and not any(old in name for old in normalized if len(old) >= 2))
            self.assertEqual(registry.available(name), expected)

    def test_names_are_lineage_derived_and_categorically_grounded(self):
        renamed = assign_world_identity(_society(), seed=934_221, forbidden=("旧东", "Old West"))
        cities = {item.identifier: item.name for item in renamed.settlements}
        civilizations = {item.identifier: item.name for item in renamed.cultures.civilizations}
        languages = {item.identifier: item.name for item in renamed.cultures.languages}
        provinces = {item.identifier: item.name for item in renamed.provinces.provinces}
        features = {item.identifier: item.name for item in renamed.geographic_features}

        # A civilization and its language are family-related, but no longer a
        # selected city with a generic suffix bolted on.
        self.assertNotEqual(civilizations[1], f"{cities['east-port']}文明圈")
        self.assertNotEqual(languages[2], f"{cities['west-river']}语")
        # Administrative districts visibly derive from their own settlement
        # seat, including non-capital provinces.
        self.assertTrue(provinces[1].startswith(cities["east-port"][:2]))
        self.assertTrue(provinces[2].startswith(cities["west-pass"][:3]))
        # The Sinitic inventory is intentionally common and short; the
        # desert also preserves a local regional formative before its legend
        # classifier rather than receiving a generic final root.
        east_profile = naming_profile(renamed.cultures.civilizations[0].name_family, style=0)
        common_east = set("".join((*east_profile.stems, *east_profile.endings, "海浦潮汀")))
        self.assertTrue(set(cities["east-port"]) <= common_east)
        self.assertTrue(features["east-desert"].endswith("沙漠"))
        self.assertIn(features["east-desert"][:-2], east_profile.landmarks)
        self.assertEqual(naming_audit(renamed, ())['schema'], "world-names-v3")

    def test_renaming_is_idempotent_and_never_uses_preexisting_labels(self):
        first = assign_world_identity(_society(), seed=934_221, forbidden=("旧东", "Old West"))
        second = assign_world_identity(first, seed=934_221, forbidden=("旧东", "Old West"))
        self.assertEqual(naming_audit(first, ()), naming_audit(second, ()))
        self.assertEqual(
            [item.name for item in first.settlements],
            [item.name for item in second.settlements],
        )
        # Stable identifiers, administrative topology, and immutable rasters
        # are left intact by the naming-only pass.
        self.assertEqual(first.politics.states, second.politics.states)
        np.testing.assert_array_equal(first.politics.state_id, second.politics.state_id)
        np.testing.assert_array_equal(first.provinces.province_id, second.provinces.province_id)

    def test_stale_record_names_do_not_affect_the_new_identity(self):
        source = _society()
        altered_states = tuple(replace(item, name=f"legacy-{item.identifier}") for item in source.politics.states)
        altered_entities = tuple(
            replace(item, formal_name=f"legacy-{item.country_identifier}" + item.formal_name[len(source.politics.states[item.country_identifier - 1].name):])
            for item in source.politics.political_entities
        )
        altered = replace(
            source,
            settlements=tuple(replace(item, name=f"legacy-{item.identifier}") for item in source.settlements),
            cultures=replace(
                source.cultures,
                civilizations=tuple(replace(item, name=f"legacy-{item.identifier}") for item in source.cultures.civilizations),
                languages=tuple(replace(item, name=f"legacy-{item.identifier}") for item in source.cultures.languages),
            ),
            politics=replace(source.politics, states=altered_states, political_entities=altered_entities),
            provinces=replace(source.provinces, provinces=tuple(replace(item, name=f"legacy-{item.identifier}") for item in source.provinces.provinces)),
            geographic_features=tuple(replace(item, name=f"legacy-{item.identifier}") for item in source.geographic_features),
        )
        baseline = assign_world_identity(source, seed=934_221)
        reworded = assign_world_identity(altered, seed=934_221)
        self.assertEqual(naming_audit(baseline, ()), naming_audit(reworded, ()))


if __name__ == "__main__":
    unittest.main()
