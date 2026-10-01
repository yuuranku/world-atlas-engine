"""Saved enrichment changes only named-record metadata and keeps its proof current."""
from dataclasses import asdict,replace
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from world_atlas.core.presentation import society_content_digest
from world_atlas.core.society.model import GeographicFeature
from test_world_identity import _society


spec=importlib.util.spec_from_file_location("geography_enrich_script",Path(__file__).resolve().parents[1]/"scripts/enrich_saved_geography.py")
script=importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


def fixture(world):
    for name in ("grid/world-grid.npz","grid/world-grid.json","source/physical-fields.npz",
                 "thematic/human-capacity.npz","society/society.npz","world-settings.json","naming-exclusions.json"):
        path=world/name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(("immutable "+name).encode())
    society=_society()
    extra=GeographicFeature("ridge-01","ridge","真实山脊",2,3,1,"detail")
    enriched=replace(society,geographic_features=(*society.geographic_features,extra))
    metadata={"geographicFeatures":[asdict(item) for item in society.geographic_features],
              "settlements":[asdict(item) for item in society.settlements],"unrelated":{"exact":"kept"}}
    (world/"society/society.json").write_text(json.dumps(metadata),encoding="utf8")
    record={"fieldSha256":script.sha256(world/"source/physical-fields.npz"),"nameDigest":"stale"}
    (world/"regeneration.json").write_text(json.dumps(record),encoding="utf8")
    (world/"repair-check.json").write_text(json.dumps({"repairedSocietyDigest":society_content_digest(society),"administration":{"unchanged":True}}),encoding="utf8")
    grid=SimpleNamespace(content_digest=lambda:"native",metadata={"societyGeneration":{"namingSeed":123,"forbiddenNames":[]}})
    evidence={"featuresBefore":len(society.geographic_features),"featuresAfter":len(enriched.geographic_features),"originalRecordsPreserved":True}
    return grid,society,enriched,metadata,evidence


class GeographicEnrichmentScriptTests(unittest.TestCase):
    def test_metadata_only_update_preserves_every_binary_and_refreshes_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            world=Path(temporary)
            grid,society,enriched,metadata,evidence=fixture(world)
            before={path.relative_to(world).as_posix():path.read_bytes() for path in world.rglob("*") if path.is_file()}
            physical=SimpleNamespace(relative_elevation_m=np.ones((6,8)))
            ecological_sources=object()
            with patch.object(script.WorldGrid,"load",return_value=grid),patch.object(script,"load_society",side_effect=(society,enriched)), \
                    patch.object(script,"load_surface_bundle",return_value=physical), \
                    patch.object(script,"derive_ecological_sources",return_value=ecological_sources) as sources, \
                    patch.object(script,"derive_thematic_layers",return_value=object()) as themes,patch.object(script,"procedural_name_lexicon",return_value=object()), \
                    patch.object(script,"enrich_saved_geographic_features",return_value=(enriched,evidence)):
                report=script.enrich_world(world)
            sources.assert_called_once_with(grid,physical)
            themes.assert_called_once_with(grid,ecological_sources=ecological_sources)
            for name,content in before.items():
                if name not in ("society/society.json","repair-check.json","regeneration.json"):
                    self.assertEqual((world/name).read_bytes(),content)
            after=json.loads((world/"society/society.json").read_text(encoding="utf8"))
            self.assertEqual(after["geographicFeatures"][:len(metadata["geographicFeatures"])],metadata["geographicFeatures"])
            self.assertEqual(after["settlements"],metadata["settlements"])
            self.assertEqual(after["unrelated"],metadata["unrelated"])
            repair=json.loads((world/"repair-check.json").read_text(encoding="utf8"))
            self.assertEqual(repair["administration"],{"unchanged":True})
            self.assertEqual(repair["repairedSocietyDigest"],society_content_digest(enriched))
            self.assertEqual(repair["geographyEnrichment"]["nameDigest"],json.loads((world/"regeneration.json").read_text(encoding="utf8"))["nameDigest"])
            self.assertEqual(report["featuresBefore"],2)

    def test_stale_input_proof_is_rejected_before_any_write(self):
        for field in ("raw","society"):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as temporary:
                world=Path(temporary)
                grid,society,_,_,_=fixture(world)
                target=world/("regeneration.json" if field=="raw" else "repair-check.json")
                document=json.loads(target.read_text(encoding="utf8"))
                document["fieldSha256" if field=="raw" else "repairedSocietyDigest"]="incorrect"
                target.write_text(json.dumps(document),encoding="utf8")
                before={path:path.read_bytes() for path in world.rglob("*") if path.is_file()}
                with patch.object(script.WorldGrid,"load",return_value=grid),patch.object(script,"load_society",return_value=society):
                    with self.assertRaises(ValueError):
                        script.enrich_world(world)
                self.assertTrue(all(path.read_bytes()==content for path,content in before.items()))

    def test_replay_keeps_original_enrichment_baseline_and_all_file_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            world=Path(temporary)
            grid,society,_,_,_=fixture(world)
            path=world/"repair-check.json"
            original=json.loads(path.read_text(encoding="utf8"))
            original["geographyEnrichment"]={"featuresBefore":335,"featuresAfter":423,"status":"ok"}
            path.write_text(json.dumps(original),encoding="utf8")
            before={path:path.read_bytes() for path in world.rglob("*") if path.is_file()}
            physical=SimpleNamespace(relative_elevation_m=np.ones((6,8)))
            ecological_sources=object()
            with patch.object(script.WorldGrid,"load",return_value=grid),patch.object(script,"load_society",return_value=society), \
                    patch.object(script,"load_surface_bundle",return_value=physical), \
                    patch.object(script,"derive_ecological_sources",return_value=ecological_sources) as sources, \
                    patch.object(script,"derive_thematic_layers",return_value=object()) as themes,patch.object(script,"procedural_name_lexicon",return_value=object()), \
                    patch.object(script,"enrich_saved_geographic_features",return_value=(society,{"featuresBefore":423,"featuresAfter":423})):
                report=script.enrich_world(world)
            sources.assert_called_once_with(grid,physical)
            themes.assert_called_once_with(grid,ecological_sources=ecological_sources)
            self.assertEqual(report["featuresBefore"],335)
            self.assertTrue(all(path.read_bytes()==content for path,content in before.items()))


if __name__=="__main__":
    unittest.main()
