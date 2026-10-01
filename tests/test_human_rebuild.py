"""Human-only regeneration must never rebuild or modify saved physical truth."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts import rebuild_human_world as rebuild
from world_atlas.settings import WorldSettings


def fixture(root):
    root=root.resolve()
    world, output = root / "original", root / "new-human-world"
    (world / "grid").mkdir(parents=True)
    (world / "source").mkdir()
    land = np.array([[True, True, False, False], [True, False, False, False]])
    np.savez_compressed(world / "grid/world-grid.npz", water=np.where(land, 0, 1).astype(np.uint8))
    (world / "grid/world-grid.json").write_text('{"original":true}', encoding="utf-8")
    old_bundle = world / "source/physical-fields.npz"
    old_bundle.write_bytes(b"original-bundle")
    reference = world / "source/physical-reference.png"
    reference.write_bytes(b"original-reference")
    verified = root / "verified-v3.npz"
    verified.write_bytes(b"verified-raw-ground")
    proof_path = root / "verified-v3.json"
    recipe = {"seed":17,"width":4,"height":2}
    provenance = {"schema":"physical-source-v3", "recipe":recipe}
    settings = WorldSettings(17, 18,
        planet={"solarConstantWm2":1620.0,"radiusKm":6400.0,"gravityMps2":9.80665,
                "rotationPeriodHours":26.0,"rotationDirection":"retrograde",
                "orbitalPeriodDays":420.0,"axialTiltDegrees":28.0},
        society={"settlementCount":720,"civilizationCount":14,"minimumCivilizationCount":14,
                 "languageCount":24,"religionCount":10,"stateCount":80,"populationMin":100_000_000,
                 "populationMax":160_000_000,"frontierTargetShare":.25},
        technology_era="preindustrial",travel_capabilities=())
    request = {**settings.society,"humanSeed":17,"namingSeed":18,"technologyEra":"preindustrial",
               "namingProfile":"procedural","forbiddenNames":["excluded-name"]}
    metadata = {"societyGeneration":request,"planet":settings.planet,
                "proceduralPhysicalSource":{"fieldBundle":{"path":"old-origin.npz","sha256":rebuild.sha256(old_bundle)}}}
    grid = SimpleNamespace(water=np.where(land, 0, 1), shape=land.shape, metadata=metadata,
                           content_digest=lambda:"preserved-grid-digest")
    source = SimpleNamespace(relative_elevation_m=np.where(land, 10.0, -5.0),land_mask=land,
                             diagnostics={"seed":17})
    proof = {"schema":"verified-physical-surface-replay-v1","verifiedExact":True,
             "originalSourceUnchanged":True,"nativePaletteReconstructionExact":True,
             "sourceBundleSha256":rebuild.sha256(old_bundle),"verifiedBundleSha256":rebuild.sha256(verified),
             "recipe":recipe,"comparisons":{name:{"exact":True,"different":0} for name in rebuild._REPLAY_FIELDS}}
    documents = {world / "world-settings.json":settings.document(),world / "naming-exclusions.json":{"forbidden":["excluded-name"]},
                 world / "source/provenance.json":provenance,proof_path:proof,
                 world / "worldgen.json":{"source":{"path":"source/physical-reference.png","sha256":rebuild.sha256(reference),
                     "fieldBundle":{"path":"source/physical-fields.npz","sha256":rebuild.sha256(old_bundle)}},"output":{"directory":"."}}}
    for path, document in documents.items():
        path.write_text(json.dumps(document), encoding="utf-8")
    return SimpleNamespace(world=world,output=output,verified=verified,proof_path=proof_path,
                           grid=grid,source=source,proof=proof,old_society=object())


def mocked_inputs(stack, data):
    stack.enter_context(patch.object(rebuild.WorldGrid, "load", return_value=data.grid))
    stack.enter_context(patch.object(rebuild, "load_surface_bundle", return_value=data.source))
    stack.enter_context(patch.object(rebuild, "load_society", return_value=data.old_society))
    data.ecological_sources=object()
    stack.enter_context(patch.object(rebuild,"derive_ecological_sources",return_value=data.ecological_sources))
    return stack.enter_context(patch.object(rebuild, "_society_generation_request", return_value=("preserved-name-pools",{"state_count":80})))


class HumanRebuildTests(unittest.TestCase):
    def test_rejects_wrong_raw_ground_before_creating_any_output(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            data=fixture(Path(directory));mocked_inputs(stack,data)
            data.source.relative_elevation_m[0,0] = -1
            with self.assertRaisesRegex(ValueError,"raw ground"):
                rebuild.rebuild_human_world(data.world,data.verified,data.proof_path,data.output)
            self.assertFalse(data.output.exists())

    def test_rejects_nonexact_proof_and_existing_output_without_overwriting(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            data=fixture(Path(directory));mocked_inputs(stack,data)
            data.proof["comparisons"]["elevation"]["different"] = 1
            data.proof_path.write_text(json.dumps(data.proof),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"replay proof"):
                rebuild.rebuild_human_world(data.world,data.verified,data.proof_path,data.output)
            self.assertFalse(data.output.exists())
            data.output.mkdir();sentinel=data.output/"sentinel.txt";sentinel.write_text("keep",encoding="utf-8")
            with self.assertRaises(FileExistsError):
                rebuild.rebuild_human_world(data.world,data.verified,data.proof_path,data.output)
            self.assertEqual(sentinel.read_text(encoding="utf-8"),"keep")

    def test_new_human_layers_use_new_themes_while_grid_settings_and_inputs_stay_exact(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            data=fixture(Path(directory));request=mocked_inputs(stack,data)
            old_files={str(path.relative_to(data.world)):path.read_bytes() for path in data.world.rglob('*') if path.is_file()}
            thematic=SimpleNamespace(land_potential=np.zeros(data.grid.shape,dtype=np.float32),land_potential_band=np.zeros(data.grid.shape,dtype=np.uint8),
                                     habitability=np.full(data.grid.shape,.4,dtype=np.float32),habitability_band=np.ones(data.grid.shape,dtype=np.uint8),climate=object(),
                                     physiography=SimpleNamespace(wetland_support=np.zeros(data.grid.shape,dtype=np.float32),wetland=np.zeros(data.grid.shape,dtype=bool)))
            society=SimpleNamespace(population=object())
            derive=stack.enter_context(patch.object(rebuild,"derive_thematic_layers",return_value=thematic))
            human=stack.enter_context(patch.object(rebuild,"derive_society_layers",return_value=society))
            save=stack.enter_context(patch.object(rebuild,"save_society",side_effect=lambda value,path,**kwargs:path.mkdir()))
            stack.enter_context(patch.object(rebuild,"population_density",return_value=np.zeros(data.grid.shape,dtype=np.float32)))
            vegetation_fraction=np.where(data.grid.water==0,.72,0).astype(np.float32)
            vegetation=stack.enter_context(patch.object(rebuild,"derive_vegetation_cover",return_value=SimpleNamespace(fraction=vegetation_fraction)))
            stack.enter_context(patch.object(rebuild,"human_checks",return_value=({"status":"ok","failures":[],"counts":{"countries":80,"provinces":420}},
                {"nameDigest":"new-names"},{"status":"ok"})))
            checks=rebuild.rebuild_human_world(data.world,data.verified,data.proof_path,data.output)
            derive.assert_called_once_with(data.grid,ecological_sources=data.ecological_sources)
            human.assert_called_once_with(data.grid,thematic,"preserved-name-pools",
                                          raw_elevation_m=data.source.relative_elevation_m,state_count=80)
            save.assert_called_once_with(society,data.output/"society",grid_digest="preserved-grid-digest")
            vegetation.assert_called_once_with(data.grid,thematic.climate,ecological_sources=data.ecological_sources)
            self.assertEqual(request.call_count,1)
            self.assertTrue(checks["physicalGridSemanticUnchanged"])
            self.assertEqual((data.output/"source/physical-fields.npz").read_bytes(),data.verified.read_bytes())
            self.assertEqual(old_files,{str(path.relative_to(data.world)):path.read_bytes() for path in data.world.rglob('*') if path.is_file()})
            config=json.loads((data.output/"worldgen.json").read_text(encoding="utf-8"))
            self.assertEqual(config["source"]["fieldBundle"]["sha256"],rebuild.sha256(data.verified))
            self.assertEqual(config["source"]["fieldBundle"]["path"],"source/physical-fields.npz")
            self.assertFalse((data.output/"review").exists())
            record=json.loads((data.output/"regeneration.json").read_text(encoding="utf-8"))
            self.assertEqual(record["status"],"building")
            self.assertEqual(record["humanStatus"],"ok")
            with np.load(data.output/"thematic/human-capacity.npz",allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["habitability"],thematic.habitability)
                np.testing.assert_array_equal(archive["vegetation_cover"],vegetation_fraction)
            with np.load(data.output/"thematic/wetland-support.npz",allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["support"],thematic.physiography.wetland_support)
                np.testing.assert_array_equal(archive["mask"],thematic.physiography.wetland)

    def test_failed_generated_checks_cannot_be_published_as_a_building_world(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            data=fixture(Path(directory));mocked_inputs(stack,data)
            stack.enter_context(patch.object(rebuild,"derive_thematic_layers",side_effect=RuntimeError("failed human source")))
            with self.assertRaisesRegex(RuntimeError,"failed human source"):
                rebuild.rebuild_human_world(data.world,data.verified,data.proof_path,data.output)
            record=json.loads((data.output/"regeneration.json").read_text(encoding="utf-8"))
            self.assertEqual(record["status"],"failed")
            self.assertEqual(record["humanStatus"],"failed")

    def test_administrative_count_gate_checks_generated_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            data=fixture(Path(directory));land=data.grid.water==0
            thematic=SimpleNamespace(land_potential=np.where(land,.5,0),habitability=np.where(land,.6,0),
                land_potential_band=np.where(land,3,0),habitability_band=np.where(land,3,0))
            settings=rebuild.load_world_settings(data.world/"world-settings.json")
            for countries,provinces in [(80,420),(79,420),(80,399),(80,501)]:
                with self.subTest(countries=countries,provinces=provinces), ExitStack() as stack:
                    state_id=np.where(land,1,-1);state_id[0,1]=40;state_id[1,0]=countries
                    province_id=np.where(land,1,-1);province_id[0,1]=100;province_id[1,0]=provinces
                    records=tuple(SimpleNamespace(identifier=index,state_identifier=40 if index==100 else countries if index==provinces else 1)
                                  for index in range(1,provinces+1))
                    society=SimpleNamespace(population=SimpleNamespace(population_weight=land.astype(np.float32)/land.sum(),
                        population_band=np.where(land,3,0).astype(np.uint8),population_min=100_000_000,population_max=160_000_000),
                        settlements=(),transport=SimpleNamespace(routes=(),bridges=()),
                        politics=SimpleNamespace(state_id=state_id,states=tuple(range(countries))),
                        provinces=SimpleNamespace(province_id=province_id,provinces=records),
                        cultures=SimpleNamespace(civilizations=tuple(range(14)),languages=tuple(range(24))),
                        religions=SimpleNamespace(religions=tuple(range(10))))
                    prepared={"settings":settings,"exclusions":{"forbidden":[]},"original":society}
                    stack.enter_context(patch.object(rebuild,"naming_audit",return_value={"oldNameMatches":[],"duplicateNames":[],"profiles":[]}))
                    stack.enter_context(patch.object(rebuild,"assign_world_identity",return_value=society))
                    stack.enter_context(patch.object(rebuild,"audit_administrative_topology",return_value={"status":"ok"}))
                    report,_,_=rebuild.human_checks(data.grid,thematic,society,prepared)
                    self.assertEqual(report["counts"]["countries"],countries)
                    self.assertEqual(report["counts"]["provinces"],provinces)
                    self.assertEqual(report["status"],"ok" if (countries,provinces)==(80,420) else "failed")


if __name__ == "__main__":
    unittest.main()
