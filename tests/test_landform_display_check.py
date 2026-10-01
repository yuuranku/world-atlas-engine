"""Independent delivered masks, glyph direction and paint-server counterexamples."""
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np


spec=importlib.util.spec_from_file_location("landform_consumer",Path(__file__).resolve().parents[1]/"scripts/check_landform_display.py")
consumer=importlib.util.module_from_spec(spec)
spec.loader.exec_module(consumer)


def fixture():
    shape=(4,5)
    grid=SimpleNamespace(shape=shape,water=np.zeros(shape,dtype=np.uint8),
        metadata={"extents":{"north":20.,"south":-20.,"west":-25.,"east":25.}})
    grid.water[:,4]=1
    masks={kind:np.zeros(shape,dtype=bool) for kind in (*consumer.AREA_TYPES,"steep-slope")}
    masks["wetland"][:,0]=True
    masks["snow-mountain"][:,1]=True
    masks["arid-upland"][:,2]=True
    masks["steep-slope"][1,3]=True
    east=np.zeros(shape)
    east[1,3]=1
    inventory=SimpleNamespace(masks=masks,direction_east=east,direction_north=np.zeros(shape))
    defs=['<defs><clipPath id="land-silhouette-clip" clipPathUnits="userSpaceOnUse"><path d="M0,0h4v4h-4Z"/></clipPath>']
    hints=[]
    textures=[]
    for column,kind in enumerate(consumer.AREA_TYPES):
        identifier="pattern-"+kind
        defs.append(f'<pattern id="{identifier}" data-landform-pattern="{kind}" patternUnits="userSpaceOnUse" '
                    'patternContentUnits="userSpaceOnUse" width="8" height="8"><path d="M0,0h2" fill="none" stroke="blue"/></pattern>')
        common=f'data-landform-type="{kind}" d="M{column},0h1v4h-1Z" clip-path="url(#land-silhouette-clip)"'
        hints.append(f'<path {common} fill="#abc" opacity=".1"/>')
        textures.append(f'<path {common} fill="url(#{identifier})" data-landform-texture="true" data-min-scale="4" data-max-scale="256"/>')
    slope=('<path data-landform-type="steep-slope" data-landform-texture="true" data-landform-hachure="downslope" '
           'data-min-scale="32" data-max-scale="256" fill="none" stroke="brown" vector-effect="non-scaling-stroke" '
           'clip-path="url(#land-silhouette-clip)" d="M3.42,1.5l.16,0"/>')
    body=''.join(defs)+'</defs><g data-tile-layer="geographic-textures">'+''.join(hints+textures)+slope+'</g>'
    tile=consumer.export_tile(body,grid,key="full")
    overview='<defs>'+defs[0].removeprefix('<defs>')+'</defs><g data-tile-layer="geographic-textures">'+''.join(hints)+'</g>'
    return grid,inventory,body,tile,overview


class LandformDisplayCheckTests(unittest.TestCase):
    def test_actual_overview_reads_water_clip_without_licensing_water_clipped_landforms(self):
        grid,inventory,_,_,overview=fixture()
        water='<clipPath id="water-silhouette-clip" clipPathUnits="userSpaceOnUse"><path d="M4,0h1v4h-1Z"/></clipPath>'
        overview=overview.replace('</defs>',water+'</defs>')
        overview+='<path clip-path="url(#water-silhouette-clip)" d="M4,1h1" fill="none" stroke="blue"/>'
        tile=consumer.export_tile(overview,grid,key="actual-overview",overview=True)
        self.assertEqual(set(tile["clips"]),{"land-silhouette-clip","water-silhouette-clip"})
        self.assertEqual(consumer.check_markup(overview,tile,grid,inventory,overview=True)["status"],"ok")
        wet_water=overview.replace('data-landform-type="wetland" d="M0,0h1v4h-1Z" clip-path="url(#land-silhouette-clip)"',
            'data-landform-type="wetland" d="M0,0h1v4h-1Z" clip-path="url(#water-silhouette-clip)"')
        with self.assertRaisesRegex(ValueError,"shared-land composition"):
            consumer.check_markup(wet_water,tile,grid,inventory,overview=True)

    def test_actual_overview_requires_both_unique_native_physical_clips(self):
        grid,_,_,_,overview=fixture()
        with self.assertRaisesRegex(ValueError,"lacks"):
            consumer.export_tile(overview,grid,key="actual-overview",overview=True)
        water='<clipPath id="water-silhouette-clip" clipPathUnits="userSpaceOnUse"><path d="M4,0h1v4h-1Z"/></clipPath>'
        markup=overview.replace('</defs>',water+'</defs>')
        for incorrect in (markup.replace('id="water-silhouette-clip"','id="foreign-clip"'),
                          markup.replace('</defs>',water+'</defs>'),
                          markup.replace('clipPathUnits="userSpaceOnUse"','clipPathUnits="objectBoundingBox"')):
            with self.subTest(markup=incorrect),self.assertRaisesRegex(ValueError,"foreign, duplicate or non-native"):
                consumer.export_tile(incorrect,grid,key="actual-overview",overview=True)

    def static_proof_fixture(self,directory):
        world=Path(directory)
        review=world/"review"
        grid_path=world/"grid"
        source_path=world/"source"/"physical-fields.npz"
        for path in (review,grid_path,source_path.parent):
            path.mkdir()
        fingerprints={}
        for path in (grid_path/"world-grid.npz",grid_path/"world-grid.json",source_path):
            path.write_bytes(("original-"+path.name).encode())
            fingerprints[path.relative_to(world).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
        original=world/"accepted-source"
        for name in fingerprints:
            destination=original/name
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes((world/name).read_bytes())
        field_sha=fingerprints["source/physical-fields.npz"]
        (world/"regeneration.json").write_text(json.dumps({"schema":"accepted-world-v2","status":"complete",
            "rebuildKind":"human-only-v1","humanStatus":"ok","sourceWorld":str(original),"fieldSha256":field_sha}))
        (source_path.parent/"physical-fields-verification.json").write_text(json.dumps({
            "schema":"verified-physical-surface-replay-v1","verifiedExact":True,"originalSourceUnchanged":True,
            "nativePaletteReconstructionExact":True,"verifiedBundleSha256":field_sha,"recipe":{"seed":17}}))
        (source_path.parent/"provenance.json").write_text(json.dumps({"recipe":{"seed":17}}))
        source_hashes,contract=consumer.accepted_source_evidence(review,grid_path,source_path,require_complete=True)
        grid,inventory,markup,tile,_=fixture()
        (review/"landform-regions.svg").write_text(markup)
        measured=consumer.check_markup(markup,tile,grid,inventory)
        proof={"schema":"delivered-landform-display-v1","scope":"final-static-landform-precheck",
               "status":"ok","static":measured,"inputs":{"review":str(review),"grid":str(grid_path),
               "physicalSource":str(source_path),"landformSVG_SHA256":hashlib.sha256(markup.encode()).hexdigest(),
               "sourceSHA256":source_hashes},"acceptedSource":contract}
        report_path=world/"static-proof.json"
        report_path.write_text(json.dumps(proof))
        return review,grid_path,source_path,report_path

    def test_static_reuse_binds_exact_export_report_and_previously_recorded_source_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            review,grid,source,report=self.static_proof_fixture(directory)
            proof,binding=consumer.bind_static_proof(review,grid_path=grid,physical_source_path=source,static_report_path=report)
            self.assertEqual(proof["static"]["status"],"ok")
            self.assertEqual(binding["reportSHA256"],hashlib.sha256(report.read_bytes()).hexdigest())
            self.assertEqual(len(binding["immutableSourceSHA256"]),3)

    def test_changed_export_or_native_source_cannot_reuse_an_old_static_pass(self):
        for name in ("export","grid","raw"):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as directory:
                review,grid,source,report=self.static_proof_fixture(directory)
                target={"export":review/"landform-regions.svg","grid":grid/"world-grid.npz","raw":source}[name]
                target.write_bytes(target.read_bytes()+b"changed")
                with self.assertRaisesRegex(ValueError,"changed"):
                    consumer.bind_static_proof(review,grid_path=grid,physical_source_path=source,static_report_path=report)

    def test_failed_or_different_input_static_proof_is_rejected(self):
        for mutate in (lambda proof:proof.update(status="failed"),
                       lambda proof:proof["inputs"].update(grid="other-grid"),
                       lambda proof:proof["static"]["textures"]["wetland"].update(mismatchCount=1)):
            with tempfile.TemporaryDirectory() as directory:
                review,grid,source,report=self.static_proof_fixture(directory)
                proof=json.loads(report.read_text())
                mutate(proof)
                report.write_text(json.dumps(proof))
                with self.assertRaises(ValueError):
                    consumer.bind_static_proof(review,grid_path=grid,physical_source_path=source,static_report_path=report)

    def test_current_source_contract_must_have_exact_verified_bundle_sha(self):
        for path_name,field,value in (("regeneration.json","fieldSha256","wrong-sha"),
                                      ("source/physical-fields-verification.json","verifiedBundleSha256","wrong-sha"),
                                      ("source/physical-fields-verification.json","verifiedExact",False),
                                      ("source/physical-fields-verification.json","recipe",{"seed":18})):
            with self.subTest(field=field),tempfile.TemporaryDirectory()as directory:
                review,grid,source,report=self.static_proof_fixture(directory)
                path=Path(directory)/path_name
                record=json.loads(path.read_text());record[field]=value;path.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError,"incorrect SHA or evidence"):
                    consumer.bind_static_proof(review,grid_path=grid,physical_source_path=source,static_report_path=report)

    def test_late_fingerprinting_cannot_reuse_a_static_pass_after_all_sources_change(self):
        with tempfile.TemporaryDirectory()as directory:
            review,grid,source,report=self.static_proof_fixture(directory)
            world=Path(directory)
            source.write_bytes(source.read_bytes()+b"changed")
            (world/"accepted-source/source/physical-fields.npz").write_bytes(source.read_bytes())
            digest=hashlib.sha256(source.read_bytes()).hexdigest()
            for name,field in (("regeneration.json","fieldSha256"),
                               ("source/physical-fields-verification.json","verifiedBundleSha256")):
                path=world/name
                record=json.loads(path.read_text());record[field]=digest;path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,"own recorded input fingerprints"):
                consumer.bind_static_proof(review,grid_path=grid,physical_source_path=source,static_report_path=report)

    def test_building_export_can_capture_sources_but_cannot_claim_full_runtime_acceptance(self):
        with tempfile.TemporaryDirectory()as directory:
            review,grid,source,report=self.static_proof_fixture(directory)
            path=Path(directory)/"regeneration.json"
            record=json.loads(path.read_text());record["status"]="building";path.write_text(json.dumps(record))
            hashes,_contract=consumer.accepted_source_evidence(review,grid,source,require_complete=False)
            self.assertEqual(len(hashes),3)
            with self.assertRaisesRegex(ValueError,"current accepted human rebuild"):
                consumer.bind_static_proof(review,grid_path=grid,physical_source_path=source,static_report_path=report)

    def test_obsolete_repair_file_does_not_replace_current_source_contract(self):
        with tempfile.TemporaryDirectory()as directory:
            review,grid,source,report=self.static_proof_fixture(directory)
            world=Path(directory)
            (world/"regeneration.json").unlink()
            (world/"repair-check.json").write_text(json.dumps({"status":"ok"}))
            with self.assertRaises(FileNotFoundError):
                consumer.bind_static_proof(review,grid_path=grid,physical_source_path=source,static_report_path=report)

    def test_actual_source_masks_and_measured_direction_pass(self):
        grid,inventory,markup,tile,_=fixture()
        result=consumer.check_markup(markup,tile,grid,inventory)
        self.assertEqual(result["status"],"ok")
        self.assertEqual(result["hachures"]["checkedLines"],1)
        self.assertEqual(result["patterns"],3)
        self.assertEqual(result["tints"]["wetland"]["checkedCenters"],20)
        self.assertEqual(result["textures"]["snow-mountain"]["sourceSupportCenters"],4)

    def test_missing_region_wrong_mask_and_dry_land_spill_are_detected(self):
        for old,new in (('d="M0,0h1v4h-1Z"','d="M0,0h.1v4h-.1Z"'),
                        ('d="M1,0h1v4h-1Z"','d="M3,0h1v4h-1Z"'),
                        ('d="M2,0h1v4h-1Z"','d="M2,0h2v4h-2Z"')):
            grid,inventory,markup,tile,_=fixture()
            with self.subTest(new=new):
                result=consumer.check_markup(markup.replace(old,new),tile,grid,inventory)
                self.assertEqual(result["status"],"failed")

    def test_common_clip_really_prevents_water_paint(self):
        grid,inventory,markup,tile,_=fixture()
        markup=markup.replace('d="M2,0h1v4h-1Z"','d="M2,0h3v4h-3Z"')
        result=consumer.check_markup(markup,tile,grid,inventory)
        self.assertEqual(result["tints"]["arid-upland"]["waterPaintCount"],0)
        with self.assertRaisesRegex(ValueError,"shared-land"):
            consumer.check_markup(markup.replace(' clip-path="url(#land-silhouette-clip)"',''),tile,grid,inventory)

    def test_uphill_or_shifted_or_duplicated_hachures_fail(self):
        for mutate in (lambda text:text.replace('d="M3.42,1.5l.16,0"','d="M3.58,1.5l-.16,0"'),
                       lambda text:text.replace('d="M3.42,1.5l.16,0"','d="M3.42,2.5l.16,0"'),
                       lambda text:text.replace('</g>',text[text.index('<path data-landform-type="steep-slope"'):text.index('</g>')]+'</g>')):
            grid,inventory,markup,tile,_=fixture()
            self.assertEqual(consumer.check_markup(mutate(markup),tile,grid,inventory)["status"],"failed")

    def test_physical_direction_accounts_for_latitude_and_non_square_degree_cells(self):
        grid,inventory,markup,tile,_=fixture()
        grid.metadata["extents"]={"north":70.,"south":30.,"west":-50.,"east":50.}
        inventory.direction_east[1,3]=np.sqrt(.5)
        inventory.direction_north[1,3]=np.sqrt(.5)
        cos=np.cos(np.radians(55.))
        ratio=10/(20*cos)
        dx=.16*ratio/(ratio+1)
        dy=-.16/(ratio+1)
        markup=markup.replace('d="M3.42,1.5l.16,0"',f'd="M{3.5-dx/2},{1.5-dy/2}l{dx},{dy}"')
        self.assertEqual(consumer.check_markup(markup,tile,grid,inventory)["status"],"ok")

    def test_patterns_are_unique_kind_matched_and_scaled_not_sand(self):
        changes=(('data-landform-pattern="wetland"','data-landform-pattern="arid-upland"'),
                 ('data-min-scale="4"','data-min-scale="0"'),('width="8"','width="0"'),
                 ('fill="url(#pattern-wetland)"','fill="url(#foreign)"'))
        for old,new in changes:
            grid,inventory,markup,tile,_=fixture()
            with self.subTest(new=new),self.assertRaises(ValueError):
                consumer.check_markup(markup.replace(old,new),tile,grid,inventory)

    def test_overview_is_actual_common_clip_hint_without_micro_texture(self):
        grid,inventory,markup,tile,overview=fixture()
        result=consumer.check_markup(overview,tile,grid,inventory,overview=True)
        self.assertEqual(result["status"],"ok")
        self.assertEqual(result["patterns"],0)
        with self.assertRaisesRegex(ValueError,"overview"):
            consumer.check_markup(markup,tile,grid,inventory,overview=True)
        # Unrelated hidden source images are harmless; hidden actual landform
        # paint is still rejected rather than silently counted as coverage.
        self.assertEqual(consumer.check_markup(overview+'<image hidden="hidden"/>',tile,grid,inventory,overview=True)["status"],"ok")

    def test_pattern_ids_must_be_local_when_actual_native_blocks_mount_together(self):
        grid,inventory,markup,tile,_=fixture()
        tile["key"]="detail/0-0"
        with self.assertRaisesRegex(ValueError,"collide"):
            consumer.check_markup(markup,tile,grid,inventory,local=True)
        for kind in consumer.AREA_TYPES:
            markup=markup.replace('pattern-'+kind,'pattern-'+kind+'-tile-detail-0-0')
        self.assertEqual(consumer.check_markup(markup,tile,grid,inventory,local=True)["status"],"ok")


if __name__=="__main__":
    unittest.main()
