"""Independent source-root and delivery checks for shared physical relief."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import shapely
from scipy.optimize import brentq
from world_atlas.core.continuous_terrain import PhysicalTerrainField
from world_atlas.core.hypsometry import elevation_display_indices, elevation_display_thresholds
from world_atlas.core.cartographic_tiles import geometry_path_data
from world_atlas.core.physical_contour_stage import _binding
from world_atlas.core.svg_artifacts import encode_svgz, read_svgz

spec=importlib.util.spec_from_file_location('physical_relief_consumer',Path(__file__).resolve().parents[1]/'scripts/check_physical_relief_display.py')
consumer=importlib.util.module_from_spec(spec);spec.loader.exec_module(consumer)

def fixture(*,whole_land=False):
    raw=np.full((8,8),-40.);raw[:,:4]=650
    if whole_land:raw=np.repeat((100*(np.arange(8)+.5))[:,None],8,axis=1)
    terrain=PhysicalTerrainField(raw,land_mask=raw>0,sea_level_m=671,elevation_scale_m=1000,elevation_exponent=1.06)
    land=shapely.box(0,0,8 if whole_land else 4,8)
    palette=np.column_stack((np.arange(24)*7,np.arange(24)*3,np.arange(24)*5))
    level=int(elevation_display_indices(terrain.palette_elevation([650]))[0])
    color='#'+''.join(f'{int(value):02x}'for value in palette[level])
    surface=f'<g data-tile-layer="elevation-bands"><path d="M0,0 4,0 4,8 0,8Z" fill="{color}" clip-path="url(#land)"/></g>'
    tile={'key':'detail/0-0','payload':{'bounds':[0,0,8,8],'surface':surface,'ink':''},'land':land,'rectangle':shapely.box(0,0,8,8),'clips':{'land':land},'landClipId':'land'}
    return tile,terrain,palette,np.where(raw>0,0,1)

def sources(terrain,records=()):
    return consumer._source_curves(records,terrain,coordinate_error=0.)[0]

def city_fixture():
    tile,_,palette,_=fixture(whole_land=True)
    raw=np.repeat((10000+100*(np.arange(8)+.5))[:,None],8,axis=1)
    terrain=PhysicalTerrainField(raw,land_mask=raw>0,sea_level_m=671,elevation_scale_m=1000,elevation_exponent=1.06)
    tile['key']='city-detail/0-0';tile['citySupportMaskId']='support'
    tile['payload']['surface']='<defs><mask id="support" style="mask-type:alpha"><ellipse cx="4" cy="4" rx="3" ry="3"/></mask></defs>'
    tile['payload']['ink']=('<g data-city-refinement="true" mask="url(#support)" clip-path="url(#land)">'
        '<g data-tile-layer="elevation-contours"><path data-city-relief="true" data-contour-kind="minor" '
        'data-source-field="accepted-continuous-ground" data-height-m="10300" d="M1,3l5,0" fill="none"/></g></g>')
    return tile,terrain,palette,np.zeros((8,8),dtype=np.uint8),sources(terrain,[(10300,shapely.LineString(((1,3),(6,3))))])


def physical_export_fixture():
    _tile,terrain,palette,water=fixture()
    band=int(elevation_display_indices(terrain.palette_elevation([650]))[0])
    colour='#'+''.join(f'{int(value):02x}'for value in palette[band])
    document=('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 8 8">'
        '<defs><clipPath id="land-silhouette-clip" clipPathUnits="userSpaceOnUse">'
        '<path clip-rule="evenodd" d="M0,0h4v8h-4Z"/></clipPath></defs>'
        '<g clip-path="url(#land-silhouette-clip)"><g id="elevation-bands">'
        f'<g data-band="{band}"><path d="M0,0h8v8h-8Z" fill="{colour}" '
        'fill-rule="evenodd" stroke="none"/></g></g></g></svg>')
    return document,terrain,palette,water,band


def source_stage_fixture(directory):
    tile,terrain,_palette,_water=fixture(whole_land=True)
    identity={'source':'actual-linear-ground-fixture'}
    palettes=np.asarray(elevation_display_thresholds(),dtype=float)
    all_heights=np.asarray(terrain.contour_height_m(palettes),dtype=float)
    positive=all_heights>0;palettes,heights=palettes[positive],all_heights[positive]
    binding=_binding(terrain,heights,identity)
    fingerprint=consumer.sha256(json.dumps(binding,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    root=Path(directory);root.mkdir(exist_ok=True)
    body=[];graphs=[]
    for index,(palette,height)in enumerate(zip(palettes,heights,strict=True)):
        paths=[]
        if 50<height<750:
            y=brentq(lambda value:float(terrain.sample_points(np.array((1.,)),np.array((value,)))[0])-height,
                     .5,7.5,xtol=5e-15)
            path=np.array(((0,y),(1,y),(2+1e-10,y),(2+2e-10,y),(4,y),(8,y)),dtype=float)
            paths=[path]
            body.append(f'<path d="{geometry_path_data(shapely.LineString(path))}" fill="none" data-level="{float(palette)}"/>')
        points=np.concatenate(paths)if paths else np.empty((0,2),dtype=float)
        offsets=np.r_[0,np.cumsum([len(path)for path in paths],dtype=np.int64)]
        path=root/f'level-{index:03d}.npz'
        np.savez(path,level=np.asarray(height,dtype=np.float64),points=points,offsets=offsets)
        header={'schema':'physical-height-graphs-v1','fingerprint':fingerprint,'index':index,
                'heightMHex':float(height).hex(),'sha256':consumer._file_digest(path),
                'paths':len(paths),'vertices':len(points)}
        (root/f'level-{index:03d}.json').write_text(json.dumps(header),encoding='utf-8')
        graphs.append(paths)
    manifest={'schema':'physical-height-graphs-v1','binding':binding,'fingerprint':fingerprint,
              'status':'complete','completedHeights':len(heights),'totalHeights':len(heights)}
    (root/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    export=root/'elevation-contours.svgz'
    export.write_bytes(encode_svgz('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 8 8">'
        '<defs><clipPath id="land-silhouette-clip" clipPathUnits="userSpaceOnUse">'
        '<path d="M0,0h8v8h-8Z" clip-rule="evenodd"/></clipPath></defs>'
        '<g clip-path="url(#land-silhouette-clip)">'+''.join(body)+'</g></svg>'))
    source,_proof=consumer._read_major_sources(export,terrain,physical_land=tile['land'])
    return terrain,source,identity,graphs,export

class PhysicalReliefDisplayTests(unittest.TestCase):
    def test_committed_source_graphs_replay_all_svg_integer_nodes(self):
        with tempfile.TemporaryDirectory()as directory:
            terrain,source,identity,graphs,_export=source_stage_fixture(directory)
            result=consumer.check_staged_major_sources(directory,source,terrain=terrain,source_identity=identity)
            self.assertEqual(result['status'],'ok');self.assertEqual(result['mismatchCount'],0)
            self.assertEqual(result['sourceVertices']['mismatchCount'],0)
            count=sum(len(paths)for paths in graphs)
            self.assertGreater(count,1);self.assertEqual(result['checkedSourcePaths'],count)
            self.assertEqual(result['checkedDeliveredPaths'],count)
            self.assertEqual(result['checkedSourceVertices'],6*count)
            self.assertEqual(result['encodedAdjacentDuplicates'],count)
            self.assertEqual(len(result['files']),result['checkedHeights']+1)
            self.assertTrue(all('headerDigest'in item for name,item in result['files'].items()if name.endswith('.npz')))

    def test_saved_source_node_cannot_be_deleted_even_on_a_straight_run(self):
        with tempfile.TemporaryDirectory()as directory:
            terrain,source,identity,_graphs,export=source_stage_fixture(directory)
            document=consumer._tiles.ET.fromstring(read_svgz(export))
            node=next(node for node in document.iter()if node.get('data-level')is not None)
            points=consumer._line_points(node.get('d'))[0]
            node.set('d',geometry_path_data(shapely.LineString(np.delete(points,1,axis=0))))
            export.write_bytes(encode_svgz(consumer._tiles.ET.tostring(document,encoding='unicode')))
            changed,root_proof=consumer._read_major_sources(export,terrain,physical_land=shapely.box(0,0,8,8))
            self.assertEqual(root_proof['mismatchCount'],0)
            result=consumer.check_staged_major_sources(directory,changed,terrain=terrain,source_identity=identity)
            self.assertEqual(result['pathCountMismatchCount'],0)
            self.assertEqual(result['integerCoordinateMismatchCount'],1)
            self.assertEqual(result['status'],'failed')

    def test_svg_cannot_omit_an_entire_saved_source_curve(self):
        with tempfile.TemporaryDirectory()as directory:
            terrain,source,identity,_graphs,_export=source_stage_fixture(directory)
            changed=dict(source);changed['geometries']=source['geometries'][1:];changed['heights']=source['heights'][1:]
            result=consumer.check_staged_major_sources(directory,changed,terrain=terrain,source_identity=identity)
            self.assertEqual(result['pathCountMismatchCount'],1)
            self.assertEqual(result['status'],'failed')

    def test_staged_graph_identity_recipe_and_complete_headers_are_required(self):
        changes=('identity','incomplete','fingerprint','bytes','foreign')
        for change in changes:
            with self.subTest(change=change),tempfile.TemporaryDirectory()as directory:
                terrain,source,identity,graphs,_export=source_stage_fixture(directory)
                root=Path(directory);index=next(i for i,paths in enumerate(graphs)if paths)
                if change=='identity':identity={'source':'different-source'}
                elif change=='incomplete':
                    manifest=json.loads((root/'manifest.json').read_text());manifest['status']='building'
                    (root/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
                elif change=='fingerprint':
                    path=root/f'level-{index:03d}.json';header=json.loads(path.read_text());header['fingerprint']='foreign'
                    path.write_text(json.dumps(header),encoding='utf-8')
                elif change=='bytes':
                    path=root/f'level-{index:03d}.npz';path.write_bytes(path.read_bytes()+b'changed')
                else:(root/'level-999.npz').write_bytes(b'foreign-stage')
                with self.assertRaises(ValueError):
                    consumer.check_staged_major_sources(directory,source,terrain=terrain,source_identity=identity)

    def test_unquantized_stage_roots_are_checked_even_when_svg_integers_match(self):
        with tempfile.TemporaryDirectory()as directory:
            terrain,source,identity,graphs,_export=source_stage_fixture(directory)
            root=Path(directory)
            options=[(abs(float(terrain.sample_points(0,np.rint(paths[0][0,1]*1e8)/1e8))-
                          float(terrain.sample_points(0,paths[0][0,1]))),i)
                     for i,paths in enumerate(graphs)if paths]
            residual,index=max(options);self.assertGreater(residual,2e-8)
            path=root/f'level-{index:03d}.npz'
            with np.load(path,allow_pickle=False)as bundle:
                level,points,offsets=bundle['level'],bundle['points'],bundle['offsets']
            points[0,1]=np.rint(points[0,1]*1e8)/1e8
            np.savez(path,level=level,points=points,offsets=offsets)
            header_path=root/f'level-{index:03d}.json';header=json.loads(header_path.read_text())
            header['sha256']=consumer._file_digest(path);header_path.write_text(json.dumps(header),encoding='utf-8')
            result=consumer.check_staged_major_sources(directory,source,terrain=terrain,source_identity=identity)
            self.assertEqual(result['integerCoordinateMismatchCount'],0)
            self.assertGreater(result['sourceVertices']['mismatchCount'],0)
            self.assertEqual(result['status'],'failed')

    def test_native_coast_coverage_and_clear_water(self):
        tile,terrain,palette,water=fixture()
        result=consumer.check_tile(tile,terrain=terrain,palette=palette,water=water,source=sources(terrain))
        self.assertEqual(result['checkedDryCenters'],32)
        for name in ('colorMismatchCount','waterPaintCount','missingLandArea'):self.assertEqual(result[name],0)

    def test_wrong_colour_and_missing_coastal_fringe(self):
        for change in ('color','fringe'):
            tile,terrain,palette,water=fixture();old=tile['payload']['surface']
            tile['payload']['surface']=(old.replace(old.split('fill="')[1].split('"')[0],'#000000')if change=='color'else old.replace('4,0 4,8','3.4,0 3.4,8'))
            result=consumer.check_tile(tile,terrain=terrain,palette=palette,water=water,source=sources(terrain))
            self.assertGreater(result['colorMismatchCount'],0)
            if change=='fringe':self.assertGreater(result['missingLandArea'],0)

    def test_major_tile_replays_same_source_and_rejects_shift(self):
        tile,terrain,palette,water=fixture(whole_land=True)
        level=elevation_display_thresholds()[8];height=float(terrain.contour_height_m(level));y=height/100
        source=sources(terrain,[(height,shapely.LineString(((1,y),(6,y))))])
        for offset in (0,.5):
            tile['payload']['ink']=f'<g data-tile-layer="elevation-contours"><path d="M1,{y+offset}l5,0" fill="none" data-level="{level}" clip-path="url(#land)"/></g>'
            result=consumer.check_tile(tile,terrain=terrain,palette=palette,water=water,source=source)
            self.assertEqual(result['contours']['mismatchCount'],int(bool(offset)))

    def test_zero_boundary_not_drawn_as_land_contour(self):
        tile,terrain,palette,water=fixture();level=elevation_display_thresholds()[0]
        tile['payload']['ink']=f'<g data-tile-layer="elevation-contours"><path d="M4,0v8" fill="none" data-level="{level}" clip-path="url(#land)"/></g>'
        result=consumer.check_tile(tile,terrain=terrain,palette=palette,water=water,source=sources(terrain))
        self.assertEqual(result['zeroContourPathCount'],1);self.assertEqual(result['contours']['checkedPaths'],0)

    def test_base_colour_decoder_preserves_ancestor_clip(self):
        tile,terrain,palette,water=fixture()
        tile['payload']['surface']='<g clip-path="url(#land)">'+tile['payload']['surface'].replace(' clip-path="url(#land)"','')+'</g>'
        result=consumer.check_tile(tile,terrain=terrain,palette=palette,water=water,source=sources(terrain))
        self.assertEqual(result['colorMismatchCount'],0);self.assertEqual(result['missingLandArea'],0)

    def test_minor_metres_shared_cuts_and_source_immutable(self):
        tile,terrain,palette,water,source=city_fixture();original=terrain.native_m.copy()
        result=consumer.check_city_tile(tile,terrain=terrain,palette=palette,water=water,source=source)
        self.assertEqual(result['fineMinorContourPathCount'],1);self.assertEqual(result['contours']['mismatchCount'],0)
        np.testing.assert_array_equal(terrain.native_m,original)
        tile['payload']['ink']=tile['payload']['ink'].replace('M1,3','M1,3.03')
        result=consumer.check_city_tile(tile,terrain=terrain,palette=palette,water=water,source=source)
        self.assertEqual(result['contours']['mismatchCount'],1)

    def test_colour_overrides_rejected_inside_and_outside_support(self):
        for data in ('M1,3h2v2h-2Z','M0,0h4v1h-4Z'):
            tile,terrain,palette,water,source=city_fixture()
            tile['payload']['surface']+=f'<g data-tile-layer="elevation-bands"><path d="{data}" fill="#000000"/></g>'
            with self.assertRaisesRegex(ValueError,'no colour or major override'):consumer.check_city_tile(tile,terrain=terrain,palette=palette,water=water,source=source)

    def test_correct_height_major_override_rejected(self):
        tile,terrain,palette,water,source=city_fixture()
        tile['payload']['ink']=tile['payload']['ink'].replace('data-contour-kind="minor"','data-contour-kind="major"')
        with self.assertRaisesRegex(ValueError,'no colour or major override'):consumer.check_city_tile(tile,terrain=terrain,palette=palette,water=water,source=source)

    def test_old_sampling_palette_and_missing_source_metadata_rejected(self):
        for before,after in (('data-height-m="10300"','data-height-m="10300" data-sampling-cells="0.0625"'),
                             ('data-height-m="10300"','data-height-m="10300" data-level="0.5"'),
                             ('data-height-m="10300"','data-height-m="10301"'),
                             ('data-source-field="accepted-continuous-ground"',''),('data-contour-kind="minor"','')):
            tile,terrain,palette,water,source=city_fixture();tile['payload']['ink']=tile['payload']['ink'].replace(before,after)
            with self.assertRaises(ValueError):consumer.check_city_tile(tile,terrain=terrain,palette=palette,water=water,source=source)

    def test_single_alpha_and_exact_detail_clip_inherited(self):
        for before,after in (('mask="url(#support)"',''),('mask="url(#support)"','mask="url(#foreign)"'),
                             ('clip-path="url(#land)"',''),('clip-path="url(#land)"','clip-path="url(#foreign)"'),
                             ('data-city-refinement="true"','')):
            tile,terrain,palette,water,source=city_fixture();tile['payload']['ink']=tile['payload']['ink'].replace(before,after)
            with self.assertRaises(ValueError):consumer.check_city_tile(tile,terrain=terrain,palette=palette,water=water,source=source)

    def test_defs_nonpaint_and_flat_city_no_payload(self):
        tile,terrain,palette,water,source=city_fixture()
        tile['payload']['surface']=tile['payload']['surface'].replace('<defs>','<defs><path fill="#000000" d="M0,0h8v8h-8Z"/>')
        result=consumer.check_city_tile(tile,terrain=terrain,palette=palette,water=water,source=source)
        self.assertEqual(result['contours']['mismatchCount'],0)
        tile['payload']['ink']=''
        with self.assertRaisesRegex(ValueError,'Flat city'):consumer.check_city_tile(tile,terrain=terrain,palette=palette,water=water,source=source)

    def test_wrong_actual_source_not_excused_by_bilinear_mesh(self):
        _,terrain,_,_,_=city_fixture();points=np.array(((1.,3.),(6.,3.)))
        proof=consumer._contour_check(points,np.full(2,10300.01),terrain,coordinate_error=0.)
        self.assertEqual(proof['mismatchCount'],2);self.assertGreater(proof['maxResidualMeters'],.009)

    def test_text_coordinate_interval_is_the_only_source_allowance(self):
        _,terrain,_,_,_=city_fixture();points=np.array(((1.,3.000000004),(6.,3.000000004)))
        proof=consumer._contour_check(points,np.full(2,10300.),terrain,coordinate_error=5e-9)
        self.assertEqual(proof['mismatchCount'],0)
        wrong=consumer._contour_check(points,np.full(2,10300.01),terrain,coordinate_error=5e-9)
        self.assertEqual(wrong['mismatchCount'],2)

    def test_city_source_document_binds_actual_grid_and_minor_contract(self):
        _,terrain,_,_,_=city_fixture()
        document={'schema':'accepted-physical-minor-curves-v1','coordinateSpace':'native-cell','gridDigest':'current',
                  'curves':[{'heightM':10300.,'geometry':shapely.geometry.mapping(shapely.LineString(((1,3),(6,3))))}]}
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'city-contours.json';path.write_text(json.dumps(document),encoding='utf-8')
            source,proof=consumer._read_city_sources(path,terrain,'current')
            self.assertEqual(proof['mismatchCount'],0);self.assertEqual(len(source['geometries']),1)
            with self.assertRaisesRegex(ValueError,'canonical grid'):consumer._read_city_sources(path,terrain,'foreign')
            document['curves'][0]['heightM']=10300.01;path.write_text(json.dumps(document),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'100-metre'):consumer._read_city_sources(path,terrain,'current')

    def test_cut_endpoints_inherit_original_source_root_proof(self):
        tile,terrain,palette,water,source=city_fixture();tile['payload']['bounds']=[2,0,2,8]
        tile['rectangle']=shapely.box(2,0,4,8)
        tile['payload']['ink']=tile['payload']['ink'].replace('M1,3l5,0','M1.5,3l3,0')
        result=consumer.check_city_tile(tile,terrain=terrain,palette=palette,water=water,source=source)
        self.assertEqual(result['contours']['mismatchCount'],0)

    def test_major_source_owns_the_same_clip_and_defs_are_not_roots(self):
        tile,terrain,_,_=fixture(whole_land=True)
        level=elevation_display_thresholds()[8];y=float(terrain.contour_height_m(level))/100
        document=('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 8 8"><defs>'
            '<clipPath id="land-silhouette-clip" clipPathUnits="userSpaceOnUse">'
            '<path d="M0,0h8v8h-8Z" clip-rule="evenodd"/></clipPath></defs>'
            f'<g clip-path="url(#land-silhouette-clip)"><path d="M1,{y:.8f}l5,0" '
            f'fill="none" data-level="{level}"/></g></svg>')
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'elevation-contours.svgz';path.write_bytes(encode_svgz(document))
            source,proof=consumer._read_major_sources(path,terrain,physical_land=tile['land'])
            self.assertEqual(len(source['geometries']),1)
            self.assertEqual(proof['checkedPoints'],2);self.assertEqual(proof['mismatchCount'],0)
            path.write_bytes(encode_svgz(document.replace('clip-path="url(#land-silhouette-clip)"','')))
            with self.assertRaises(ValueError):consumer._read_major_sources(path,terrain,physical_land=tile['land'])
            path.write_bytes(encode_svgz(document))
            with self.assertRaisesRegex(ValueError,'physical land authority'):
                consumer._read_major_sources(path,terrain,physical_land=shapely.box(0,0,7,8))
            for before,after in (('viewBox="0 0 8 8"','viewBox="0 0 16 8"'),
                                 ('<g clip-path=','<g transform="translate(1,0)" clip-path=')):
                path.write_bytes(encode_svgz(document.replace(before,after)))
                with self.assertRaisesRegex(ValueError,'native'):
                    consumer._read_major_sources(path,terrain,physical_land=tile['land'])
            for before,after in (('<defs>','<defs transform="translate(1,0)">'),
                                 ('<clipPath id=','<clipPath transform="translate(1,0)" id='),
                                 ('<path d="M0,0h8v8h-8Z"','<path transform="translate(1,0)" d="M0,0h8v8h-8Z"'),
                                 ('clip-rule="evenodd"','clip-rule="nonzero"')):
                path.write_bytes(encode_svgz(document.replace(before,after)))
                with self.assertRaisesRegex(ValueError,'native'):
                    consumer._read_major_sources(path,terrain,physical_land=tile['land'])

    def test_major_ink_requires_its_actual_ancestor_clip(self):
        tile,terrain,palette,water=fixture(whole_land=True)
        level=elevation_display_thresholds()[8];height=float(terrain.contour_height_m(level));y=height/100
        source=sources(terrain,[(height,shapely.LineString(((1,y),(6,y))))])
        for clip in ('',' clip-path="url(#foreign)"'):
            tile['payload']['ink']=f'<g data-tile-layer="elevation-contours"{clip}><path d="M1,{y}l5,0" fill="none" data-level="{level}"/></g>'
            with self.assertRaises(ValueError):consumer.check_tile(tile,terrain=terrain,palette=palette,water=water,source=source)

    def test_full_physical_export_checks_every_owned_clip_and_land_colour(self):
        document,terrain,palette,water,_band=physical_export_fixture()
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'physical-surface.svgz';path.write_bytes(encode_svgz(document))
            land,result=consumer.check_physical_export(path,terrain=terrain,palette=palette,water=water)
            self.assertTrue(land.equals(shapely.box(0,0,4,8)))
            self.assertEqual(result['checkedNativeCenters'],64)
            self.assertEqual(result['checkedDryCenters'],32);self.assertEqual(result['checkedWetCenters'],32)
            for key in ('clipMismatchCount','missingLandColorCount','wrongLandColorCount','colorMismatchCount','waterPaintCount'):
                self.assertEqual(result[key],0)
            self.assertEqual(result['status'],'ok');self.assertEqual(result['exportBytes'],path.stat().st_size)
            self.assertEqual(result['exportDigest'],consumer._file_digest(path))
            self.assertEqual(result['exportEncoding'],'gzip')
            self.assertEqual(result['decodedExportBytes'],len(document.encode('utf-8')))
            self.assertEqual(result['decodedExportDigest'],consumer.sha256(document.encode('utf-8')).hexdigest())

    def test_full_physical_source_requires_the_canonical_compressed_artifact(self):
        document,terrain,palette,water,_band=physical_export_fixture()
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'physical-surface.svg'
            path.write_text(document,encoding='utf-8')
            with self.assertRaises(ValueError):
                consumer.check_physical_export(path,terrain=terrain,palette=palette,water=water)
            path=Path(directory)/'physical-surface.svgz'
            for content in (document.encode('utf-8'),encode_svgz(document)[:-5]):
                path.write_bytes(content)
                with self.assertRaises((ValueError,EOFError,OSError)):
                    consumer.check_physical_export(path,terrain=terrain,palette=palette,water=water)

    def test_full_contour_source_requires_a_complete_gzip_and_reports_both_digests(self):
        with tempfile.TemporaryDirectory()as directory:
            terrain,_source,_identity,_graphs,path=source_stage_fixture(directory)
            markup=read_svgz(path)
            _source,proof=consumer._read_major_sources(path,terrain,physical_land=shapely.box(0,0,8,8))
            self.assertEqual(proof['artifact']['encoding'],'gzip')
            self.assertEqual(proof['artifact']['bytes'],path.stat().st_size)
            self.assertEqual(proof['artifact']['decodedBytes'],len(markup.encode('utf-8')))
            self.assertEqual(proof['artifact']['digest'],consumer._file_digest(path))
            self.assertEqual(proof['artifact']['decodedDigest'],consumer.sha256(markup.encode('utf-8')).hexdigest())
            legacy=path.with_suffix('.svg');legacy.write_text(markup,encoding='utf-8')
            with self.assertRaises(ValueError):
                consumer._read_major_sources(legacy,terrain,physical_land=shapely.box(0,0,8,8))
            path.write_bytes(encode_svgz(markup)[:-5])
            with self.assertRaises((ValueError,EOFError,OSError)):
                consumer._read_major_sources(path,terrain,physical_land=shapely.box(0,0,8,8))

    def test_full_export_missing_colour_is_not_hidden_by_surface_backfill(self):
        document,terrain,palette,water,band=physical_export_fixture()
        backfill=f'<rect x="0" y="0" width="8" height="8" fill="#{palette[band,0]:02x}{palette[band,1]:02x}{palette[band,2]:02x}"/>'
        document=document.replace('<g clip-path=',backfill+'<g clip-path=',1).replace('M0,0h8v8h-8Z','M0,0h3.4v8h-3.4Z')
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'physical-surface.svgz';path.write_bytes(encode_svgz(document))
            _land,result=consumer.check_physical_export(path,terrain=terrain,palette=palette,water=water)
            self.assertEqual(result['clipMismatchCount'],0);self.assertEqual(result['missingLandColorCount'],8)
            self.assertEqual(result['status'],'failed')

    def test_full_export_detects_wrong_native_land_clip_and_water_colour(self):
        document,terrain,palette,water,_band=physical_export_fixture()
        document=document.replace('M0,0h4v8h-4Z','M0,0h5v8h-5Z')
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'physical-surface.svgz';path.write_bytes(encode_svgz(document))
            _land,result=consumer.check_physical_export(path,terrain=terrain,palette=palette,water=water)
            self.assertEqual(result['waterCentersDisplayedAsLand'],8);self.assertEqual(result['waterPaintCount'],8)
            self.assertEqual(result['status'],'failed')

    def test_full_export_replays_actual_band_paint_order(self):
        document,terrain,palette,water,band=physical_export_fixture()
        correct=document.split(f'<g data-band="{band}">')[1].split('</g>')[0]
        wrong='<g data-band="0"><path d="M0,0h2v8h-2Z" fill="#000000" fill-rule="evenodd" stroke="none"/></g>'
        for restore in (False,True):
            markup=document.replace('</g></g></g></svg>',f'</g>{wrong}'+(f'<g data-band="{band}">{correct}</g>'if restore else '')+'</g></g></svg>')
            with tempfile.TemporaryDirectory()as directory:
                path=Path(directory)/'physical-surface.svgz';path.write_bytes(encode_svgz(markup))
                _land,result=consumer.check_physical_export(path,terrain=terrain,palette=palette,water=water)
                self.assertEqual(result['wrongLandColorCount'],0 if restore else 16)
                self.assertEqual(result['paintOrder'],[band,0,band]if restore else[band,0])

    def test_full_export_rejects_foreign_missing_or_secondary_clips(self):
        document,terrain,palette,water,_band=physical_export_fixture()
        changes=((' clip-path="url(#land-silhouette-clip)"',''),
                 (' clip-path="url(#land-silhouette-clip)"',' clip-path="url(#foreign)"'),
                 ('<g id="elevation-bands">','<g id="elevation-bands" clip-path="url(#land-silhouette-clip)">'))
        for before,after in changes:
            with tempfile.TemporaryDirectory()as directory:
                path=Path(directory)/'physical-surface.svgz';path.write_bytes(encode_svgz(document.replace(before,after)))
                with self.assertRaisesRegex(ValueError,'owned|clipping'):
                    consumer.check_physical_export(path,terrain=terrain,palette=palette,water=water)

    def test_full_export_definitions_are_not_paint_and_frame_is_required(self):
        document,terrain,palette,water,_band=physical_export_fixture()
        document=document.replace('<defs>','<defs><path d="M0,0h8v8h-8Z" fill="#000000" data-band="0"/>')
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'physical-surface.svgz';path.write_bytes(encode_svgz(document))
            _land,result=consumer.check_physical_export(path,terrain=terrain,palette=palette,water=water)
            self.assertEqual(result['colorMismatchCount'],0);self.assertEqual(result['bandPathCount'],1)
            path.write_bytes(encode_svgz(document.replace('viewBox="0 0 8 8"','viewBox="0 0 16 8"')))
            with self.assertRaisesRegex(ValueError,'native world frame'):
                consumer.check_physical_export(path,terrain=terrain,palette=palette,water=water)

    def test_full_export_requires_actual_opaque_palette_and_native_coordinates(self):
        document,terrain,palette,water,band=physical_export_fixture()
        colour='#'+''.join(f'{int(value):02x}'for value in palette[band])
        for before,after in ((f'fill="{colour}"','fill="#010101"'),
                             (f'data-band="{band}"',''),('<g id="elevation-bands">','<g id="elevation-bands" opacity=".5">'),
                             ('<g id="elevation-bands">','<g id="elevation-bands" transform="translate(1,0)">')):
            with tempfile.TemporaryDirectory()as directory:
                path=Path(directory)/'physical-surface.svgz';path.write_bytes(encode_svgz(document.replace(before,after)))
                with self.assertRaises(ValueError):consumer.check_physical_export(path,terrain=terrain,palette=palette,water=water)

if __name__=='__main__':unittest.main()
