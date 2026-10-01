from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from world_atlas.core.svg_artifacts import encode_svgz, read_svgz


class SvgArtifactsTests(unittest.TestCase):
    def test_full_document_preserves_paths_clips_and_unicode(self):
        document = ('<svg xmlns="http://www.w3.org/2000/svg"><title>真实地表</title>'
                    '<defs><clipPath id="land"><path d="M.00000001,0h1v1h-1z"/>'
                    '</clipPath></defs><g clip-path="url(#land)">' 
                    '<path data-level="4085.31305416" d="M1,2l.00000001,-.00000001 0,1"/>'
                    '</g></svg>')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'physical-surface.svgz'
            path.write_bytes(encode_svgz(document))
            restored = read_svgz(path)
        self.assertEqual(restored, document)
        self.assertEqual(ET.tostring(ET.fromstring(restored)), ET.tostring(ET.fromstring(document)))

    def test_repeated_encoding_is_identical(self):
        document = '<svg><path d="M0,0l1,1"/></svg>'
        self.assertEqual(encode_svgz(document), encode_svgz(document))

    def test_corrupt_stream_is_rejected(self):
        data = bytearray(encode_svgz('<svg><path d="M0,0l1,1"/></svg>'))
        data[-8] ^= 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'physical-surface.svgz'
            path.write_bytes(data)
            with self.assertRaises(OSError):
                read_svgz(path)


if __name__ == '__main__':
    unittest.main()
