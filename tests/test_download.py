import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'skills/generate-world-atlas/scripts/download_engine.py'


class Response(io.BytesIO):
    def geturl(self):
        return 'https://release-assets.githubusercontent.com/test/asset'


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.is_file(), 'missing verified release downloader')
        spec = importlib.util.spec_from_file_location('atlas_download', SCRIPT)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.assets = Path(self.folder.name) / 'assets'
        self.assets.mkdir()
        self.cache = Path(self.folder.name) / 'cache'
        self.payload = b'verified wheel bytes'
        self.name = 'world_atlas_engine-1.1.0-py3-none-any.whl'
        self.manifest = {'repository': 'yuuranku/world-atlas-engine', 'engineVersion': '1.1.0',
                         'assets': {'engine': {'name': self.name, 'bytes': len(self.payload),
                         'sha256': hashlib.sha256(self.payload).hexdigest()}}}
        self.save()

    def save(self):
        (self.assets / 'release.json').write_text(json.dumps(self.manifest), encoding='utf-8')

    def fetch(self):
        return self.module.fetch_asset(self.assets, self.cache, 'engine')

    def test_pinned_url_and_hash_then_cached_offline_reuse(self):
        with patch.object(self.module, 'open_download', return_value=Response(self.payload)) as request:
            result = self.fetch()
        self.assertEqual(result.read_bytes(), self.payload)
        request.assert_called_once_with('https://github.com/yuuranku/world-atlas-engine/releases/download/v1.1.0/' + self.name)
        with patch.object(self.module, 'open_download', side_effect=AssertionError('network on cache hit')):
            self.assertEqual(self.fetch(), result)

    def test_bad_hash_never_installs_or_promotes_partial_file(self):
        with patch.object(self.module, 'open_download', return_value=Response(b'x' * len(self.payload))):
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                self.fetch()
        self.assertEqual(list(self.cache.rglob('*')), [])

    def test_pinned_development_release_uses_its_exact_version(self):
        self.name = 'world_atlas_engine-1.4.0.dev3-py3-none-any.whl'
        self.manifest['engineVersion'] = '1.4.0.dev3'
        self.manifest['assets']['engine']['name'] = self.name
        self.save()
        with patch.object(self.module, 'open_download', return_value=Response(self.payload)) as request:
            result = self.fetch()
        self.assertEqual(result.read_bytes(), self.payload)
        request.assert_called_once_with('https://github.com/yuuranku/world-atlas-engine/releases/download/v1.4.0.dev3/' + self.name)

    def test_truncated_download_never_promoted(self):
        with patch.object(self.module, 'open_download', return_value=Response(b'short')):
            with self.assertRaisesRegex(ValueError, 'size'):
                self.fetch()
        self.assertEqual(list(self.cache.rglob('*')), [])

    def test_oversized_download_never_promoted(self):
        with patch.object(self.module, 'open_download', return_value=Response(self.payload * 2)):
            with self.assertRaisesRegex(ValueError, 'size'):
                self.fetch()
        self.assertEqual(list(self.cache.rglob('*')), [])

    def test_corrupt_cache_fails_without_overwriting_user_file(self):
        self.cache.mkdir()
        file = self.cache / self.name
        file.write_bytes(b'corrupt')
        with patch.object(self.module, 'open_download', side_effect=AssertionError('network on corrupt cache')):
            with self.assertRaisesRegex(ValueError, 'cache'):
                self.fetch()
        self.assertEqual(file.read_bytes(), b'corrupt')

    def test_manifest_rejects_path_traversal_and_unpinned_version(self):
        self.manifest['assets']['engine']['name'] = '../other.whl'
        self.save()
        with self.assertRaises(ValueError):
            self.fetch()
        self.manifest['assets']['engine']['name'] = self.name
        self.manifest['engineVersion'] = 'latest'
        self.save()
        with self.assertRaises(ValueError):
            self.fetch()
        self.assertFalse(self.cache.exists())

    def test_network_failure_keeps_cache_clean(self):
        with patch.object(self.module, 'open_download', side_effect=OSError('offline')):
            with self.assertRaises(OSError):
                self.fetch()
        self.assertEqual(list(self.cache.rglob('*')), [])

    def test_redirect_rejects_non_https_and_untrusted_host(self):
        handler = self.module.ReleaseRedirectHandler()
        for url in ('http://github.com/a', 'https://example.com/asset'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                handler.redirect_request(None, None, 302, 'Found', {}, url)
