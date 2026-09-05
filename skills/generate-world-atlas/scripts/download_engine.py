"""Download immutable release assets, verify before use, and reuse a verified cache."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


def trusted_url(url):
    parsed = urlsplit(url)
    if (parsed.scheme != 'https' or parsed.username or parsed.password
            or parsed.port not in (None, 443)
            or parsed.hostname not in {'github.com', 'release-assets.githubusercontent.com',
                                       'objects.githubusercontent.com'}):
        raise ValueError('release downloads require HTTPS on GitHub release hosts')


class ReleaseRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        trusted_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_download(url):
    trusted_url(url)
    return build_opener(ReleaseRedirectHandler()).open(
        Request(url, headers={'User-Agent': 'world-atlas-skill-bootstrap'}), timeout=45)


def load_release(assets):
    record = json.loads((assets / 'release.json').read_text(encoding='utf-8'))
    version = record['engineVersion']
    if not re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+', version):
        raise ValueError('release must pin a numeric engine version, never latest')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', record['repository']):
        raise ValueError('invalid GitHub repository')
    if not record['assets']:
        raise ValueError('release contains no assets')
    for item in record['assets'].values():
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', item['name']):
            raise ValueError('unsafe release asset name')
        if not re.fullmatch(r'[a-f0-9]{64}', item['sha256']):
            raise ValueError('invalid SHA-256 in release')
        if type(item['bytes']) is not int or not 0 < item['bytes'] < 2**31:
            raise ValueError('invalid asset size')
    if record['assets']['engine']['name'] != f'world_atlas_engine-{version}-py3-none-any.whl':
        raise ValueError('engine wheel and version disagree')
    return record


def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def fetch_asset(assets: Path, cache: Path, key: str) -> Path:
    release = load_release(assets)
    item = release['assets'][key]
    root = cache.resolve()
    destination = root / item['name']
    if destination.is_symlink() or not destination.resolve().is_relative_to(root):
        raise ValueError('cache path escapes the download directory')
    if destination.exists():
        if (not destination.is_file() or destination.stat().st_size != item['bytes']
                or digest(destination) != item['sha256']):
            raise ValueError(f'corrupt cache file; preserve it and select a new --cache directory: {destination}')
        return destination
    url = f"https://github.com/{release['repository']}/releases/download/v{release['engineVersion']}/{item['name']}"
    root.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=root, suffix='.part', delete=False) as output:
            temporary = Path(output.name)
            with open_download(url) as response:
                trusted_url(response.geturl())
                count = 0
                while chunk := response.read(1024 * 1024):
                    count += len(chunk)
                    if count > item['bytes']:
                        raise ValueError('download size exceeds release manifest')
                    output.write(chunk)
        if count != item['bytes']:
            raise ValueError('download size does not match release manifest')
        if digest(temporary) != item['sha256']:
            raise ValueError('download SHA-256 does not match release manifest')
        # Hard-link promotion is atomic and refuses an existing destination.
        # The temporary file and cache are on the same filesystem.
        try:
            destination.hardlink_to(temporary)
        except FileExistsError:
            if destination.is_symlink() or digest(destination) != item['sha256']:
                raise ValueError('cache changed during download; select a new cache directory')
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--asset', choices=('engine', 'accepted-v5'), default='engine')
    args = parser.parse_args()
    try:
        path = fetch_asset(Path(__file__).resolve().parents[1] / 'assets', args.cache, args.asset)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f'Download failed: {error}\nCheck network/release access; no unverified package was installed.\n')
    print(json.dumps({'path': str(path), 'sha256': digest(path), 'verified': True}))


if __name__ == '__main__':
    main()
