"""Install the pinned engine and renderer in a NEW local runtime directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv
from download_engine import fetch_asset, load_release


def check_assets(assets: Path) -> dict:
    manifest_path = assets / 'asset-manifest.json'
    if not manifest_path.is_file():
        return {'ok': False, 'assets': [], 'error': 'assets manifest is missing'}
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    records = []
    root = assets.resolve()
    for relative, expected in manifest['files'].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError('asset path escapes skill directory')
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        records.append({'file': relative, 'ok': actual == expected, 'sha256': actual})
    version = manifest['engineVersion']
    release = load_release(assets)
    required = {'release.json', 'renderer/package.json', 'renderer/package-lock.json'}
    complete = required.issubset(manifest['files'])
    return {'ok': complete and release['engineVersion'] == version and all(record['ok'] for record in records),
            'assets': records, 'engineVersion': version}


def install(target: Path, assets: Path, cache: Path | None = None) -> dict:
    target = target.resolve()
    if target.exists():
        raise FileExistsError(f'install target already exists: {target}')
    if sys.version_info[:2] != (3,14):
        raise ValueError('use Python 3.14 to install this release')
    check = check_assets(assets)
    if not check['ok']:
        raise ValueError(f'assets failed verification: {check}')
    npm = shutil.which('npm.cmd' if os.name == 'nt' else 'npm')
    if not shutil.which('node') or not npm:
        raise ValueError('install Node.js/npm before the world renderer')
    version = check['engineVersion']
    cache = cache or target.parent / '.world-atlas-downloads' / version
    wheel = fetch_asset(assets, cache, 'engine')
    target.mkdir(parents=True)
    environment = target / 'python'
    venv.EnvBuilder(with_pip=True).create(environment)
    interpreter = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    subprocess.run([str(interpreter), '-m', 'pip', 'install', str(wheel)], check=True)
    renderer = target / 'renderer'
    renderer.mkdir()
    for filename in ('package.json','package-lock.json'):
        shutil.copy2(assets / 'renderer' / filename, renderer / filename)
    # Explicit cwd works with npm on Windows; never depend on the old repository.
    subprocess.run([npm, 'ci', '--ignore-scripts', '--no-audit', '--no-fund'], cwd=renderer, check=True)
    entry = renderer / 'node_modules/mapshaper/bin/mapshaper'
    subprocess.run([str(interpreter), '-m', 'world_atlas', 'doctor', '--mapshaper', str(entry)], check=True)
    record = {'python': str(interpreter), 'mapshaper': str(entry), 'engine': f'world-atlas-engine=={version}',
              'command': [str(interpreter), '-m', 'world_atlas'], 'assetsVerified': True,
              'downloadedWheel': str(wheel), 'release': load_release(assets)}
    (target / 'runtime.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', required=True, type=Path)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--cache', type=Path)
    args = parser.parse_args()
    assets = Path(__file__).resolve().parents[1] / 'assets'
    try:
        result = check_assets(assets) if args.check_only else install(args.target, assets, args.cache)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0 if result.get('ok', True) else 1


if __name__ == '__main__':
    raise SystemExit(main())
