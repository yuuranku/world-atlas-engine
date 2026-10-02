"""Seal the versioned engine and lightweight download skill."""
import argparse
import hashlib
import json
from pathlib import Path
import tomllib
import zipfile

REPOSITORY = 'yuuranku/world-atlas-engine'


def write_zip(destination: Path, entries: dict[str, Path]) -> None:
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(entries.items()):
            info = zipfile.ZipInfo(name, (2026, 9, 5, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError(f'archive integrity error: {destination}')


def fingerprint(path):
    return {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def content_files(folder):
    return [p for p in folder.rglob('*') if p.is_file()
            and not {'__pycache__', 'node_modules'}.intersection(p.parts)]


def seal(root: Path) -> dict:
    root = root.resolve()
    skill = root / 'skills/generate-world-atlas'
    assets = skill / 'assets'
    dist = root / 'dist'
    version = tomllib.loads((root/'pyproject.toml').read_text(encoding='utf-8'))['project']['version']
    wheel = dist / f'world_atlas_engine-{version}-py3-none-any.whl'
    if not wheel.is_file() or not (skill / 'SKILL.md').is_file():
        raise ValueError('build the engine wheel first')
    for filename in ('terrain.json', 'world-settings.json', 'terrain-v38.json', 'terrain-v38.acceptance.json', 'terrain-v39.json'):
        (assets/filename).write_text((root/'examples'/filename).read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
    (assets/'renderer').mkdir(exist_ok=True)
    for filename in ('package.json', 'package-lock.json'):
        (assets/'renderer'/filename).write_text((root/'runtime'/filename).read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
    # No circular source/skill ZIP hash in the download descriptor.
    release = {'repository': REPOSITORY, 'engineVersion': version, 'assets': {
        'engine': {'name': wheel.name, **fingerprint(wheel)}}}
    (assets/'release.json').write_text(json.dumps(release, indent=2)+'\n', encoding='utf-8', newline='\n')
    records = {p.relative_to(assets).as_posix(): fingerprint(p)['sha256']
               for p in sorted(content_files(assets)) if p.name != 'asset-manifest.json'}
    if any(name.endswith(('.zip', '.whl')) for name in records):
        raise ValueError('lightweight skill assets must not contain bundled packages')
    (assets/'asset-manifest.json').write_text(json.dumps({'engineVersion': version, 'files': records}, indent=2)+'\n', encoding='utf-8', newline='\n')
    skill_zip = dist / 'generate-world-atlas-skill.zip'
    write_zip(skill_zip, {'generate-world-atlas/'+p.relative_to(skill).as_posix(): p for p in content_files(skill)})
    entries = {}
    for folder in ('src/world_atlas', 'tests', 'scripts', 'tools', 'docs', 'skills', '.github'):
        for path in content_files(root/folder):
            entries[path.relative_to(root).as_posix()] = path
    for name in ('pyproject.toml', 'MANIFEST.in', 'README.md', '.gitignore', '.gitattributes',
                 'runtime/package.json', 'runtime/package-lock.json',
                 'examples/terrain.json', 'examples/world-settings.json',
                 'examples/terrain-v38.json', 'examples/terrain-v38.acceptance.json',
                 'examples/terrain-v39.json'):
        entries[name] = root/name
    source_zip = dist / f'world-atlas-engine-{version}-source.zip'
    write_zip(source_zip, entries)
    report = {'repository': REPOSITORY, 'version': version,
              'files': {p.name: fingerprint(p) for p in (wheel, source_zip, skill_zip)}}
    (dist/'release-manifest.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8', newline='\n')
    return report


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(seal(root), indent=2))
