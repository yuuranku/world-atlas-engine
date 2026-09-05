"""Check a terrain run against a portable, player-approved baseline record."""
import argparse
import hashlib
import json
from pathlib import Path

import world_atlas
from world_atlas.checks import array_digest


def verify(baseline: Path, output: Path) -> dict:
    expected = json.loads(baseline.read_text(encoding='utf-8'))
    root = output.resolve()
    checks = {}
    for name, digest in expected['files'].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root):
            raise ValueError('baseline artifact escapes output directory')
        checks[name] = path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest
    checks['arrayDigest'] = array_digest(root/'source/physical-fields.npz') == expected['fieldArrayDigest']
    provenance = json.loads((root/'source/provenance.json').read_text(encoding='utf-8'))
    checks['engineVersion'] = provenance['engineVersion'] == expected['engineVersion'] == world_atlas.__version__
    checks['recipe'] = provenance['recipe'] == json.loads((baseline.parent/expected['recipeFile']).read_text(encoding='utf-8'))
    report = {'ok':all(checks.values()), 'terrainId':expected['terrainId'], 'checks':checks,
              'engineModule':world_atlas.__file__, 'output':str(root)}
    (root/'accepted-terrain-check.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    report = verify(args.baseline,args.output)
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if report['ok'] else 1)
