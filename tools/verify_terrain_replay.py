"""Verify independently generated terrain, ignoring absolute output paths."""
import argparse
import hashlib
import json
from pathlib import Path

from world_atlas.checks import array_digest


def verify(first: Path, replay: Path) -> dict:
    fingerprints = {}
    for name in ('source/physical-fields.npz', 'review/terrain.png', 'review/contours.svg'):
        hashes = [hashlib.sha256((root/name).read_bytes()).hexdigest() for root in (first,replay)]
        fingerprints[name] = {'first':hashes[0], 'replay':hashes[1], 'equal':hashes[0] == hashes[1]}
    digests = [array_digest(root/'source/physical-fields.npz') for root in (first,replay)]
    report = {'ok':all(value['equal'] for value in fingerprints.values()) and digests[0] == digests[1],
              'first':str(first.resolve()), 'replay':str(replay.resolve()),
              'files':fingerprints, 'fieldArrayDigests':digests}
    (replay/'terrain-replay-check.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first',type=Path)
    parser.add_argument('replay',type=Path)
    args = parser.parse_args()
    result = verify(args.first,args.replay)
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['ok'] else 1)
