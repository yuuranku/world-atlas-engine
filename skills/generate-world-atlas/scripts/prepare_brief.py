"""Record all 15 answers; random choice requires explicit --random."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def build_brief(answers: str | None, *, randomize: bool, seed: int) -> dict:
    if isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("seed must be unsigned 32-bit")
    if randomize:
        selected = {str(i): 'ABC'[int.from_bytes(hashlib.sha256(f'world-atlas-15-v1:{seed}:{i}'.encode()).digest()[:8], 'big') % 3] for i in range(1, 16)}
    else:
        parts = re.split(r'[\s,，;；]+', (answers or '').strip())
        selected = {}
        for part in parts:
            match = re.fullmatch(r'(1[0-5]|[1-9])([ABCabc])', part)
            if not match or match[1] in selected:
                raise ValueError("provide all 15 answers exactly once, e.g. 1B 2A ... 15C")
            selected[match[1]] = match[2].upper()
        if len(selected) != 15:
            raise ValueError("all 15 answers are required; do not silently fill missing choices")
    return {"schema": "world-atlas-15-v1", "questionnaireSeed": seed,
            "selection": "explicit-random-request" if randomize else "player-answers",
            "answers": {str(i): selected[str(i)] for i in range(1, 16)},
            "status": "awaiting-parameter-confirmation", "effectiveParameters": {}, "unsupportedChoices": [],
            "instruction": "Confirm conflicts and engine support before generation. Empty unsupportedChoices is not a validation result."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--answers')
    group.add_argument('--random', action='store_true')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        record = build_brief(args.answers, randomize=args.random, seed=args.seed)
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(args.output.resolve())


if __name__ == '__main__':
    main()
