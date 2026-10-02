"""Continue a validated society checkpoint through the canonical renderer."""
import argparse
import json
import logging
import os
from pathlib import Path

from world_atlas.checks import semantic_checks
from world_atlas.rebuild import finish_accepted_world
from world_atlas.runtime import require_renderer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mapshaper", type=Path, required=True)
    args = parser.parse_args()
    _, entry = require_renderer(args.mapshaper)
    os.environ["WORLD_ATLAS_MAPSHAPER"] = str(entry)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    finish_accepted_world(args.output)
    checks = semantic_checks(args.output)
    (args.output / "review/release-checks.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
