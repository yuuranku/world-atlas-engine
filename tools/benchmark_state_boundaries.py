"""Compare published and accelerated boundary settlement on the SAME real graph."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from benchmark_spatial import reference_function
from world_atlas.core.model import WorldGrid
from world_atlas.core.render import _society_generation_request
from world_atlas.core.society import state_formation
from world_atlas.core.society.model import NameLexicon
from world_atlas.core.society.names import load_name_lexicon
from world_atlas.core.society.politics import derive_politics
from world_atlas.core.society.storage import load_society
from world_atlas.core.thematic import derive_thematic_layers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('world', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    grid = WorldGrid.load(args.world / 'grid')
    society = load_society(args.world / 'society', expected_grid_digest=grid.content_digest())
    source, request = _society_generation_request(grid)
    old = reference_function(Path(__file__).resolve().parents[1], 'v1.1.0', 'state_formation', 'settle_state_boundaries')
    optimized = state_formation.settle_state_boundaries
    records = []

    def compare(graph, *positional, **keywords):
        started = time.perf_counter()
        expected = old(graph, *positional, **keywords)
        before = time.perf_counter() - started
        started = time.perf_counter()
        actual = optimized(graph, *positional, **keywords)
        after = time.perf_counter() - started
        np.testing.assert_array_equal(actual, expected)
        records.append(dict(regions=graph.region_count, beforeSeconds=before, afterSeconds=after,
                            speedup=before/after, identical=True))
        print(json.dumps(records[-1]), flush=True)
        return actual

    state_formation.settle_state_boundaries = compare
    try:
        derive_politics(grid, derive_thematic_layers(grid), society.population, society.settlements,
                       society.cultures, society.transport,
                       source if isinstance(source, NameLexicon) else load_name_lexicon(source),
                       state_count=request['state_count'], frontier_target_share=request['frontier_target_share'])
    finally:
        state_formation.settle_state_boundaries = optimized
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump({'referenceRevision': 'v1.1.0', 'records': records}, stream, indent=2)


if __name__ == '__main__':
    main()
