# Shared transport drawing performance

The completed v103 world logged shared roads and crossing facilities from
21:06:22 to 21:18:08 on 2026-10-03: about 11 minutes 46 seconds. A local profile
identified repeated polygon scans in road simplification and curve checks.
The renderer prepared the land geometry, but the actual road checks used a
separate land surface with physical river channels removed. That surface was
not prepared, so every candidate chord scanned its large boundary again.

## Changes

- Prepare the actual road surface and river geometry once for the shared
  network, alongside the existing land preparation. GEOS reuses its indexes
  for all chord and corner checks.
- Check the prepared river predicate before requesting an intersection.
  Disjoint chords need no overlay geometry.
- Attribute shared network pieces with a distance-index query and vectorized
  coverage over prepared source corridors. Attribution still requires full
  coverage and uses the same importance ranking.
- Share one polygon triangulation among bank passages in the same accepted
  route corridor. The triangle ordering, shortest-path costs, exact funnel
  predicates and anchor coordinates are unchanged.
- Include the necessary dev16 bank fix: reserve numerical dry clearance when
  navigating and retain the original deck portals when adding dry approaches.
  Bridge spans and all crossing audits still use the original water channel.

No routes, physical sources, bridge evidence or grade checks are removed.
The longest-valid-chord simplification rule is unchanged. Its worst-case
candidate count remains quadratic; the measured cost was scanning the
physical surface, rather than the candidate count itself. The optimization
therefore avoids changing the resulting road layout to obtain speed.

## Measured local result

Source: `D:/2/fictional-world-v103-complete-dev16/review`. Read the land clip
from the saved `physical-surface.svgz` and the exact channel and prepared
roads from `transport-crossings.json`. The land clip has 185,110 coordinates;
the channel has 220,225 coordinates. Both implementations received the same
exported geometry. Select the six longest prepared roads, with 51, 50, 47,
44, 42 and 42 vertices respectively.

| Operation | Before | After | Ratio |
| --- | ---: | ---: | ---: |
| Simplification and corner curves, six actual roads | 17.979 s | 0.331 s | 54.3× |
| Four passages through the actual 2,354-coordinate bank fixture | 0.378 s | 0.096 s | 3.95× |

The resulting coordinates were exactly equal. Before optimization, 17.564 s
of the first benchmark was spent in polygon `covers`. File loading and
surface extraction are excluded from the comparison. Grade engineering,
bridge resolution and the rest of generation are also outside this local
benchmark, so these ratios do not claim a 54× speedup for the entire world.

The reusable benchmark is `tools/benchmark_transport_geometry.py`:

```powershell
$env:PYTHONPATH = 'D:/2/world-generation-speed/src'
& 'D:/2/world-atlas-v102-dev15-public-install/python/Scripts/python.exe' `
  tools/benchmark_transport_geometry.py `
  D:/2/fictional-world-v103-complete-dev16 `
  D:/2/transport-benchmark `
  --baseline-source D:/2/world-v103-final-dev16/src/world_atlas/core/transport_geometry.py
```

It writes a source hash, actual road identifiers, vertex counts, measured
durations and coordinate equality to `timing.json`. It only reads the saved
world; it does not regenerate it.

## Focused validation

The geometry, polygon navigation, road engineering and transport crossing
tests pass: 50 tests in total. The crossing tests still reject missing
facilities and unlicensed water crossings, and validate confluence banks,
native bank transitions and endpoint access. The new reuse test proves that
different passages share exactly one triangulation and retain the original
paths. No full world was rerun for this change.
