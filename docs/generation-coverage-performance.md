# Shared coverage dissolves

`shared_display_coverage` now dissolves the faces produced by its common
snap-rounded, polygonized graph with Shapely `coverage_union_all`. Those faces
are disjoint and carry identical shared nodes, so a general intersection
overlay repeats work already performed by noding. Original input geometries
can overlap; their general `union_all` remains unchanged.

The original ownership rule, precision, coverage completeness check, and
coverage topology check remain in place. The final coverage dissolve runs
only after the result passes `coverage_is_valid`, excluding empty category
slots from its geometry input while preserving those slots in the result.

One bounded comparison used the raw population regression fixture and a
128×96 common graph. Both implementations ran once against the same source:

| Workload | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| Actual population noding fixture | 0.001260 s | 0.000594 s | 2.12× |
| Common graph, 15,360 coordinates | 0.312404 s | 0.163352 s | 1.91× |

Both returned valid geometries and valid shared coverages, with empty
per-category symmetric differences, identical coordinate counts, unchanged
source bytes, and zero differences at 1,600 and 12,288 ownership witnesses.
These are local dissolve measurements, not an estimate of whole-world time.

The benchmark also examined the existing world's biome paint clipped to the
physical shore in a 300×300 coastal region. That sample formed a valid
coverage and differed from the authoritative shore by only 9.1e-14 area.
This does not establish an exact shared-edge contract for all independent
floating shoreline overlays. `coastal_partition` and overview theme grouping
retain general unions; no rounding, buffering, or repair was added to make
those inputs qualify for a coverage union.

Validation: `tests.test_cartographic_features`, `tests.test_coastal_partition`,
and `tests.test_continuous_ecology`.

Reproduce the bounded measurement with the installed runtime and:

```powershell
python tools/benchmark_shared_coverage.py --review D:/2/fictional-world-v103-complete-dev16/review --output D:/2/world-speed-evidence/shared-coverage.json
```
