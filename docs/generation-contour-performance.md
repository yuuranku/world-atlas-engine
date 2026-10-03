# Continuous terrain and cartographic contour performance

## Scope and contract

World drawing now samples the existing `RefinedTerrainField` once at four
intervals per native cell, discovers display branches with ContourPy, and
solves every sampled-edge crossing against that original continuous field.
The physical simulation remains 2176 × 1088 for the saved benchmark world;
its arrays, native heights, drainage relief, coordinate deformation, shore
solver and river model are unchanged. The global colour faces and major ink
consume the same new display graph. City minor contours use the same sampler
on their existing small, disjoint support patches and share true edge roots.

This is a **cartographic topology contract**. A peak or loop wholly between
quarter-cell observations is not certified. It replaces the former global
binary64 interval proof, rather than claiming identical output geometry.
Vertices still solve the actual model; chords approximate the path between
them. Exact model queries remain available through `terrain_level_curves`
for shoreline reconstruction and physical analysis. They are a separate
concern, not an alternate generation mode or old checkpoint reader.

`physical-cartographic-height-graphs-v2` binds the subdivision, topology and
root contract, physical arrays, refinement coefficients, datum, physical
input identity, current source hashes and numerical library versions. Old
exact graph checkpoints fail the current schema/code binding. No manifest is
rewritten or rebound. Successfully saved heights survive a later failure.
The former spawned-per-height stage has been removed. One immutable sampled
grid serves all pending heights, eliminating repeated field construction,
coarse discovery and twelve large worker copies.

## Representative saved-world measurements

Data: `D:/2/fictional-world-v103-complete-dev16`, terrain seed 3859980484.
Python: the existing dev15 runtime; optimized source from this checkout.
The baseline imports the unchanged HEAD contour modules from the original
checkout. These are local benchmarks, not a second complete world run.

| Operation | Previous | Current | Notes |
| --- | ---: | ---: | --- |
| 16 × 16 coastal valley, four heights, curve extraction | 16.177 s | 0.077 s | 211 × for this region; excludes 3.45 s common field construction |
| Same region including field construction | 19.652 s | 3.532 s | 5.6 × for this deliberately small query |
| 256 × 256 region, 4 × sampling, 1,050,625 values | — | 1.829 s | Continuous physical model evaluations |
| ContourPy discovery of all 22 heights on those values | — | 0.061 s | Excludes true-edge solving and final polygon/tile assembly |

The full-world estimate from area alone is not a measured end-to-end result.
Source construction, true roots, polygon assembly and downstream tiles must
be included in the single complete-render benchmark before claiming a world
completion time.

For the valley at [1600, 440, 1616, 456], every emitted current vertex was
evaluated on the original field. The largest height residual was
`9.87e-11 m`. Densified Hausdorff distance between the cropped old exact and
new display graphs was 0.094–0.135 native cells across the four cuts. The new
graph has fewer vertices (455 versus 2442), and some fine branches outside
or between sampled supports are absent under the explicit display contract.
An SVG and raw JSON/NPZ measurements are retained in
`D:/2/world-generation-speed-benchmarks`.

Before changing the display contract, interval arithmetic alone was also
optimized: shared level queries, batched four-piece drainage distances,
batched Hermite basis bounds and allocation-free interval products reduced
the region's profiled time from 17.75 to 8.77 s while preserving all output
coordinates bit for bit. That improvement was insufficient for the requested
world-generation budget, so it remains useful only for exact physical queries.

## Focused verification

Tests cover original-field root residuals, true refinement versus base PCHIP,
closed native-centred peaks, plateaus/empty cuts, identical connections on
adjacent patches, the explicit unobserved-peak limit, strict checkpoint/source
binding and corruption rejection, city minor contours, and immutable samples.
The middle unit-spaced four-node PCHIP specialization also matches SciPy's
binary64 coefficients across random scales, slope switches and plateaus;
existing shared-cache and worker-payload tests are retained.

```powershell
$env:PYTHONPATH='D:/2/world-generation-speed/src'
& D:/2/world-atlas-v102-dev15-public-install/python/Scripts/python.exe -m unittest tests.test_cartographic_contours tests.test_physical_contour_stage tests.test_physical_relief tests.test_city_relief tests.test_continuous_pchip tests.test_continuous_terrain tests.test_continuous_scalar
& D:/2/world-atlas-v102-dev15-public-install/python/Scripts/python.exe scripts/benchmark_contours.py --world D:/2/fictional-world-v103-complete-dev16 --bounds 1600 440 1616 456 --cartographic --output D:/2/world-generation-speed-benchmarks/contour-default-wall.json
```

The benchmark scripts do not write to the delivered world or regenerate its
physical simulation. Do not repeat a full render simply to reproduce these
small measurements.
