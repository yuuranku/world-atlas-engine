# Same-world coastline correction, 2026-09-05

Scope: regenerate terrain only from v36/v5's confirmed seed 2116268501 and
unchanged 2176×1088 recipe. Keep the published v5 and its human layers intact.

## Evidence and hypothesis

The original raster already has broad rounded bays: the SVG renderer is not
the primary cause. Basal crust footprints use a sine-axis curve. Fine coastal
detail does not alter that long outline enough.

The player's red-line example on the eastern landmass clarified the scale:
keep the coast's overall course, but interrupt long smooth arcs with irregular
recesses, projections and changes of direction. Do not redraw continents as
giant spikes. The reference is a design constraint, not a prescribed coastline
to hard-code or a raster mask to paste into the generator.

The selected implementation displaces continuous coastal surface coordinates
before the common sea-level cut. Alongshore motion is piecewise affine at
uneven relay zones; the compact cross-shore support moves neighboring relief
with the coast. Existing erosion and ocean-floor calculations follow. Coarse
plate ownership and motion remain the source; drawing is unchanged.

This uses a bounded motion budget, not a hard clip collapsing all fast plates
to one displacement. It is not a calibrated tectonic history, and absolute
present-day plate speed alone does not explain a passive margin's ancestry.

## Rejected experiments

- A: replacing basal crust footprints changed inland relief too much (land
  IoU 0.7954 after seam alignment); rejected for this same-world repair.
- B: late inlet stamps retained the large arcs and produced artificial cuts;
  rejected despite a high land IoU (0.9679).
- C: excessively strong structural motion created repeated large points;
  100 modified sectors had slip 623.6–640 km. Rejected despite greater coarse
  perimeter. The hard-clipping regression and red-line scale guided correction.
- D: reduced offsets reproduced independently, but individual promontories
  remained too large compared with the red-line reference. It remains a
  calibration result, not the selected shoreline.

Outputs remain in validation/coast-v37-a through -d for audit. The independent
D replay is fictional-world-terrain-v37. The selected E result is released as
fictional-world-terrain-v38. Rejected footprint/stamp code was removed. Never
infer realism from longer perimeter alone.

## References consulted

- NPS, Rocky Coast Landforms: https://www.nps.gov/articles/rocky-coast-landforms.htm
  Faults, folds, bedrock resistance and drowned valleys produce different
  coastal profiles; not every shoreline should be equally jagged.
- NPS, Passive Continental Margins:
  https://www.nps.gov/subjects/geology/plate-tectonics-passive-continental-margins.htm
  Rifted crust, subsequent subsidence and sedimentation explain broad plains
  and shelves, even far from a currently active plate boundary.
- NOAA ETOPO: https://www.ncei.noaa.gov/products/etopo-global-relief-model
  Use one continuous land/sea surface, not a disconnected coastal decoration.
- https://github.com/dandrino/terrain-erosion-3-ways
  Reviewed its discussion of drainage-based versus noise-first terrain; use
  network structure for inlets, not more global noise. No code copied.

This remains a procedural approximation, not a calibrated geological history.

## Verification

Selected E uses 250–750 km alongshore half-lengths and 16.58–102.44 km realized
motion budgets across 450 sectors. It preserves the broad continental course;
the illustrated eastern coast now has several smaller changes of direction
rather than a few large points. Some stable/sedimentary arcs remain deliberately
gentle. The user accepted v38 on 2026-09-05 and requested its incorporation in
the engine and skill. It is now the current accepted terrain baseline; this is
not a claim that all seeds or every shoreline have passed geographic review.

Measured at 2176×1088 against the aligned v36 source:

- Identical seed, recipe, planet settings, plate IDs, velocities and boundaries.
- Land area 34.999988%; both display edges remain ocean.
- Land intersection-over-union 0.963486; 1.30175% of global surface changes class.
- Finite elevation/depth; signed surface and sea/land masks agree exactly.
- 19 package/regression tests pass in the independently installed wheel runtime.
- Actual browser overview, four coastal views and maximum zoom inspected;
  resource loading, dragging, contour toggle and fit-to-world checks pass.
- Source field array digest:
  `fb5685278d51fa2b0d3118d7d0492266472552c5b7d8351cb6938c290ef1dace`.
- PNG SHA-256:
  `b3709bf21ec3f4ced0edfcadf75946b22cc33ed57b43503fc8e758082f735b9f`.

Independent full-resolution replay and final browser report are saved alongside
the delivered terrain. The published v5 physical bundle and human maps were
not modified. This approval covers terrain; it does not automatically authorize
regenerating or overwriting the existing human-world maps.
