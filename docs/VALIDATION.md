# 1.0.0 extraction and validation

Historical record for 1.0.0. Current coastline acceptance is documented in `coastline-1.1.0.md`; current download/bootstrap instructions are in the root README.

## Scope

Separate numerical generation from agent judgement. Preserve the original project and published `fictional-world-v5`; do not regenerate the user's current world as a side effect of packaging.

- Installable Python namespace: `world_atlas`; 53 extracted existing modules plus the public API/CLI/runtime checks.
- Required Python libraries: NumPy 2.3.5, ContourPy 1.3.3, Pillow 12.3.0, Shapely 2.1.2.
- Required full-map geometry renderer: Node.js and Mapshaper 0.7.56, with an independent npm lockfile. The older replay ZIP had not included this dependency; this package does.
- No LLM requests. No sys.path mutation or lookup of the old project directory. Optional preview fonts still use platform font candidates; they do not affect simulation.

## Observed integration evidence

1. Built a wheel, installed it (not editable) in a new virtual environment, and executed outside the package source directory.
2. Regenerated the full 2176×1088 terrain from the v36 recipe, not a cached picture. The PNG and entire NPZ bytes equal the accepted originals:
   - PNG `3acf2508e117fb766e6823e958ee7af8c050c429138cb902bd97ff9113111b95`
   - NPZ `dc8ce7baa1ce77ec2b4d471f4e1164be66274b4202cc779fb3258c3828adbc86`
3. Recomputed climate/hydrology and the entire human simulation with the installed package. An original-directory reference in the Mapshaper subprocess cwd was caught at render time, reproduced in a two-polygon test, and removed. Rebuilt/reinstalled the wheel and completed rendering from the verified society checkpoint. This was checkpoint recovery, not an uninterrupted initial CLI run.
4. All three downstream semantic digests equal the published v5:
   - grid `6fe880a72de0b65163a2e7e987f870ef0156bd56b49dc416bc7d061ecdb8168f`
   - society arrays `7598cecfc4288a6195fc60532384d8f587ed2c426c5ac2fc44061e3d0195189a`
   - society document `c4c08d601ba2fc119b2372017e30107f69ec6a14a84b38173fff0f945e56aa3a`
5. The standalone `verify` command passed: no old names/duplicates, no missing province cells, no sea routes across land, no roads across water, exact source coastline and plate cells, no left/right edge land.
6. Actual Edge screenshots and all 11 themes passed browser checks; no page/resource errors. Observed 80 capital labels and 10 holy-city symbols. Tectonic and monsoon views load. Inspected the political overview image.
7. The skill install script installed its bundled wheel and pinned renderer in another new local runtime, and its `doctor` check passed.
8. Original published v5 HTML remains `cbb9924ead9b094f5b2891373a05248f0272db38aefcc88f2d3b629c1beabf1f`.

## Limits retained, not hidden

- v36 is an accepted approximation, not a complete geological dynamics simulation. Two previous terrain aesthetic regression gates are not claimed fixed by packaging.
- The historical painted-artwork fixture suite is not the standalone package's acceptance target; no claim that every old-project test passes.
- Some of the 15 design questions need additional engine work (upstream planet radius, numerical tides, industrial routes, configurable frontier share). The skill must identify these and get a real implementation or an explicit player tradeoff before generation, not silently ignore answers.
- Unit/contract test command: set `WORLD_ATLAS_SKILL` to the installed skill directory, then `python -m unittest discover -s tests -v`. Browser and full terrain/world checks are separate integration evidence, not hidden behind mocks.
- Fresh package + skill contract suite: 15 tests passed, including the 15-question gate, explicit seeded random answers, refusal to overwrite, tampered source rejection, namespace isolation and actual shared-polygon rendering.
- Latest user feedback: coastlines remain too rounded. Read-only source/image inspection identifies the shared smooth coastal-corridor footprint plus noise-based detail, and the uniform display simplifier, as targets for the next terrain iteration. No coastline algorithm was changed during packaging; the current release is not claimed to pass this new aesthetic requirement.

## Skill methodology

The explicitly requested skill-creator shaped a short core SKILL, progressively loaded questionnaire/commands/geography/human-world/acceptance references, generated UI metadata, and validation with quick_validate. Superpowers shaped failing contract tests, reproduction of the packaging bug, and evidence-backed delivery. The absent optional executing-plans reference did not block inline execution.

Packaging structure follows the [PyPA packaging guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/). Geological research links and their usage boundaries are in the companion skill, not runtime network dependencies.
