# World Atlas Engine + Skill Implementation Plan

Historical 1.0.0 extraction plan, retained as an audit record. Use the root README and the GitHub-release plan for current commands.

**Goal:** Separate deterministic world computation from agent-led research, decisions and visual review, preserving the published v5.

**Architecture:** Extract the current Python calculation modules into one installable `world_atlas` package. Expose terrain generation, accepted-terrain downstream generation, replay and verification through one CLI. Keep Node/Mapshaper an explicit pinned rendering dependency. Install `generate-world-atlas` with commands, geological review criteria, naming rules and portable release assets; do not duplicate engine logic in the skill.

**Tech Stack:** Python 3.14; NumPy 2.3.5, ContourPy 1.3.3, Pillow 12.3.0, Shapely 2.1.2; Node + Mapshaper 0.7.56; setuptools wheel; Codex skill-creator.

## Tasks

- [x] Add failing package contract tests: namespaced imports, no sys.path mutation, CLI commands, invalid config rejection, output preservation, explicit renderer dependency.
- [x] Extract only required modules; rewrite package imports mechanically; replace original-directory discovery with package-relative or explicit inputs. Do not modify historical world outputs.
- [x] Implement CLI `doctor`, `seeds`, `terrain`, `world`, `reproduce`, `verify`; export the same API. Fail before long work if required dependencies or input hashes are invalid.
- [x] Build and install a wheel in an independent virtual environment. Compare packaged terrain output and replayed v5 semantic hashes against frozen references.
- [x] Initialize the skill using skill-creator; write core workflow plus command, geology/research, human-world, and visual-acceptance references. Include pinned install assets and frozen accepted inputs.
- [x] Run unit tests, renderer and browser smoke tests, skill validation, archive inventory/hash checks. Record real results and remaining limitations.

## Commands and pass criteria

```text
python -m unittest discover -s tests -v
python -m build
python -m world_atlas doctor --mapshaper <explicit JS entry>
python -m world_atlas terrain --recipe examples/terrain-v36.json --output <new directory>
python -m world_atlas reproduce --inputs examples/accepted-v5 --output <new directory> --mapshaper <explicit JS entry>
python <skill-creator>/scripts/quick_validate.py <installed-skill-directory>
```

Require: successful fresh installation, original terrain array/hash equivalence, downstream semantic hashes unchanged, no forbidden-name matches, all declared resources bundled, no changed v5 index hash. Treat geological beauty as manual review, not a score that automatically authorizes delivery.

## Execution notes

Work inline. The optional executing-plans reference is absent in this skill installation; use the available TDD and verification instructions. No Git repository or agent delegation is required.

Completed evidence and the recovered render-stage packaging error are recorded in `docs/VALIDATION.md`. The added requirement to ask the player 15 questions is in the skill's first-action gate and tested. New feedback about rounded coastlines is diagnosed and recorded as an unfixed terrain issue, not concealed by package acceptance.
