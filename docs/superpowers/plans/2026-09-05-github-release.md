# GitHub release implementation plan

Goal: publish the engine, a lightweight companion skill, and verified download assets to the user-approved public repository `yuuranku/world-atlas-engine`.

Architecture: `src/world_atlas` owns numerical computation. `skills/generate-world-atlas` owns the questionnaire, judgement and bootstrap. Versioned GitHub Releases own large binary artifacts. The installed skill is a deployed copy, not a second source of truth.

Implementation and verification (inline, no delegation):

1. Add download tests in `tests/test_download.py`: pinned HTTPS URL; valid cache reuse without network; bad hash/truncated downloads never promoted; corrupt cache preserved with an error; unsafe manifest paths rejected. Run them before implementation.
2. Add `scripts/download_engine.py`; update `scripts/install_engine.py` to use it. Use the standard library, temporary files, SHA-256 and explicit fresh runtime targets. Keep Python/npm installations local. Keep new-world questions ahead of bootstrap.
3. Update `tools/package_release.py` to seal wheel/input checksums into the skill, then build a lightweight skill ZIP and source ZIP without environment files or prior worlds. Add root README and technical-route documentation.
4. Run `python -m unittest discover -s tests -v`, skill validation and an isolated install; audit the Git file list before the initial push. Publish tag v1.1.0 and all five manifest-listed release files.
5. Download the actual public skill ZIP without credentials into a clean directory; use its downloader to fetch the published wheel and optional inputs. Install and run doctor plus the package tests. Record evidence and verify remote asset hashes.

No terrain algorithm or accepted map changes. No invented open-source license or credentials in the skill. Subsequent published releases must use new immutable version tags; no silent latest-version executable downloads.
