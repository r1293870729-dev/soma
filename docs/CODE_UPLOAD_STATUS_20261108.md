# Code upload status — 2026-11-08

## Uploaded as individual, browsable source files

The final-B CPU reference and its development entry points are committed:

- `src/soma_b/signal.py` — B/mass implementation, fixed white reference and empirical edge.
- `src/soma_b/replay.py` — offline replay from saved residual arrays.
- `scripts/check_b_contract.py` — main-signal contract preflight.
- `scripts/run_b_development.py` — development-only fixed-topology VAE runner.
- `src/soma_protocol/` — numerical/reference helpers already in the repository.
- `tests/test_final_b.py` — final-B unit tests.
- `configs/signal_b_final.json`, `configs/dev_b_cpu.json` — final policy and development fixture.
- `docs/B_FINAL_CODE_README.md` — scope and run instructions.

## Local source-only bundle prepared, not yet committed to GitHub

The uploaded conversation source files were assembled into `SOMA_CODE_ONLY.zip` (90 entries; 358,298 bytes; SHA-256 `ba7009c8b0e5599a27b3434b560c6584100ff9490fa62f1d8e2e7a51b46b276b`). It contains:

- earlier A/B/C spectral experiments;
- dataset-generalization / Xi scripts;
- unified historical Psi/Xi-B code;
- I1-I4 residual-intervention source;
- earlier active SOMA pilot package and scripts;
- an older HAM prototype source snapshot;
- final-B reference code and configs.

The archive excludes datasets, checkpoints, trained weights, generated plots, and bulky run logs. It is available as a conversation download but has **not** been copied into this repository as a binary archive. The legacy source trees therefore are not yet individually browsable in GitHub.

## Verification

The final-B test file was run locally against the supplied B-final source package: **30 tests passed**. This is unit-test evidence only; it does not mean Q1/Q2/Q3 scientific validation passed.

## Access and license

The repository is private. Classmates need to be invited as collaborators or granted team access before they can clone/download it. No source license was included in the supplied archive; do not make the repository public or redistribute code outside the authorized group until the owner confirms the license and dataset/dependency terms.
