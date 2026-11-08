# B-final code snapshot

The primary implementation is `src/soma_b/signal.py`, with replay and development entry points under `src/soma_b/` and `scripts/`.

This is a CPU float64 reference implementation and a small fixed-topology development VAE. It is **not** the full production SOMA system and does not constitute formal Q1/Q2/Q3 validation.

Legacy experimental implementations from the handoff archive are not yet copied into the repository as individual source files. The original local source archives are preserved separately; do not confuse old MP/EMA code with the final B signal implementation.

## Run

```bash
python -m pip install numpy torch
PYTHONPATH=src python scripts/check_b_contract.py
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python scripts/run_b_development.py --out runs/B_dev_001
```

The final config intentionally has null numeric/statistical fields and formal testing remains locked. The development config is explicitly development-only.
