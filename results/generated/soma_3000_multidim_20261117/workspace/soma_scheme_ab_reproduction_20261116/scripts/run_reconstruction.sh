#!/bin/zsh
set -u
cd "${0:A:h:h}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$PWD/.cache/matplotlib"
run_id=reconstructed_B_001
failed=0
for dim in 2 4 8 16 32; do
  for mode in loss_plus_xi loss_only; do
    case_id="B_z${dim}_${mode}"
    .venv/bin/python reconstruction/run_case.py --dim "$dim" --mode "$mode" --run-id "$run_id" > "logs/${run_id}_${case_id}.log" 2>&1
    rc=$?
    if (( rc == 0 )); then
      .venv/bin/python reconstruction/fixed_probe.py --dim "$dim" --mode "$mode" --run-id "$run_id" >> "logs/${run_id}_${case_id}.log" 2>&1
      rc=$?
    fi
    if (( rc != 0 )); then failed=$((failed+1)); fi
    print -r -- "${case_id},${rc}" >> "results/${run_id}/job_ledger.csv"
    .venv/bin/python scripts/reconstruction_status.py
  done
done
print -r -- "$failed" > "results/${run_id}/job_failures.txt"
.venv/bin/python scripts/reconstruction_status.py
if [[ -f scripts/analyze_reconstruction.py ]]; then
  .venv/bin/python scripts/analyze_reconstruction.py > "logs/${run_id}_analysis.log" 2>&1
fi
exit "$failed"
