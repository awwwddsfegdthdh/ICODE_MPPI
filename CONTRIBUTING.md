# Contributing Guide

## Branch and Commit

- Create a feature branch from `exp_adaptive_try`.
- Use focused commits (one intent per commit).
- Suggested commit prefixes: `feat:`, `fix:`, `refactor:`, `docs:`, `chore:`.

## Reproducibility

- When changing navigation behavior, report fixed seeds (at least 6100/6101/6102/6103).
- Include:
  - command line used,
  - key metrics from `metrics.json`,
  - failure-chain evidence (`action_source_step`, `trigger_reason_step`, `active_target_step_xy`).

## File Placement

- Core algorithm code: root-level python modules (`mppi.py`, `mppi_nav_utils.py`, `safety_supervisor.py`, etc.).
- Tests: `tests/`.
- Design and experiment reports: `docs/` (stage outputs in `docs/stage_artifacts/`).
- Utility runners: `scripts/`.

## Before Push

```bash
python -m py_compile mppi.py mppi_nav_utils.py run_icode_mppi_e1_test.py run_icode_mppi_e1_viewer.py safety_supervisor.py
pytest -q
```

If full pytest is heavy, run targeted tests and note scope in your PR/commit message.
