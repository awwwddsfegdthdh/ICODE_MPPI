from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_SCRIPT = ROOT / "run_icode_mppi_e1_test.py"
VIEWER_SCRIPT = ROOT / "run_icode_mppi_e1_viewer.py"

REQUIRED_COST_ARGS = [
    "--cost-task-goal",
    "--cost-task-final",
    "--cost-task-progress",
    "--cost-task-heading",
    "--cost-task-reverse",
    "--cost-task-path-track",
    "--cost-task-path-terminal",
    "--cost-task-path-progress",
    "--cost-task-lateral",
    "--cost-task-goal-visibility",
    "--cost-safe-collision",
    "--cost-safe-near",
    "--cost-safe-bounds",
    "--cost-safe-bounds-terminal",
    "--cost-ctrl-effort",
    "--cost-ctrl-smooth",
    "--cost-ctrl-spin",
    "--cost-ctrl-wheel-diff",
    "--cost-terminal-progress-deficit",
    "--cost-terminal-near-goal-stall",
    "--cost-terminal-stop",
    "--cost-terminal-overshoot",
    "--action-post-delta-eps",
]

REMOVED_ARGS = [
    "--w-away-goal",
    "--w-path-backtrack",
    "--w-goal-motion-away",
    "--w-reverse-away",
    "--warmup-steps",
    "--goal-slowdown-radius",
    "--action-ema-alpha",
    "--nav-no-reverse",
    "--gap-drive-enable",
    "--startup-no-reverse-steps",
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_cost_cli_contract_test_and_viewer_are_aligned() -> None:
    test_txt = _read(TEST_SCRIPT)
    viewer_txt = _read(VIEWER_SCRIPT)

    for arg in REQUIRED_COST_ARGS:
        assert arg in test_txt
        assert arg in viewer_txt

    for old_arg in REMOVED_ARGS:
        assert old_arg not in test_txt
        assert old_arg not in viewer_txt


def test_stage4_observability_fields_exist() -> None:
    test_txt = _read(TEST_SCRIPT)
    viewer_txt = _read(VIEWER_SCRIPT)

    assert '"action_post_delta_ratio"' in test_txt
    assert '"cost_group_mean"' in test_txt
    assert "post_delta_ratio=" in viewer_txt
    assert "cost_group_mean=" in viewer_txt
