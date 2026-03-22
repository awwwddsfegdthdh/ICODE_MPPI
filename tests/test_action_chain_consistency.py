from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_SCRIPT = ROOT / "run_icode_mppi_e1_test.py"
VIEWER_SCRIPT = ROOT / "run_icode_mppi_e1_viewer.py"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _assert_mppi_chain_minimal(script_text: str) -> None:
    assert "goal_slowdown" not in script_text
    assert "warmup" not in script_text
    assert "action_ema" not in script_text
    assert "nav_no_reverse" not in script_text
    assert "gap_drive" not in script_text
    assert "startup_no_reverse" not in script_text
    assert "action = np.clip(action, prev_action - args.max_delta_u, prev_action + args.max_delta_u)" in script_text
    assert "action = np.clip(action, env.ctrl_low, env.ctrl_high).astype(np.float32)" in script_text


def test_mppi_action_chain_minimal_in_test_script() -> None:
    _assert_mppi_chain_minimal(_read(TEST_SCRIPT))


def test_mppi_action_chain_minimal_in_viewer_script() -> None:
    _assert_mppi_chain_minimal(_read(VIEWER_SCRIPT))
