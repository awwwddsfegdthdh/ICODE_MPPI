import numpy as np

from depth_consistency_gate import DepthConsistencyGate


def test_depth_gate_blocks_inconsistent_depth():
    gate = DepthConsistencyGate(window=6, max_sector_delta_mean=0.35, max_front_delta=0.45)
    lidar = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    # Highly inconsistent near-field depth should be rejected.
    d_bad = np.full((120, 160), 0.12, dtype=np.float32)
    rep_bad = gate.update(d_bad, lidar_triplet=lidar, default_far=6.0, min_valid_depth=0.10)
    assert rep_bad.allow_depth is False


def test_depth_gate_allows_consistent_depth():
    gate = DepthConsistencyGate(window=6, max_sector_delta_mean=0.80, max_front_delta=1.10)
    lidar = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    d_ok = np.full((120, 160), 1.05, dtype=np.float32)
    rep_ok = gate.update(d_ok, lidar_triplet=lidar, default_far=6.0, min_valid_depth=0.10)
    assert rep_ok.allow_depth is True
