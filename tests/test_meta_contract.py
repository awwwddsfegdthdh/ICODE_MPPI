import numpy as np

from state_convention import (
    STATE_CONVENTION_VERSION,
    assert_meta_contract,
    convention_as_meta,
)


def test_meta_contract_ok():
    meta = convention_as_meta(
        drive_sign=-1.0,
        pose_source="odom",
        heading_source="base",
        yaw_source="odom_yaw",
    )
    assert_meta_contract(meta)


def test_meta_contract_fail_on_version():
    meta = convention_as_meta(
        drive_sign=-1.0,
        pose_source="odom",
        heading_source="base",
        yaw_source="odom_yaw",
    )
    meta["meta__state_convention_version"] = np.array([f"{STATE_CONVENTION_VERSION}_bad"], dtype=object)
    failed = False
    try:
        assert_meta_contract(meta)
    except ValueError:
        failed = True
    assert failed
