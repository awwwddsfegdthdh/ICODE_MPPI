from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


STATE_CONVENTION_VERSION = "e1_unified_v1"
CONTROL_DEFINITION = "u=[dqL,dqR]"
STATE_FIELDS = ("x", "y", "yaw", "v", "w", "dqL", "dqR")


@dataclass(frozen=True)
class StateConvention:
    version: str = STATE_CONVENTION_VERSION
    state_fields: tuple[str, ...] = STATE_FIELDS
    control_definition: str = CONTROL_DEFINITION
    yaw_reference: str = "body/base yaw"
    drive_sign_values: tuple[float, float] = (-1.0, 1.0)


def canonical_drive_sign(drive_sign: float) -> float:
    s = float(drive_sign)
    if s not in (-1.0, 1.0):
        raise ValueError(f"drive_sign must be +1 or -1, got: {drive_sign}")
    return s


def wrap_to_pi(angle: float | np.ndarray) -> float | np.ndarray:
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def diff_drive_forward(
    u: np.ndarray,
    wheel_radius: float,
    wheel_base: float,
    drive_sign: float,
) -> np.ndarray:
    arr = np.asarray(u, dtype=np.float32)
    if arr.shape[-1] != 2:
        raise ValueError(f"u must end with 2 dims [dqL,dqR], got shape {arr.shape}")
    dq_l = arr[..., 0]
    dq_r = arr[..., 1]
    s = canonical_drive_sign(drive_sign)
    v = s * float(wheel_radius) * 0.5 * (dq_r + dq_l)
    w = float(wheel_radius) * (dq_r - dq_l) / max(float(wheel_base), 1e-6)
    return np.stack([v, w], axis=-1).astype(np.float32)


def diff_drive_inverse(
    v: float | np.ndarray,
    w: float | np.ndarray,
    wheel_radius: float,
    wheel_base: float,
    drive_sign: float,
) -> np.ndarray:
    vv = np.asarray(v, dtype=np.float32)
    ww = np.asarray(w, dtype=np.float32)
    s = canonical_drive_sign(drive_sign)
    inv_r = 1.0 / max(float(wheel_radius), 1e-6)
    v_term = (vv / s) * inv_r
    w_term = 0.5 * float(wheel_base) * ww * inv_r
    dq_r = v_term + w_term
    dq_l = v_term - w_term
    return np.stack([dq_l, dq_r], axis=-1).astype(np.float32)


def world_to_body(vec_xy: np.ndarray, yaw: float | np.ndarray) -> np.ndarray:
    vec = np.asarray(vec_xy, dtype=np.float32)
    yy = np.asarray(yaw, dtype=np.float32)
    c = np.cos(yy)
    s = np.sin(yy)
    x = c * vec[..., 0] + s * vec[..., 1]
    y = -s * vec[..., 0] + c * vec[..., 1]
    return np.stack([x, y], axis=-1).astype(np.float32)


def body_to_world(vec_xy: np.ndarray, yaw: float | np.ndarray) -> np.ndarray:
    vec = np.asarray(vec_xy, dtype=np.float32)
    yy = np.asarray(yaw, dtype=np.float32)
    c = np.cos(yy)
    s = np.sin(yy)
    x = c * vec[..., 0] - s * vec[..., 1]
    y = s * vec[..., 0] + c * vec[..., 1]
    return np.stack([x, y], axis=-1).astype(np.float32)


def assert_state_contract(state: np.ndarray) -> None:
    x = np.asarray(state)
    if x.ndim == 1:
        if x.shape[0] < 7:
            raise ValueError(f"state length must be >=7, got {x.shape[0]}")
        return
    if x.shape[-1] < 7:
        raise ValueError(f"state last dim must be >=7, got {x.shape}")


def _as_scalar_str(meta: Mapping[str, object], key: str) -> str:
    if key not in meta:
        raise KeyError(f"Missing meta key: {key}")
    v = meta[key]
    if isinstance(v, np.ndarray):
        if v.size <= 0:
            raise ValueError(f"Meta key {key} has empty array")
        return str(v.reshape(-1)[0])
    return str(v)


def assert_meta_contract(meta: Mapping[str, object]) -> None:
    required = (
        "meta__state_convention_version",
        "meta__drive_sign",
        "meta__pose_source",
        "meta__heading_source",
        "meta__control_definition",
    )
    for key in required:
        if key not in meta:
            raise KeyError(f"Missing meta key: {key}")

    version = _as_scalar_str(meta, "meta__state_convention_version")
    if version != STATE_CONVENTION_VERSION:
        raise ValueError(
            f"Unsupported state convention version: {version}; expected {STATE_CONVENTION_VERSION}"
        )

    drive_sign_raw = _as_scalar_str(meta, "meta__drive_sign")
    try:
        drive_sign = float(drive_sign_raw)
    except ValueError as exc:
        raise ValueError(f"meta__drive_sign is not numeric: {drive_sign_raw}") from exc
    canonical_drive_sign(drive_sign)

    control_def = _as_scalar_str(meta, "meta__control_definition")
    if control_def != CONTROL_DEFINITION:
        raise ValueError(f"Unexpected control definition: {control_def}")


def read_npz_meta(npz: Mapping[str, np.ndarray]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key in npz.keys():
        if key.startswith("meta__"):
            out[key] = npz[key]
    return out


def convention_as_meta(drive_sign: float, pose_source: str, heading_source: str, yaw_source: str) -> dict[str, np.ndarray]:
    s = canonical_drive_sign(drive_sign)
    return {
        "meta__state_convention_version": np.array([STATE_CONVENTION_VERSION], dtype=object),
        "meta__drive_sign": np.array([s], dtype=np.float32),
        "meta__pose_source": np.array([str(pose_source)], dtype=object),
        "meta__heading_source": np.array([str(heading_source)], dtype=object),
        "meta__yaw_source": np.array([str(yaw_source)], dtype=object),
        "meta__control_definition": np.array([CONTROL_DEFINITION], dtype=object),
    }


def merge_and_validate_meta(meta_list: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if len(meta_list) <= 0:
        raise ValueError("meta_list must be non-empty")
    first = dict(meta_list[0])
    assert_meta_contract(first)
    keys = (
        "meta__state_convention_version",
        "meta__drive_sign",
        "meta__pose_source",
        "meta__heading_source",
        "meta__control_definition",
    )
    ref = {k: _as_scalar_str(first, k) for k in keys}
    for idx, meta in enumerate(meta_list[1:], start=1):
        assert_meta_contract(meta)
        cur = {k: _as_scalar_str(meta, k) for k in keys}
        if cur != ref:
            raise ValueError(
                f"State convention mismatch at input #{idx}: {cur} != {ref}"
            )
    return first
