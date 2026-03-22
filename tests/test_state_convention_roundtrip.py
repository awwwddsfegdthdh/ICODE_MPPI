import numpy as np

from state_convention import canonical_drive_sign, diff_drive_forward, diff_drive_inverse


def test_diff_drive_roundtrip():
    rng = np.random.default_rng(42)
    for drive_sign in (-1.0, 1.0):
        canonical_drive_sign(drive_sign)
        for _ in range(200):
            v = float(rng.uniform(-1.2, 1.2))
            w = float(rng.uniform(-3.0, 3.0))
            u = diff_drive_inverse(v=v, w=w, wheel_radius=0.085, wheel_base=0.37, drive_sign=drive_sign)
            vw = diff_drive_forward(u=u, wheel_radius=0.085, wheel_base=0.37, drive_sign=drive_sign)
            assert np.allclose(vw[0], v, atol=1e-6)
            assert np.allclose(vw[1], w, atol=1e-6)
