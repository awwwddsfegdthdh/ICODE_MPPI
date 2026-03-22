import numpy as np

from convert_multilayer_dataset import integrate_diff_drive_odometry
from state_convention import diff_drive_forward


def test_convert_rollout_matches_kinematic_integration():
    n = 120
    dt = 0.02
    sim_time = np.arange(n, dtype=np.float32) * dt
    raw_episode = np.zeros((n,), dtype=np.int32)

    dq_l = np.linspace(0.2, 1.8, num=n, dtype=np.float32)
    dq_r = np.linspace(0.3, 1.6, num=n, dtype=np.float32)
    wheel_vel = np.stack([dq_l, dq_r], axis=1)
    imu_gyro = np.zeros((n, 3), dtype=np.float32)

    x_odom, y_odom, psi_odom, v_body, wz_body = integrate_diff_drive_odometry(
        raw_episode=raw_episode,
        sim_time=sim_time.astype(np.float64),
        wheel_vel=wheel_vel,
        imu_gyro=imu_gyro,
        wheel_radius=0.085,
        wheel_base=0.37,
        yaw_blend_alpha=1.0,
        drive_sign=-1.0,
        init_pose_by_episode=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
    )

    x = 0.0
    y = 0.0
    yaw = 0.0
    for i in range(1, n):
        vw = diff_drive_forward(
            u=np.array([dq_l[i], dq_r[i]], dtype=np.float32),
            wheel_radius=0.085,
            wheel_base=0.37,
            drive_sign=-1.0,
        )
        v = float(vw[0])
        w = float(vw[1])
        x += v * np.cos(yaw) * dt
        y += v * np.sin(yaw) * dt
        yaw = float((yaw + w * dt + np.pi) % (2.0 * np.pi) - np.pi)

    assert abs(x_odom[-1] - x) < 1e-4
    assert abs(y_odom[-1] - y) < 1e-4
    assert abs(psi_odom[-1] - yaw) < 1e-4
    assert np.isfinite(v_body).all()
    assert np.isfinite(wz_body).all()
