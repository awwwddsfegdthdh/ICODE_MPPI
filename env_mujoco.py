import mujoco
import numpy as np


class CarEnv:
    def __init__(self, xml_path="car_scene.xml"):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        return self.get_state()
        
    def step(self, action):
        # action: [forward_force, turning_torque]
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        return self.get_state()
        
    def get_state(self):
        # 返回状态: [x, y, theta, v_x, v_y, omega]
        qpos = self.data.qpos.copy() # [x, y, theta]
        qvel = self.data.qvel.copy() # [vx, vy, omega]
        
        x, y = qpos[0], qpos[1]
        theta = qpos[2]
        
        vx, vy = qvel[0], qvel[1]
        omega = qvel[2]
        
        return np.array([x, y, theta, vx, vy, omega])


class E1RobotEnv:
    """
    E1 MuJoCo wrapper that exposes:
    - control step with decimation to match model dt (~0.02)
    - 7D state estimate used by ICODE/MPPI:
      [x_odom, y_odom, psi_odom, v_body, wz_body, dqL, dqR]
    """

    def __init__(
        self,
        xml_path: str,
        wheel_radius: float = 0.085,
        wheel_base: float = 0.37,
        yaw_blend_alpha: float = 0.35,
        control_decimation: int = 10,
        pose_source: str = "odom",
        heading_source: str = "camera",
    ):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.wheel_radius = float(wheel_radius)
        self.wheel_base = float(wheel_base)
        self.yaw_blend_alpha = float(yaw_blend_alpha)
        self.control_decimation = max(1, int(control_decimation))
        self.pose_source = str(pose_source).lower()
        if self.pose_source not in ("odom", "gt"):
            raise ValueError(f"pose_source must be 'odom' or 'gt', got: {pose_source}")
        self.heading_source = str(heading_source).lower()
        if self.heading_source not in ("base", "camera"):
            raise ValueError(f"heading_source must be 'base' or 'camera', got: {heading_source}")

        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].copy().astype(np.float32)
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].copy().astype(np.float32)

        obs_body_candidates = ("obs_front", "obs_left", "obs_right", "obs_mid_left", "obs_mid_right")
        self._obstacle_specs = []
        for bname in obs_body_candidates:
            gname = f"{bname}_geom"
            sname = f"{bname}_pos_gt"
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, bname)
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, gname)
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sname)
            if bid >= 0 and gid >= 0 and sid >= 0:
                self._obstacle_specs.append(
                    {
                        "body_name": bname,
                        "body_id": int(bid),
                        "geom_id": int(gid),
                        "sensor_name": sname,
                    }
                )
        if len(self._obstacle_specs) == 0:
            raise KeyError(
                f"No obstacle triplets found in '{xml_path}'. "
                "Expected body/geom/sensor like obs_front, obs_front_geom, obs_front_pos_gt."
            )
        self._obs_geom_ids = [s["geom_id"] for s in self._obstacle_specs]
        self._obs_body_ids = [s["body_id"] for s in self._obstacle_specs]
        self._goal_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal_marker")
        self._sensor_names = {
            "wheel_l_vel": "wheel_l_vel",
            "wheel_r_vel": "wheel_r_vel",
            "imu_gyro": "imu_gyro",
            "base_pos_gt": "base_pos_gt",
            "base_quat_gt": "base_quat_gt",
            "camera_quat_gt": "camera_quat_gt",
            "goal_pos_gt": "goal_pos_gt",
            "touch_front_force": "touch_front_force",
        }
        for spec in self._obstacle_specs:
            sname = spec["sensor_name"]
            self._sensor_names[sname] = sname
        self._sensor_meta = {}
        for k, name in self._sensor_names.items():
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sid < 0:
                raise KeyError(f"Sensor '{name}' not found in model '{xml_path}'.")
            self._sensor_meta[k] = (self.model.sensor_adr[sid], self.model.sensor_dim[sid])

        self._odom_x = 0.0
        self._odom_y = 0.0
        self._odom_psi = 0.0
        self._v_body = 0.0
        self._wz_body = 0.0
        self._dq_l = 0.0
        self._dq_r = 0.0
        self._last_time = 0.0
        self._heading_offset = 0.0
        self._state_est = np.zeros((7,), dtype=np.float32)

    def _sensor(self, key: str) -> np.ndarray:
        adr, dim = self._sensor_meta[key]
        return self.data.sensordata[adr : adr + dim].copy()

    @staticmethod
    def _quat_wxyz_to_yaw(quat_wxyz: np.ndarray) -> float:
        w, x, y, z = quat_wxyz
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(np.arctan2(siny_cosp, cosy_cosp))

    @staticmethod
    def _wrap_to_pi(a: float) -> float:
        return float((a + np.pi) % (2.0 * np.pi) - np.pi)

    def _compute_heading_offset(self) -> float:
        if self.heading_source != "camera":
            return 0.0
        try:
            base_quat = self._sensor("base_quat_gt")
            cam_quat = self._sensor("camera_quat_gt")
        except KeyError:
            return 0.0
        base_yaw = self._quat_wxyz_to_yaw(base_quat)
        cam_yaw = self._quat_wxyz_to_yaw(cam_quat)
        return self._wrap_to_pi(cam_yaw - base_yaw)

    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        return self._reset_common()

    def reset_with_pose(self, x: float, y: float, yaw: float) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0] = float(x)
        self.data.qpos[1] = float(y)
        half = 0.5 * float(yaw)
        quat_wxyz = np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)
        self.data.qpos[3:7] = quat_wxyz
        self.data.qvel[:] = 0.0
        return self._reset_common()

    def _reset_common(self) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)

        base_pos = self._sensor("base_pos_gt")
        base_quat = self._sensor("base_quat_gt")
        self._heading_offset = self._compute_heading_offset()
        self._odom_x = float(base_pos[0])
        self._odom_y = float(base_pos[1])
        base_yaw = self._quat_wxyz_to_yaw(base_quat)
        self._odom_psi = self._wrap_to_pi(base_yaw + self._heading_offset)
        self._v_body = 0.0
        self._wz_body = 0.0
        self._dq_l = 0.0
        self._dq_r = 0.0
        self._last_time = float(self.data.time)
        self._state_est[:] = np.array(
            [self._odom_x, self._odom_y, self._odom_psi, self._v_body, self._wz_body, self._dq_l, self._dq_r],
            dtype=np.float32,
        )
        return self._state_est.copy()

    def _update_state_estimate(self) -> np.ndarray:
        t_now = float(self.data.time)
        dt = max(0.0, t_now - self._last_time)
        self._last_time = t_now

        dq_l = float(self._sensor("wheel_l_vel")[0])
        dq_r = float(self._sensor("wheel_r_vel")[0])
        imu_gyro = self._sensor("imu_gyro")
        imu_wz = float(imu_gyro[2]) if imu_gyro.shape[0] >= 3 else 0.0

        v = self.wheel_radius * 0.5 * (dq_r + dq_l)
        wz_wheel = self.wheel_radius * (dq_r - dq_l) / max(self.wheel_base, 1e-6)
        wz = self.yaw_blend_alpha * wz_wheel + (1.0 - self.yaw_blend_alpha) * imu_wz

        if self.pose_source == "gt":
            base_pos = self._sensor("base_pos_gt")
            base_quat = self._sensor("base_quat_gt")
            self._odom_x = float(base_pos[0])
            self._odom_y = float(base_pos[1])
            base_yaw = self._quat_wxyz_to_yaw(base_quat)
            self._odom_psi = self._wrap_to_pi(base_yaw + self._heading_offset)
        else:
            self._odom_x += v * np.cos(self._odom_psi) * dt
            self._odom_y += v * np.sin(self._odom_psi) * dt
            self._odom_psi = self._wrap_to_pi(self._odom_psi + wz * dt)

        self._v_body = v
        self._wz_body = wz
        self._dq_l = dq_l
        self._dq_r = dq_r

        self._state_est[:] = np.array(
            [self._odom_x, self._odom_y, self._odom_psi, self._v_body, self._wz_body, self._dq_l, self._dq_r],
            dtype=np.float32,
        )
        return self._state_est.copy()

    def step(self, action: np.ndarray) -> np.ndarray:
        act = np.asarray(action, dtype=np.float32)
        act = np.clip(act, self.ctrl_low, self.ctrl_high)
        for _ in range(self.control_decimation):
            self.data.ctrl[:] = act
            mujoco.mj_step(self.model, self.data)
        return self._update_state_estimate()

    def get_state(self) -> np.ndarray:
        return self._state_est.copy()

    def get_goal_xy(self) -> np.ndarray:
        g = self._sensor("goal_pos_gt")
        return g[:2].astype(np.float32)

    def set_goal_xy(self, x: float, y: float) -> bool:
        if self._goal_body_id < 0:
            return False
        self.model.body_pos[self._goal_body_id, 0] = float(x)
        self.model.body_pos[self._goal_body_id, 1] = float(y)
        mujoco.mj_forward(self.model, self.data)
        return True

    def set_obstacles_xy(self, obs_xy: np.ndarray) -> bool:
        arr = np.asarray(obs_xy, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] != len(self._obs_body_ids):
            raise ValueError(
                f"obs_xy must have shape ({len(self._obs_body_ids)}, 2), got {arr.shape}"
            )
        if any(bid < 0 for bid in self._obs_body_ids):
            return False
        for i, bid in enumerate(self._obs_body_ids):
            self.model.body_pos[bid, 0] = float(arr[i, 0])
            self.model.body_pos[bid, 1] = float(arr[i, 1])
        mujoco.mj_forward(self.model, self.data)
        return True

    def set_obstacles_radii(self, obs_radii: np.ndarray) -> bool:
        arr = np.asarray(obs_radii, dtype=np.float32).reshape(-1)
        if arr.shape[0] != len(self._obs_geom_ids):
            raise ValueError(
                f"obs_radii must have shape ({len(self._obs_geom_ids)},), got {arr.shape}"
            )
        if any(gid < 0 for gid in self._obs_geom_ids):
            return False
        for i, gid in enumerate(self._obs_geom_ids):
            r = float(max(1e-3, arr[i]))
            gtype = int(self.model.geom_type[gid])
            if gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                self.model.geom_size[gid, 0] = r
            elif gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                self.model.geom_size[gid, 0] = r
            elif gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
                # Keep circumscribed radius approximately equal to r.
                half = r / np.sqrt(2.0)
                self.model.geom_size[gid, 0] = half
                self.model.geom_size[gid, 1] = half
            else:
                self.model.geom_size[gid, 0] = r
        mujoco.mj_forward(self.model, self.data)
        return True

    def get_obstacles_xyr(self) -> np.ndarray:
        obs = np.stack(
            [self._sensor(spec["sensor_name"]) for spec in self._obstacle_specs],
            axis=0,
        ).astype(np.float32)
        radii = []
        for gid in self._obs_geom_ids:
            gtype = int(self.model.geom_type[gid])
            gsize = self.model.geom_size[gid]
            if gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                r = float(gsize[0])
            elif gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
                r = float(np.sqrt(gsize[0] ** 2 + gsize[1] ** 2))
            elif gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                r = float(gsize[0])
            else:
                r = float(max(gsize[0], gsize[1], gsize[2]))
            radii.append(r)
        return np.concatenate([obs[:, :2], np.asarray(radii, dtype=np.float32)[:, None]], axis=1)

    def get_base_xy_gt(self) -> np.ndarray:
        b = self._sensor("base_pos_gt")
        return b[:2].astype(np.float32)

    def get_touch_force(self) -> float:
        return float(self._sensor("touch_front_force")[0])
