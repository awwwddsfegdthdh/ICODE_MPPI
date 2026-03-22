import numpy as np

from local_occupancy_map import LocalOccupancyMap
from obs_geometry import LidarRayModel


def test_lidar_map_produces_obstacles():
    m = LocalOccupancyMap(x_range=(-1.0, 4.0), y_range=(-2.0, 2.0), resolution=0.05)
    model = LidarRayModel()
    rays = model.build_rays(
        ranges=np.array([1.2, 0.9, 1.1], dtype=np.float32),
        default_far=6.0,
        min_range=0.12,
        max_range=3.0,
    )
    m.update_from_rays(
        base_xy=np.array([0.0, 0.0], dtype=np.float32),
        base_yaw=0.0,
        rays=rays,
        min_range=0.12,
        max_range=3.0,
    )
    obs = m.extract_obstacles_as_circles(threshold=0.0, min_cluster_cells=1, max_obstacles=32)
    assert obs.shape[0] > 0
