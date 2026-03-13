import numpy as np
import os
from env_mujoco import CarEnv

def generate_data():
    print('Generating dataset from MuJoCo...')
    env = CarEnv('car_scene.xml')
    num_trajectories = 100
    horizon = 50
    state_dim = 6
    action_dim = 2

    X = np.zeros((num_trajectories, horizon, state_dim))
    U = np.zeros((num_trajectories, horizon, action_dim))
    X_next = np.zeros((num_trajectories, horizon, state_dim))

    for i in range(num_trajectories):
        state = env.reset()
        for t in range(horizon):
            # 随机油门和方向盘
            action = np.random.uniform(-1, 1, size=(2,)) * np.array([10.0, 5.0])
            next_state = env.step(action)
            
            X[i, t] = state
            U[i, t] = action
            X_next[i, t] = next_state
            
            state = next_state

    # 保存为 numpy 格式
    np.save('mujoco_dataset.npy', {'X': X, 'U': U, 'X_next': X_next})
    print('Dataset saved to mujoco_dataset.npy successfully!')

if __name__ == "__main__":
    generate_data()
