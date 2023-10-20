import gym
import numpy as np


class Carla:
    cnt: int = 0

    def __init__(self,
                 **kwargs) -> None:
        from .carla_api.carla_env import CarlaEnv
        ports = kwargs.get("ports", [2000])
        port = ports[min(len(ports) - 1, Carla.cnt)]
        # set traffic manager port to carla_port + 6001, change if you encounter conflicts
        tm_port = port - 2000 + 8001
        print(f"Creating CarlaEnv carla_port={port}, tm_port={tm_port}")
        weather = kwargs.get('fix_weather', 'ClearNoon')
        if weather == "Changing":
            weather = None

        self._env = CarlaEnv(
            changing_weather_speed=kwargs.get('changing_weather_speed', 0.1),
            rl_image_size=kwargs.get('rl_image_size', 64),
            max_episode_steps=kwargs.get('max_episode_steps', 1000),
            frame_skip=kwargs.get('frame_skip', 1),
            is_other_cars=kwargs.get('is_other_cars', True),
            port=port,
            fix_weather=weather,
            num_cameras=kwargs.get('num_cameras', 1),
            num_other_vehicles=kwargs.get('num_other_vehicles', 80),
            trafficmanager_port=tm_port,
            collision_cost=kwargs.get('collision_cost', 1e-3),
            global_speed_diff=kwargs.get('global_speed_diff', 30.0),
            centering_reward_type=kwargs.get('centering_reward_type', 'div'),
            centering_reward_weight=kwargs.get('centering_reward_weight', 1.0),
            clip_collision_reward=kwargs.get('clip_collision_reward', 0.0),
            steer_coeff=kwargs.get('steer_coeff', 1.0),
            centering_border=kwargs.get('centering_border', 1.0),
            use_branch_lane_cut=kwargs.get('use_branch_lane_cut', False),
        )
        Carla.cnt += 1

    @property
    def obs_space(self):
        shape = self._env.observation_space.shape
        shape = (shape[1], shape[2], shape[0])
        return {
            "image": gym.spaces.Box(0, 255, shape, np.uint8),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            "dist_s": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "collision_cost": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "steer_cost": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "centering_reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
        }

    @property
    def act_space(self):
        return {"action": gym.spaces.Box(self._env.action_space.low.min(),
                                         self._env.action_space.high.max(),
                                         self._env.action_space.shape, np.float32)}

    def step(self, action):
        action = action["action"]
        obs, reward, done, info = self._env.step(action)
        obs = {"image": obs.transpose((1, 2, 0))}
        obs["reward"] = float(reward)
        obs["is_first"] = False
        obs["is_last"] = done
        obs["is_terminal"] = done
        obs["dist_s"] = self._env.dist_s
        obs['collision_cost'] = self._env.collision_reward
        obs['steer_cost'] = self._env.steer_reward
        obs['centering_cost'] = self._env.centering_reward
        return obs

    def reset(self):
        obs = self._env.reset()
        obs = {"image": obs.transpose((1, 2, 0))}
        obs["reward"] = 0.0
        obs["is_first"] = True
        obs["is_last"] = False
        obs["is_terminal"] = False
        obs["dist_s"] = self._env.dist_s
        obs['collision_cost'] = self._env.collision_reward
        obs['steer_cost'] = self._env.steer_reward
        obs['centering_cost'] = self._env.centering_reward
        return obs
