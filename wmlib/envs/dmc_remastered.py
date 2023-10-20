import gym
import numpy as np


class DMCRemastered:

    def __init__(self, name, action_repeat=1, size=(64, 64), camera=None, vary=None):
        domain, task = name.split("_", 1)
        if domain == "cup":  # Only domain with multiple words.
            domain = "ball_in_cup"
        self._action_repeat = action_repeat
        self._size = size
        if camera in (-1, None):
            camera = dict(
                quadruped_walk=2,
                quadruped_run=2,
                quadruped_escape=2,
                quadruped_fetch=2,
                pentaped_walk=2,
                pentaped_run=2,
                pentaped_escape=2,
                pentaped_fetch=2,
                biped_walk=2,
                biped_run=2,
                biped_escape=2,
                biped_fetch=2,
                triped_walk=2,
                triped_run=2,
                triped_escape=2,
                triped_fetch=2,
                hexaped_walk=2,
                hexaped_run=2,
                hexaped_escape=2,
                hexaped_fetch=2,
                locom_rodent_maze_forage=1,
                locom_rodent_two_touch=1,
            ).get(name, 0)
        self._camera = camera

        self._vary = vary
        if self._vary == "all":
            self._vary = ["bg", "floor", "body", "target", "reflectance", "camera", "light"]

        from .dmcr.wrapper import make
        from .dmcr.benchmarks import uniform_seed_generator
        self._env = make(domain_name=domain, task_name=task,
                         visual_seed=uniform_seed_generator(1, 1_000_000),
                         dynamics_seed=uniform_seed_generator(1, 1_000_00),
                         height=size[0], width=size[1], frame_skip=action_repeat,
                         camera_id=camera, channels_last=True, vary=self._vary)

        self._ignored_keys = []
        for key, value in self._env.observation_spec().items():
            if value.shape == (0,):
                print(f"Ignoring empty observation key '{key}'.")
                self._ignored_keys.append(key)

    @property
    def obs_space(self):
        spaces = {
            "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
        }
        for key, value in self._env.observation_spec().items():
            if key in self._ignored_keys:
                continue
            if value.dtype == np.float64:
                spaces[key] = gym.spaces.Box(-np.inf, np.inf, value.shape, np.float32)
            elif value.dtype == np.uint8:
                spaces[key] = gym.spaces.Box(0, 255, value.shape, np.uint8)
            else:
                raise NotImplementedError(value.dtype)
        return spaces

    @property
    def act_space(self):
        spec = self._env.action_spec()
        action = gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)
        return {"action": action}

    def step(self, action):
        assert np.isfinite(action["action"]).all(), action["action"]
        # action repeat is handled by the environment.
        observation, reward, done, info = self._env.step(action["action"])
        assert self._env.raw_obs.discount in (0, 1)
        obs = {
            "reward": reward,
            "is_first": False,
            "is_last": done,
            "is_terminal": self._env.raw_obs.discount == 0,
            "image": observation,
            # "output_image": self._env.physics.render(*(256, 256), camera_id=self._camera),
        }
        obs.update(
            {
                k: v
                for k, v in dict(self._env.raw_obs.observation).items()
                if k not in self._ignored_keys
            }
        )
        return obs

    def reset(self):
        observation = self._env.reset()
        obs = {
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "image": observation,
            # "output_image": self._env.physics.render(*(256, 256), camera_id=self._camera),
        }
        obs.update(
            {
                k: v
                for k, v in dict(self._env.raw_obs.observation).items()
                if k not in self._ignored_keys
            }
        )
        return obs
