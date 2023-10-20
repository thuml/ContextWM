import collections
import random

import gym
import numpy as np
from dm_env import specs
from gym import Wrapper, core, spaces

from .import DMCR_VARY


class FrameStack(Wrapper):
    def __init__(self, env, num_stack):
        super().__init__(env)
        self._k = num_stack
        self._frames = collections.deque([], maxlen=num_stack)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * num_stack,) + shp[1:]),
            dtype=env.observation_space.dtype,
        )

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)


def _spec_to_box(spec):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = np.int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0)
    high = np.concatenate(maxs, axis=0)
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=np.float32)


def _flatten_obs(obs):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)


class DMC_Remastered_Env(core.Env):
    """
    A gym-like wrapper for DeepMind Control, that uses a list
    of visual and dynamics seeds to create a new env each reset

    source: https://github.com/denisyarats/dmc2gym
    """
    def __init__(
        self,
        task_builder,
        visual_seed_generator,
        dynamics_seed_generator,
        height=84,
        width=84,
        camera_id=0,
        frame_skip=1,
        environment_kwargs=None,
        channels_first=True,
        vary=DMCR_VARY,
    ):
        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._frame_skip = frame_skip
        self._channels_first = channels_first

        self._dynamics_seed_gen = dynamics_seed_generator
        self._visual_seed_gen = visual_seed_generator
        self._task_builder = task_builder

        self._env = self._task_builder(dynamics_seed=0, visual_seed=0, vary=vary)
        self._vary = vary

        # true and normalized action spaces
        self._true_action_space = _spec_to_box([self._env.action_spec()])
        self._norm_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=self._true_action_space.shape, dtype=np.float32
        )
        # create observation space
        shape = [3, height, width] if channels_first else [height, width, 3]
        self._observation_space = spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8
        )

        self._state_space = _spec_to_box(self._env.observation_spec().values())

        self.current_state = None
        self.raw_obs = None

        self.make_new_env()

    def make_new_env(self):
        dynamics_seed = self._dynamics_seed_gen()
        visual_seed = self._visual_seed_gen()
        self._env = self._task_builder(
            dynamics_seed=dynamics_seed, visual_seed=visual_seed, vary=self._vary,
        )
        self.seed(seed=dynamics_seed)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_obs(self, time_step):
        obs = self.render(
            height=self._height, width=self._width, camera_id=self._camera_id
        )
        if self._channels_first:
            obs = obs.transpose(2, 0, 1).copy()
        return obs

    def _convert_action(self, action):
        action = action.astype(np.float64)
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self._norm_action_space.high - self._norm_action_space.low
        action = (action - self._norm_action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        action = action.astype(np.float32)
        return action

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def state_space(self):
        return self._state_space

    @property
    def action_space(self):
        return self._norm_action_space

    def seed(self, seed):
        self._true_action_space.seed(seed)
        self._norm_action_space.seed(seed)
        self._observation_space.seed(seed)

    def step(self, action):
        assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        assert self._true_action_space.contains(action)
        reward = 0
        extra = {"internal_state": self._env.physics.get_state().copy()}

        for _ in range(self._frame_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            done = time_step.last()
            if done:
                break
        self.raw_obs = time_step
        obs = self._get_obs(time_step)
        self.current_state = _flatten_obs(time_step.observation)
        extra["discount"] = time_step.discount
        return obs, reward, done, extra

    def reset(self):
        self.make_new_env()
        time_step = self._env.reset()
        self.raw_obs = time_step
        self.current_state = _flatten_obs(time_step.observation)
        obs = self._get_obs(time_step)
        return obs

    def render(self, mode="rgb_array", height=None, width=None, camera_id=0):
        assert mode == "rgb_array", "only support rgb_array mode, given %s" % mode
        height = height or self._height
        width = width or self._width
        camera_id = camera_id or self._camera_id
        return self._env.physics.render(height=height, width=width, camera_id=camera_id)


from .import ALL_ENVS


def make(
    domain_name,
    task_name,
    visual_seed=None,
    dynamics_seed=None,
    frame_stack=3,
    height=84,
    width=84,
    camera_id=0,
    frame_skip=1,
    channels_last=False,
    vary=DMCR_VARY,
    **_,
):
    env = DMC_Remastered_Env(
        ALL_ENVS[domain_name][task_name],
        height=height,
        width=width,
        visual_seed_generator=visual_seed,
        dynamics_seed_generator=dynamics_seed,
        camera_id=camera_id,
        frame_skip=frame_skip,
        channels_first=not channels_last,
        vary=vary,
    )
    return env
