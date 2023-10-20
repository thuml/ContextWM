import collections
import os
import random
import xml.etree.ElementTree as ET

import numpy as np
from dm_control import mujoco, suite
from dm_control.rl import control
from dm_control.suite import base, common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers, rewards

from .import DMCR_VARY, SUITE_DIR, register
from .rng import dmcr_random

from .generate_visuals import get_assets


def get_model(visual_seed, vary=["camera"]):
    with open(os.path.join(SUITE_DIR, os.path.join("assets", "hopper.xml")), "r") as f:
        hopper_xml = ET.fromstring(f.read())
    if visual_seed != 0:
        with dmcr_random(visual_seed):
            camera_x = random.uniform(-0.15, 0.15)
            camera_y = random.uniform(-3.0, -2.6)
            camera_z = random.uniform(0.6, 1.0)
            euler_x = random.uniform(85, 95)
            euler_y = random.uniform(-5, 5)
            euler_z = random.uniform(-5, 5)
        if "camera" in vary:
            hopper_xml[6][0].attrib["euler"] = f"{euler_x} {euler_y} {euler_z}"
            hopper_xml[6][0].attrib["pos"] = f"{camera_x} {camera_y} {camera_z}"
    return ET.tostring(hopper_xml, encoding="utf8", method="xml")


_CONTROL_TIMESTEP = 0.02  # (Seconds)

# Default duration of an episode, in seconds.
_DEFAULT_TIME_LIMIT = 20

# Minimal height of torso over foot above which stand reward is 1.
_STAND_HEIGHT = 0.6

# Hopping speed above which hop reward is 1.
_HOP_SPEED = 2


@register("hopper", "hop")
def hop(
    time_limit=_DEFAULT_TIME_LIMIT, dynamics_seed=None, visual_seed=None, vary=DMCR_VARY
):
    model = get_model(visual_seed, vary)
    assets, _ = get_assets(visual_seed, vary)
    physics = Physics.from_xml_string(model, assets)
    task = Hopper(hopping=True, random=dynamics_seed)
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
    )


@register("hopper", "stand")
def stand(
    time_limit=_DEFAULT_TIME_LIMIT, dynamics_seed=None, visual_seed=None, vary=DMCR_VARY
):
    model = get_model(visual_seed, vary)
    assets, _ = get_assets(visual_seed, vary)
    physics = Physics.from_xml_string(model, assets)
    task = Hopper(hopping=False, random=dynamics_seed)
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
    )


class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Hopper domain."""

    def height(self):
        """Returns height of torso with respect to foot."""
        return self.named.data.xipos["torso", "z"] - self.named.data.xipos["foot", "z"]

    def speed(self):
        """Returns horizontal speed of the Hopper."""
        return self.named.data.sensordata["torso_subtreelinvel"][0]

    def touch(self):
        """Returns the signals from two foot touch sensors."""
        return np.log1p(self.named.data.sensordata[["touch_toe", "touch_heel"]])


class Hopper(base.Task):
    """A Hopper's `Task` to train a standing and a jumping Hopper."""

    def __init__(self, hopping, random=None):
        """Initialize an instance of `Hopper`.
    Args:
      hopping: Boolean, if True the task is to hop forwards, otherwise it is to
        balance upright.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
        self._hopping = hopping
        super(Hopper, self).__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        randomizers.randomize_limited_and_rotational_joints(physics, self.random)
        self._timeout_progress = 0
        super(Hopper, self).initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation of positions, velocities and touch sensors."""
        obs = collections.OrderedDict()
        # Ignores horizontal position to maintain translational invariance:
        obs["position"] = physics.data.qpos[1:].copy()
        obs["velocity"] = physics.velocity()
        obs["touch"] = physics.touch()
        return obs

    def get_reward(self, physics):
        """Returns a reward applicable to the performed task."""
        standing = rewards.tolerance(physics.height(), (_STAND_HEIGHT, 2))
        if self._hopping:
            hopping = rewards.tolerance(
                physics.speed(),
                bounds=(_HOP_SPEED, float("inf")),
                margin=_HOP_SPEED / 2,
                value_at_margin=0.5,
                sigmoid="linear",
            )
            return standing * hopping
        else:
            small_control = rewards.tolerance(
                physics.control(), margin=1, value_at_margin=0, sigmoid="quadratic"
            ).mean()
            small_control = (small_control + 4) / 5
            return standing * small_control
