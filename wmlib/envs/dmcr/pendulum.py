import collections
import os
import random
import xml.etree.ElementTree as ET

import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base, common
from dm_control.utils import containers, rewards

from .import DMCR_VARY, SUITE_DIR, register

from .generate_visuals import get_assets
from .rng import dmcr_random


def get_model(visual_seed, vary=["camera", "light"]):
    with open(
        os.path.join(SUITE_DIR, os.path.join("assets", "pendulum.xml")), "r"
    ) as f:
        xml = ET.fromstring(f.read())
    if visual_seed != 0:
        with dmcr_random(visual_seed):
            tilt = random.uniform(0.4, 0.6)
            camera_x = random.uniform(0.0, 0.2)
            camera_y = random.uniform(-1.1, -0.97)
            camera_z = random.uniform(1.3, 1.6)

            light_x = random.uniform(-0.5, 0.5)
            light_y = random.uniform(-0.2, 0.2)
            light_z = random.uniform(1.2, 2.8)
        if "light" in vary:
            xml[4][0].attrib["pos"] = f"{light_x} {light_y} {light_z}"
        if "camera" in vary:
            xml[4][2].attrib["pos"] = f"{camera_x} {camera_y} {camera_z}"
            xml[4][2].attrib["xyaxes"] = f"1 0 0 0 {tilt} 1"
    return ET.tostring(xml, encoding="utf8", method="xml")


_DEFAULT_TIME_LIMIT = 20
_ANGLE_BOUND = 8
_COSINE_BOUND = np.cos(np.deg2rad(_ANGLE_BOUND))


@register("pendulum", "swingup")
def swingup(
    time_limit=_DEFAULT_TIME_LIMIT, dynamics_seed=None, visual_seed=None, vary=DMCR_VARY
):
    model = get_model(visual_seed, vary)
    assets, _ = get_assets(visual_seed, vary)
    physics = Physics.from_xml_string(model, assets)
    task = SwingUp(random=dynamics_seed)
    return control.Environment(physics, task, time_limit=time_limit)


class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Pendulum domain."""

    def pole_vertical(self):
        """Returns vertical (z) component of pole frame."""
        return self.named.data.xmat["pole", "zz"]

    def angular_velocity(self):
        """Returns the angular velocity of the pole."""
        return self.named.data.qvel["hinge"].copy()

    def pole_orientation(self):
        """Returns both horizontal and vertical components of pole frame."""
        return self.named.data.xmat["pole", ["zz", "xz"]]


class SwingUp(base.Task):
    """A Pendulum `Task` to swing up and balance the pole."""

    def __init__(self, random=None):
        """Initialize an instance of `Pendulum`.
    Args:
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
        super(SwingUp, self).__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.
    Pole is set to a random angle between [-pi, pi).
    Args:
      physics: An instance of `Physics`.
    """
        physics.named.data.qpos["hinge"] = self.random.uniform(-np.pi, np.pi)
        super(SwingUp, self).initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation.
    Observations are states concatenating pole orientation and angular velocity
    and pixels from fixed camera.
    Args:
      physics: An instance of `physics`, Pendulum physics.
    Returns:
      A `dict` of observation.
    """
        obs = collections.OrderedDict()
        obs["orientation"] = physics.pole_orientation()
        obs["velocity"] = physics.angular_velocity()
        return obs

    def get_reward(self, physics):
        return rewards.tolerance(physics.pole_vertical(), (_COSINE_BOUND, 1))
