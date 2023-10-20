import collections
import os
import random
import xml.etree.ElementTree as ET

import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base, common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers, rewards

from .import DMCR_VARY, SUITE_DIR, register
from .rng import dmcr_random

from .generate_visuals import get_assets

_DEFAULT_TIME_LIMIT = 20
_BIG_TARGET = 0.05
_SMALL_TARGET = 0.015


def get_model(visual_seed, vary=["camera", "light"]):
    with open(os.path.join(SUITE_DIR, os.path.join("assets", "reacher.xml")), "r") as f:
        xml = ET.fromstring(f.read())
    if visual_seed != 0:
        with dmcr_random(visual_seed):
            camera_x = random.uniform(-0.05, 0.05)
            camera_y = random.uniform(-0.05, 0.05)
            camera_z = random.uniform(0.7, 0.8)

            light_pos_x = random.uniform(-0.2, 0.2)
            light_pos_y = random.uniform(-0.2, 0.2)
            light_pos_z = random.uniform(0.15, 0.75)

        if "camera" in vary:
            xml[5][1].attrib["pos"] = f"{camera_x} {camera_y} {camera_z}"
        if "light" in vary:
            xml[5][0].attrib["pos"] = f"{light_pos_x} {light_pos_y} {light_pos_z}"
    return ET.tostring(xml, encoding="utf8", method="xml")


@register("reacher", "easy")
def easy(
    time_limit=_DEFAULT_TIME_LIMIT, dynamics_seed=None, visual_seed=None, vary=DMCR_VARY
):
    model = get_model(visual_seed, vary)
    assets, _ = get_assets(visual_seed, vary)
    physics = Physics.from_xml_string(model, assets)
    task = Reacher(target_size=_BIG_TARGET, random=dynamics_seed)
    return control.Environment(physics, task, time_limit=time_limit,)


@register("reacher", "hard")
def hard(
    time_limit=_DEFAULT_TIME_LIMIT, dynamics_seed=None, visual_seed=None, vary=DMCR_VARY
):
    model = get_model(visual_seed, vary)
    assets, _ = get_assets(visual_seed, vary)
    physics = Physics.from_xml_string(model, assets)
    task = Reacher(target_size=_SMALL_TARGET, random=dynamics_seed)
    return control.Environment(physics, task, time_limit=time_limit,)


class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Reacher domain."""

    def finger_to_target(self):
        """Returns the vector from target to finger in global coordinates."""
        return (
            self.named.data.geom_xpos["target", :2]
            - self.named.data.geom_xpos["finger", :2]
        )

    def finger_to_target_dist(self):
        """Returns the signed distance between the finger and target surface."""
        return np.linalg.norm(self.finger_to_target())


class Reacher(base.Task):
    """A reacher `Task` to reach the target."""

    def __init__(self, target_size, random=None):
        """Initialize an instance of `Reacher`.
    Args:
      target_size: A `float`, tolerance to determine whether finger reached the
          target.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
        self._target_size = target_size
        super(Reacher, self).__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        physics.named.model.geom_size["target", 0] = self._target_size
        randomizers.randomize_limited_and_rotational_joints(physics, self.random)

        # Randomize target position
        angle = self.random.uniform(0, 2 * np.pi)
        radius = self.random.uniform(0.05, 0.20)
        physics.named.model.geom_pos["target", "x"] = radius * np.sin(angle)
        physics.named.model.geom_pos["target", "y"] = radius * np.cos(angle)

        super(Reacher, self).initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation of the state and the target position."""
        obs = collections.OrderedDict()
        obs["position"] = physics.position()
        obs["to_target"] = physics.finger_to_target()
        obs["velocity"] = physics.velocity()
        return obs

    def get_reward(self, physics):
        radii = physics.named.model.geom_size[["target", "finger"], 0].sum()
        return rewards.tolerance(physics.finger_to_target_dist(), (0, radii))
