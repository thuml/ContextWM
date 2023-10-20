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
from .rng import dmcr_random

from .generate_visuals import get_assets

_DEFAULT_TIME_LIMIT = 40
_CONTROL_TIMESTEP = 0.04
_JOINTS = [
    "tail1",
    "tail_twist",
    "tail2",
    "finright_roll",
    "finright_pitch",
    "finleft_roll",
    "finleft_pitch",
]


def get_model(visual_seed, vary=["camera", "light"]):
    with open(os.path.join(SUITE_DIR, os.path.join("assets", "fish.xml")), "r") as f:
        xml = ET.fromstring(f.read())
    if visual_seed != 0:
        with dmcr_random(visual_seed):
            camera_x = random.uniform(-0.05, 0.05)
            camera_y = random.uniform(-0.05, 0.05)
            camera_z = random.uniform(0.8, 1.2)

            light_x = random.uniform(-0.3, 0.3)
            light_y = random.uniform(-0.3, 0.3)
            light_z = random.uniform(0.0, 1.0)
        if "camera" in vary:
            xml[5][0].attrib["pos"] = f"{camera_x} {camera_y} {camera_z}"
        if "light" in vary:
            xml[5][6][0].attrib["pos"] = f"{light_x} {light_y} {light_z}"
    return ET.tostring(xml, encoding="utf8", method="xml")


@register("fish", "upright")
def upright(
    time_limit=_DEFAULT_TIME_LIMIT, dynamics_seed=None, visual_seed=None, vary=DMCR_VARY
):
    model = get_model(visual_seed, vary)
    assets, _ = get_assets(visual_seed, vary)
    physics = Physics.from_xml_string(model, assets)
    task = Upright(random=dynamics_seed)
    return control.Environment(
        physics, task, control_timestep=_CONTROL_TIMESTEP, time_limit=time_limit
    )


@register("fish", "swim")
def swim(
    time_limit=_DEFAULT_TIME_LIMIT, dynamics_seed=None, visual_seed=None, vary=DMCR_VARY
):
    model = get_model(visual_seed, vary)
    assets, _ = get_assets(visual_seed, vary)
    physics = Physics.from_xml_string(model, assets)
    task = Swim(random=dynamics_seed)
    return control.Environment(
        physics, task, control_timestep=_CONTROL_TIMESTEP, time_limit=time_limit
    )


class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Fish domain."""

    def upright(self):
        """Returns projection from z-axes of torso to the z-axes of worldbody."""
        return self.named.data.xmat["torso", "zz"]

    def torso_velocity(self):
        """Returns velocities and angular velocities of the torso."""
        return self.data.sensordata

    def joint_velocities(self):
        """Returns the joint velocities."""
        return self.named.data.qvel[_JOINTS]

    def joint_angles(self):
        """Returns the joint positions."""
        return self.named.data.qpos[_JOINTS]

    def mouth_to_target(self):
        """Returns a vector, from mouth to target in local coordinate of mouth."""
        data = self.named.data
        mouth_to_target_global = data.geom_xpos["target"] - data.geom_xpos["mouth"]
        return mouth_to_target_global.dot(data.geom_xmat["mouth"].reshape(3, 3))


class Upright(base.Task):
    """A Fish `Task` for getting the torso upright with smooth reward."""

    def __init__(self, random=None):
        """Initializes an instance of `Upright`.
    Args:
      random: Either an existing `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically.
    """
        super(Upright, self).__init__(random=random)

    def initialize_episode(self, physics):
        """Randomizes the tail and fin angles and the orientation of the Fish."""
        quat = self.random.randn(4)
        physics.named.data.qpos["root"][3:7] = quat / np.linalg.norm(quat)
        for joint in _JOINTS:
            physics.named.data.qpos[joint] = self.random.uniform(-0.2, 0.2)
        # Hide the target. It's irrelevant for this task.
        physics.named.model.geom_rgba["target", 3] = 0
        super(Upright, self).initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation of joint angles, velocities and uprightness."""
        obs = collections.OrderedDict()
        obs["joint_angles"] = physics.joint_angles()
        obs["upright"] = physics.upright()
        obs["velocity"] = physics.velocity()
        return obs

    def get_reward(self, physics):
        """Returns a smooth reward."""
        return rewards.tolerance(physics.upright(), bounds=(1, 1), margin=1)


class Swim(base.Task):
    """A Fish `Task` for swimming with smooth reward."""

    def __init__(self, random=None):
        """Initializes an instance of `Swim`.
    Args:
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
        super(Swim, self).__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""

        quat = self.random.randn(4)
        physics.named.data.qpos["root"][3:7] = quat / np.linalg.norm(quat)
        for joint in _JOINTS:
            physics.named.data.qpos[joint] = self.random.uniform(-0.2, 0.2)
        # Randomize target position.
        physics.named.model.geom_pos["target", "x"] = self.random.uniform(-0.4, 0.4)
        physics.named.model.geom_pos["target", "y"] = self.random.uniform(-0.4, 0.4)
        physics.named.model.geom_pos["target", "z"] = self.random.uniform(0.1, 0.3)
        super(Swim, self).initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation of joints, target direction and velocities."""
        obs = collections.OrderedDict()
        obs["joint_angles"] = physics.joint_angles()
        obs["upright"] = physics.upright()
        obs["target"] = physics.mouth_to_target()
        obs["velocity"] = physics.velocity()
        return obs

    def get_reward(self, physics):
        """Returns a smooth reward."""
        radii = physics.named.model.geom_size[["mouth", "target"], 0].sum()
        in_target = rewards.tolerance(
            np.linalg.norm(physics.mouth_to_target()),
            bounds=(0, radii),
            margin=2 * radii,
        )
        is_upright = 0.5 * (physics.upright() + 1)
        return (7 * in_target + is_upright) / 8
