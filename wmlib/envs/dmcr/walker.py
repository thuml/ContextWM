import collections
import os
import random
import xml.etree.ElementTree as ET

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base, common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers, rewards

from .import DMCR_VARY, SUITE_DIR, register
from .rng import dmcr_random

from .generate_visuals import get_assets

_DEFAULT_TIME_LIMIT = 25
_CONTROL_TIMESTEP = 0.025

# Minimal height of torso over foot above which stand reward is 1.
_STAND_HEIGHT = 1.2

# Horizontal speeds (meters/second) above which move reward is 1.
_WALK_SPEED = 1
_RUN_SPEED = 8


def get_model(visual_seed, vary=["camera"]):
    choices = {"camera": "default", "light": "default"}
    with open(os.path.join(SUITE_DIR, os.path.join("assets", "walker.xml")), "r") as f:
        xml = ET.fromstring(f.read())
    if visual_seed != 0:
        with dmcr_random(visual_seed):
            camera_x = random.uniform(-0.3, 0.3)
            camera_y = random.uniform(-2.05, -1.95)
            camera_z = random.uniform(0.6, 0.8)
        camera = f"{camera_x} {camera_y} {camera_z}"
        if "camera" in vary:
            xml[6][1][1].attrib["pos"] = camera
            choices["camera"] = camera
    return ET.tostring(xml, encoding="utf8", method="xml"), choices


@register("walker", "stand")
def stand(
    time_limit=_DEFAULT_TIME_LIMIT, dynamics_seed=None, visual_seed=None, vary=DMCR_VARY
):
    model, local_choices = get_model(visual_seed, vary)
    assets, global_choices = get_assets(visual_seed, vary)
    physics = Physics.from_xml_string(model, assets)
    task = PlanarWalker(move_speed=0, random=dynamics_seed)
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
    )


@register("walker", "walk")
def walk(
    time_limit=_DEFAULT_TIME_LIMIT, dynamics_seed=None, visual_seed=None, vary=DMCR_VARY
):
    model, local_choices = get_model(visual_seed, vary)
    assets, global_choices = get_assets(visual_seed, vary)
    physics = Physics.from_xml_string(model, assets)
    task = PlanarWalker(move_speed=_WALK_SPEED, random=dynamics_seed)
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
    )


@register("walker", "run")
def run(
    time_limit=_DEFAULT_TIME_LIMIT, dynamics_seed=None, visual_seed=None, vary=DMCR_VARY
):
    model, local_choices = get_model(visual_seed, vary)
    assets, global_choices = get_assets(visual_seed, vary)
    physics = Physics.from_xml_string(model, assets)
    task = PlanarWalker(move_speed=_RUN_SPEED, random=dynamics_seed)
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
    )


class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Walker domain."""

    def torso_upright(self):
        """Returns projection from z-axes of torso to the z-axes of world."""
        return self.named.data.xmat["torso", "zz"]

    def torso_height(self):
        """Returns the height of the torso."""
        return self.named.data.xpos["torso", "z"]

    def horizontal_velocity(self):
        """Returns the horizontal velocity of the center-of-mass."""
        return self.named.data.sensordata["torso_subtreelinvel"][0]

    def orientations(self):
        """Returns planar orientations of all bodies."""
        return self.named.data.xmat[1:, ["xx", "xz"]].ravel()


class PlanarWalker(base.Task):
    """A planar walker task."""

    def __init__(self, move_speed, random=None):
        """Initializes an instance of `PlanarWalker`.
    Args:
      move_speed: A float. If this value is zero, reward is given simply for
        standing up. Otherwise this specifies a target horizontal velocity for
        the walking task.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
        self._move_speed = move_speed
        super(PlanarWalker, self).__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.
    In 'standing' mode, use initial orientation and small velocities.
    In 'random' mode, randomize joint angles and let fall to the floor.
    Args:
      physics: An instance of `Physics`.
    """
        randomizers.randomize_limited_and_rotational_joints(physics, self.random)
        super(PlanarWalker, self).initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation of body orientations, height and velocites."""
        obs = collections.OrderedDict()
        obs["orientations"] = physics.orientations()
        obs["height"] = physics.torso_height()
        obs["velocity"] = physics.velocity()
        return obs

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        standing = rewards.tolerance(
            physics.torso_height(),
            bounds=(_STAND_HEIGHT, float("inf")),
            margin=_STAND_HEIGHT / 2,
        )
        upright = (1 + physics.torso_upright()) / 2
        stand_reward = (3 * standing + upright) / 4
        if self._move_speed == 0:
            return stand_reward
        else:
            move_reward = rewards.tolerance(
                physics.horizontal_velocity(),
                bounds=(self._move_speed, float("inf")),
                margin=self._move_speed / 2,
                value_at_margin=0.5,
                sigmoid="linear",
            )
            return stand_reward * (5 * move_reward + 1) / 6
