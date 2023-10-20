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

from .generate_visuals import get_assets
from .rng import dmcr_random


def get_model(visual_seed, vary=["camera", "light"]):
    with open(
        os.path.join(SUITE_DIR, os.path.join("assets", "humanoid.xml")), "r"
    ) as f:
        xml = ET.fromstring(f.read())
    if visual_seed != 0:
        with dmcr_random(visual_seed):
            camera_x = random.uniform(-3.5, -2.5)
            camera_y = random.uniform(-0.5, 0.5)
            camera_z = random.uniform(0.5, 1.5)

            light_x = random.uniform(-0.7, 0.7)
            light_y = random.uniform(-0.7, 0.7)
            light_z = random.uniform(1.2, 1.8)
        if "camera" in vary:
            xml[6][1][1].attrib["pos"] = f"{camera_x} {camera_y} {camera_z}"
        if "light" in vary:
            xml[6][1][0].attrib["pos"] = f"{light_x} {light_y} {light_z}"
    return ET.tostring(xml, encoding="utf8", method="xml")


_DEFAULT_TIME_LIMIT = 25
_CONTROL_TIMESTEP = 0.025
# Height of head above which stand reward is 1.
_STAND_HEIGHT = 1.4
# Horizontal speeds above which move reward is 1.
_WALK_SPEED = 1
_RUN_SPEED = 10


@register("humanoid", "stand")
def stand(
    time_limit=_DEFAULT_TIME_LIMIT, dynamics_seed=None, visual_seed=None, vary=DMCR_VARY
):
    model = get_model(visual_seed, vary)
    assets, _ = get_assets(visual_seed, vary)
    physics = Physics.from_xml_string(model, assets)
    task = Humanoid(move_speed=0, pure_state=False, random=dynamics_seed)
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
    )


@register("humanoid", "walk")
def walk(
    time_limit=_DEFAULT_TIME_LIMIT, dynamics_seed=None, visual_seed=None, vary=DMCR_VARY
):
    model = get_model(visual_seed, vary)
    assets, _ = get_assets(visual_seed, vary)
    physics = Physics.from_xml_string(model, assets)
    task = Humanoid(move_speed=_WALK_SPEED, pure_state=False, random=dynamics_seed)
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
    )


@register("humanoid", "run")
def run(
    time_limit=_DEFAULT_TIME_LIMIT, dynamics_seed=None, visual_seed=None, vary=DMCR_VARY
):
    model = get_model(visual_seed, vary)
    assets, _ = get_assets(visual_seed, vary)
    physics = Physics.from_xml_string(model, assets)
    task = Humanoid(move_speed=_RUN_SPEED, pure_state=False, random=dynamics_seed)
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
    )


class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Walker domain."""

    def torso_upright(self):
        """Returns projection from z-axes of torso to the z-axes of world."""
        return self.named.data.xmat["torso", "zz"]

    def head_height(self):
        """Returns the height of the torso."""
        return self.named.data.xpos["head", "z"]

    def center_of_mass_position(self):
        """Returns position of the center-of-mass."""
        return self.named.data.subtree_com["torso"].copy()

    def center_of_mass_velocity(self):
        """Returns the velocity of the center-of-mass."""
        return self.named.data.sensordata["torso_subtreelinvel"].copy()

    def torso_vertical_orientation(self):
        """Returns the z-projection of the torso orientation matrix."""
        return self.named.data.xmat["torso", ["zx", "zy", "zz"]]

    def joint_angles(self):
        """Returns the state without global orientation or position."""
        return self.data.qpos[7:].copy()  # Skip the 7 DoFs of the free root joint.

    def extremities(self):
        """Returns end effector positions in egocentric frame."""
        torso_frame = self.named.data.xmat["torso"].reshape(3, 3)
        torso_pos = self.named.data.xpos["torso"]
        positions = []
        for side in ("left_", "right_"):
            for limb in ("hand", "foot"):
                torso_to_limb = self.named.data.xpos[side + limb] - torso_pos
                positions.append(torso_to_limb.dot(torso_frame))
        return np.hstack(positions)


class Humanoid(base.Task):
    """A humanoid task."""

    def __init__(self, move_speed, pure_state, random=None):
        """Initializes an instance of `Humanoid`.
    Args:
      move_speed: A float. If this value is zero, reward is given simply for
        standing up. Otherwise this specifies a target horizontal velocity for
        the walking task.
      pure_state: A bool. Whether the observations consist of the pure MuJoCo
        state or includes some useful features thereof.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
        self._move_speed = move_speed
        self._pure_state = pure_state
        super(Humanoid, self).__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.
    Args:
      physics: An instance of `Physics`.
    """
        # Find a collision-free random initial configuration.
        penetrating = True
        while penetrating:
            randomizers.randomize_limited_and_rotational_joints(physics, self.random)
            # Check for collisions.
            physics.after_reset()
            penetrating = physics.data.ncon > 0
        super(Humanoid, self).initialize_episode(physics)

    def get_observation(self, physics):
        """Returns either the pure state or a set of egocentric features."""
        obs = collections.OrderedDict()
        if self._pure_state:
            obs["position"] = physics.position()
            obs["velocity"] = physics.velocity()
        else:
            obs["joint_angles"] = physics.joint_angles()
            obs["head_height"] = physics.head_height()
            obs["extremities"] = physics.extremities()
            obs["torso_vertical"] = physics.torso_vertical_orientation()
            obs["com_velocity"] = physics.center_of_mass_velocity()
            obs["velocity"] = physics.velocity()
        return obs

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        standing = rewards.tolerance(
            physics.head_height(),
            bounds=(_STAND_HEIGHT, float("inf")),
            margin=_STAND_HEIGHT / 4,
        )
        upright = rewards.tolerance(
            physics.torso_upright(),
            bounds=(0.9, float("inf")),
            sigmoid="linear",
            margin=1.9,
            value_at_margin=0,
        )
        stand_reward = standing * upright
        small_control = rewards.tolerance(
            physics.control(), margin=1, value_at_margin=0, sigmoid="quadratic"
        ).mean()
        small_control = (4 + small_control) / 5
        if self._move_speed == 0:
            horizontal_velocity = physics.center_of_mass_velocity()[[0, 1]]
            dont_move = rewards.tolerance(horizontal_velocity, margin=2).mean()
            return small_control * stand_reward * dont_move
        else:
            com_velocity = np.linalg.norm(physics.center_of_mass_velocity()[[0, 1]])
            move = rewards.tolerance(
                com_velocity,
                bounds=(self._move_speed, float("inf")),
                margin=self._move_speed,
                value_at_margin=0,
                sigmoid="linear",
            )
            move = (5 * move + 1) / 6
            return small_control * stand_reward * move
