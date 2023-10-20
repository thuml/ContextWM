import collections
import os
import random
import xml.etree.ElementTree as ET

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base, common
from dm_control.utils import containers

from .import DMCR_VARY, SUITE_DIR, register
from .rng import dmcr_random

from .generate_visuals import get_assets

_DEFAULT_TIME_LIMIT = 20  # (seconds)
_CONTROL_TIMESTEP = 0.02  # (seconds)


def get_model(visual_seed, vary=["camera", "light"]):
    with open(
        os.path.join(SUITE_DIR, os.path.join("assets", "ball_in_cup.xml")), "r"
    ) as f:
        xml = ET.fromstring(f.read())
    if visual_seed != 0:
        with dmcr_random(visual_seed):
            camera_x = random.uniform(-0.05, 0.05)
            camera_y = random.uniform(-0.9, -1.1)
            camera_z = random.uniform(0.8, 1.0)

            light_x = random.uniform(-0.5, 0.5)
            light_y = random.uniform(-0.5, 0.5)
            light_z = random.uniform(1.2, 2.7)
        if "camera" in vary:
            xml[4][2].attrib["pos"] = f"{camera_x} {camera_y} {camera_z}"
        if "light" in vary:
            xml[4][0].attrib["pos"] = f"{light_x} {light_y} {light_z}"
    return ET.tostring(xml, encoding="utf8", method="xml")


@register("ball_in_cup", "catch")
def catch(
    time_limit=_DEFAULT_TIME_LIMIT, dynamics_seed=None, visual_seed=None, vary=DMCR_VARY
):
    model = get_model(visual_seed, vary)
    assets, _ = get_assets(visual_seed, vary)
    physics = Physics.from_xml_string(model, assets)
    task = BallInCup(random=dynamics_seed)
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
    )


class Physics(mujoco.Physics):
    """Physics with additional features for the Ball-in-Cup domain."""

    def ball_to_target(self):
        """Returns the vector from the ball to the target."""
        target = self.named.data.site_xpos["target", ["x", "z"]]
        ball = self.named.data.xpos["ball", ["x", "z"]]
        return target - ball

    def in_target(self):
        """Returns 1 if the ball is in the target, 0 otherwise."""
        ball_to_target = abs(self.ball_to_target())
        target_size = self.named.model.site_size["target", [0, 2]]
        ball_size = self.named.model.geom_size["ball", 0]
        return float(all(ball_to_target < target_size - ball_size))


class BallInCup(base.Task):
    """The Ball-in-Cup task. Put the ball in the cup."""

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.
    Args:
      physics: An instance of `Physics`.
    """
        # Find a collision-free random initial position of the ball.
        penetrating = True
        while penetrating:
            # Assign a random ball position.
            physics.named.data.qpos["ball_x"] = self.random.uniform(-0.2, 0.2)
            physics.named.data.qpos["ball_z"] = self.random.uniform(0.2, 0.5)
            # Check for collisions.
            physics.after_reset()
            penetrating = physics.data.ncon > 0
        super(BallInCup, self).initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation of the state."""
        obs = collections.OrderedDict()
        obs["position"] = physics.position()
        obs["velocity"] = physics.velocity()
        return obs

    def get_reward(self, physics):
        """Returns a sparse reward."""
        return physics.in_target()
