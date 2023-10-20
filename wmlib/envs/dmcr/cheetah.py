import collections
import os
import random
import xml.etree.ElementTree as ET

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base, common
from dm_control.utils import containers, rewards

from .import DMCR_VARY, SUITE_DIR, register
from .rng import dmcr_random

from .generate_visuals import get_assets

# How long the simulation will run, in seconds.
_DEFAULT_TIME_LIMIT = 10

# Running speed above which reward is 1.
_RUN_SPEED = 10

SUITE = containers.TaggedTasks()


def get_model(visual_seed, vary=["camera", "light"]):
    with open(os.path.join(SUITE_DIR, os.path.join("assets", "cheetah.xml")), "r") as f:
        xml = ET.fromstring(f.read())
    xml[7][0].attrib["size"] = "100 15 .5"
    if visual_seed != 0:
        with dmcr_random(visual_seed):
            camera_x = random.uniform(-0.25, 0.25)
            camera_y = random.uniform(-3.2, -2.8)
            camera_z = random.uniform(-0.25, 0.25)
            if "camera" in vary:
                xml[7][1][1].attrib["pos"] = f"{camera_x} {camera_y} {camera_z}"

            light_x = random.uniform(-2, 2)
            light_y = random.uniform(-2, 2)
            light_z = random.uniform(1, 5)
            if "light" in vary:
                xml[7][1][0].attrib["pos"] = f"{light_x} {light_y} {light_z}"
    return ET.tostring(xml, encoding="utf8", method="xml")


@register("cheetah", "run")
def run(
    time_limit=_DEFAULT_TIME_LIMIT, dynamics_seed=None, visual_seed=None, vary=DMCR_VARY
):
    model = get_model(visual_seed, vary)
    assets, _ = get_assets(visual_seed, vary)
    physics = Physics.from_xml_string(model, assets)
    task = Cheetah(random=dynamics_seed)
    return control.Environment(physics, task, time_limit=time_limit,)


class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Cheetah domain."""

    def speed(self):
        """Returns the horizontal speed of the Cheetah."""
        return self.named.data.sensordata["torso_subtreelinvel"][0]


class Cheetah(base.Task):
    """A `Task` to train a running Cheetah."""

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # The indexing below assumes that all joints have a single DOF.
        assert physics.model.nq == physics.model.njnt
        is_limited = physics.model.jnt_limited == 1
        lower, upper = physics.model.jnt_range[is_limited].T
        physics.data.qpos[is_limited] = self.random.uniform(lower, upper)

        # Stabilize the model before the actual simulation.
        for _ in range(200):
            physics.step()

        physics.data.time = 0
        self._timeout_progress = 0
        super(Cheetah, self).initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation of the state, ignoring horizontal position."""
        obs = collections.OrderedDict()
        # Ignores horizontal position to maintain translational invariance.
        obs["position"] = physics.data.qpos[1:].copy()
        obs["velocity"] = physics.velocity()
        return obs

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        return rewards.tolerance(
            physics.speed(),
            bounds=(_RUN_SPEED, float("inf")),
            margin=_RUN_SPEED,
            value_at_margin=0,
            sigmoid="linear",
        )
