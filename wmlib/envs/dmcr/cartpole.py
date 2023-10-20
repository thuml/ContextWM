import collections
import os
import random
import xml.etree.ElementTree as ET

import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base, common
from dm_control.utils import containers, rewards
from lxml import etree
from six.moves import range

from .import DMCR_VARY, SUITE_DIR, register
from .rng import dmcr_random

from .generate_visuals import get_assets

_DEFAULT_TIME_LIMIT = 10


def get_model(visual_seed, vary=["camera", "light"]):
    default_model = _make_model(n_poles=1)
    xml = ET.fromstring(default_model)
    if visual_seed != 0:
        with dmcr_random(visual_seed):
            camera_x = random.uniform(-0.15, 0.15)
            camera_y = random.uniform(-5.0, -3.0)
            camera_z = random.uniform(0.8, 1.2)
            if "camera" in vary:
                xml[5][1].attrib["pos"] = f"{camera_x} {camera_y} {camera_z}"

            light_x = random.uniform(-1, 1)
            light_y = random.uniform(-1, 1)
            light_z = random.uniform(4.5, 7.5)
            if "light" in vary:
                xml[5][0].attrib["pos"] = f"{light_x} {light_y} {light_z}"
    return ET.tostring(xml, encoding="utf8", method="xml")


@register("cartpole", "balance")
def balance(
    time_limit=_DEFAULT_TIME_LIMIT, dynamics_seed=None, visual_seed=None, vary=DMCR_VARY
):
    model = get_model(visual_seed, vary)
    assets, _ = get_assets(visual_seed, vary)
    physics = Physics.from_xml_string(model, assets)
    task = Balance(swing_up=False, sparse=False, random=dynamics_seed)
    return control.Environment(physics, task, time_limit=time_limit)


@register("cartpole", "balance_sparse")
def balance_sparse(
    time_limit=_DEFAULT_TIME_LIMIT, dynamics_seed=None, visual_seed=None, vary=DMCR_VARY
):
    model = get_model(visual_seed, vary)
    assets, _ = get_assets(visual_seed, vary)
    physics = Physics.from_xml_string(model, assets)
    task = Balance(swing_up=False, sparse=True, random=dynamics_seed)
    return control.Environment(physics, task, time_limit=time_limit)


@register("cartpole", "swingup")
def swingup(
    time_limit=_DEFAULT_TIME_LIMIT, dynamics_seed=None, visual_seed=None, vary=DMCR_VARY
):
    model = get_model(visual_seed, vary)
    assets, _ = get_assets(visual_seed, vary)
    physics = Physics.from_xml_string(model, assets)
    task = Balance(swing_up=True, sparse=False, random=dynamics_seed)
    return control.Environment(physics, task, time_limit=time_limit)


@register("cartpole", "swingup_sparse")
def swingup_sparse(
    time_limit=_DEFAULT_TIME_LIMIT,
    dynamics_seed=None,
    visual_seed=None,
    vary=DMCR_VARY,
):
    model = get_model(visual_seed, vary)
    assets, _ = get_assets(visual_seed, vary)
    physics = Physics.from_xml_string(model, assets)
    task = Balance(swing_up=True, sparse=True, random=dynamics_seed)
    return control.Environment(physics, task, time_limit=time_limit)


def _make_model(n_poles):
    """Generates an xml string defining a cart with `n_poles` bodies."""
    with open(
        os.path.join(SUITE_DIR, os.path.join("assets", "cartpole.xml")), "r"
    ) as f:
        xml_string = f.read()
    if n_poles == 1:
        return xml_string
    mjcf = etree.fromstring(xml_string)
    parent = mjcf.find("./worldbody/body/body")  # Find first pole.
    # Make chain of poles.
    for pole_index in range(2, n_poles + 1):
        child = etree.Element(
            "body", name="pole_{}".format(pole_index), pos="0 0 1", childclass="pole"
        )
        etree.SubElement(child, "joint", name="hinge_{}".format(pole_index))
        etree.SubElement(child, "geom", name="pole_{}".format(pole_index))
        parent.append(child)
        parent = child
    # Move plane down.
    floor = mjcf.find("./worldbody/geom")
    floor.set("pos", "0 0 {}".format(1 - n_poles - 0.05))
    # Move cameras back.
    cameras = mjcf.findall("./worldbody/camera")
    cameras[0].set("pos", "0 {} 1".format(-1 - 2 * n_poles))
    cameras[1].set("pos", "0 {} 2".format(-2 * n_poles))
    return etree.tostring(mjcf, pretty_print=True)


class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Cartpole domain."""

    def cart_position(self):
        """Returns the position of the cart."""
        return self.named.data.qpos["slider"][0]

    def angular_vel(self):
        """Returns the angular velocity of the pole."""
        return self.data.qvel[1:]

    def pole_angle_cosine(self):
        """Returns the cosine of the pole angle."""
        return self.named.data.xmat[2:, "zz"]

    def bounded_position(self):
        """Returns the state, with pole angle split into sin/cos."""
        return np.hstack(
            (self.cart_position(), self.named.data.xmat[2:, ["zz", "xz"]].ravel())
        )


class Balance(base.Task):
    """A Cartpole `Task` to balance the pole.
  State is initialized either close to the target configuration or at a random
  configuration.
  """

    _CART_RANGE = (-0.25, 0.25)
    _ANGLE_COSINE_RANGE = (0.995, 1)

    def __init__(self, swing_up, sparse, random=None):
        """Initializes an instance of `Balance`.
    Args:
      swing_up: A `bool`, which if `True` sets the cart to the middle of the
        slider and the pole pointing towards the ground. Otherwise, sets the
        cart to a random position on the slider and the pole to a random
        near-vertical position.
      sparse: A `bool`, whether to return a sparse or a smooth reward.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
        self._sparse = sparse
        self._swing_up = swing_up
        super(Balance, self).__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.
    Initializes the cart and pole according to `swing_up`, and in both cases
    adds a small random initial velocity to break symmetry.
    Args:
      physics: An instance of `Physics`.
    """
        nv = physics.model.nv
        if self._swing_up:
            physics.named.data.qpos["slider"] = 0.01 * self.random.randn()
            physics.named.data.qpos["hinge_1"] = np.pi + 0.01 * self.random.randn()
            physics.named.data.qpos[2:] = 0.1 * self.random.randn(nv - 2)
        else:
            physics.named.data.qpos["slider"] = self.random.uniform(-0.1, 0.1)
            physics.named.data.qpos[1:] = self.random.uniform(-0.034, 0.034, nv - 1)
        physics.named.data.qvel[:] = 0.01 * self.random.randn(physics.model.nv)
        super(Balance, self).initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation of the (bounded) physics state."""
        obs = collections.OrderedDict()
        obs["position"] = physics.bounded_position()
        obs["velocity"] = physics.velocity()
        return obs

    def _get_reward(self, physics, sparse):
        if sparse:
            cart_in_bounds = rewards.tolerance(
                physics.cart_position(), self._CART_RANGE
            )
            angle_in_bounds = rewards.tolerance(
                physics.pole_angle_cosine(), self._ANGLE_COSINE_RANGE
            ).prod()
            return cart_in_bounds * angle_in_bounds
        else:
            upright = (physics.pole_angle_cosine() + 1) / 2
            centered = rewards.tolerance(physics.cart_position(), margin=2)
            centered = (1 + centered) / 2
            small_control = rewards.tolerance(
                physics.control(), margin=1, value_at_margin=0, sigmoid="quadratic"
            )[0]
            small_control = (4 + small_control) / 5
            small_velocity = rewards.tolerance(physics.angular_vel(), margin=5).min()
            small_velocity = (1 + small_velocity) / 2
            return upright.mean() * small_control * small_velocity * centered

    def get_reward(self, physics):
        """Returns a sparse or a smooth reward, as specified in the constructor."""
        return self._get_reward(physics, sparse=self._sparse)
