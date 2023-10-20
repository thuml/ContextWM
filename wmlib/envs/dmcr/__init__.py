import os

SUITE_DIR = os.path.dirname(__file__)
DMCR_VARY = ["bg", "floor", "body", "target", "reflectance", "camera", "light"]
ALL_ENVS = {}


def register(domain, task):
    def _register(func):
        if domain not in ALL_ENVS:
            ALL_ENVS[domain] = {}
        ALL_ENVS[domain][task] = func
        return func

    return _register


# register all the tasks
from .ball_in_cup import catch
from .benchmarks import classic, visual_generalization
from .cartpole import balance, balance_sparse, swingup, swingup_sparse
from .cheetah import run
from .finger import spin, turn_easy, turn_hard
from .fish import swim, upright
from .generate_visuals import get_assets
from .hopper import hop, stand
from .humanoid import run, stand, walk
from .pendulum import swingup
from .reacher import easy, hard
from .walker import get_model, run, stand, walk
from .wrapper import make
