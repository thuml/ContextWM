import random

import numpy as np

from .import ALL_ENVS, DMCR_VARY

from .wrapper import DMC_Remastered_Env, FrameStack


def uniform_seed_generator(low, high):
    def _generate():
        return random.randint(low, high)

    return _generate


def fixed_seed_generator(seed):
    return lambda: seed


def visual_sim2real(
    domain,
    task,
    num_levels,
    frame_stack=3,
    height=84,
    width=84,
    frame_skip=1,
    channels_last=False,
    vary=DMCR_VARY,
):

    random_start = random.randint(1, 1_000_000)
    train_env = DMC_Remastered_Env(
        task_builder=ALL_ENVS[domain][task],
        visual_seed_generator=uniform_seed_generator(
            random_start, random_start + num_levels
        ),
        dynamics_seed_generator=uniform_seed_generator(0, 1_000_000),
        height=height,
        width=width,
        camera_id=0,
        frame_skip=frame_skip,
        channels_first=not channels_last,
        vary=vary,
    )
    test_env = DMC_Remastered_Env(
        task_builder=ALL_ENVS[domain][task],
        visual_seed_generator=fixed_seed_generator(0),
        dynamics_seed_generator=uniform_seed_generator(0, 1_000_000),
        height=height,
        width=width,
        camera_id=0,
        frame_skip=frame_skip,
        channels_first=not channels_last,
        vary=vary,
    )
    train_env = FrameStack(train_env, frame_stack)
    test_env = FrameStack(test_env, frame_stack)
    return train_env, test_env


def classic(
    domain,
    task,
    visual_seed=0,
    frame_stack=3,
    height=84,
    width=84,
    frame_skip=1,
    channels_last=False,
    vary=DMCR_VARY,
):
    train_env = DMC_Remastered_Env(
        task_builder=ALL_ENVS[domain][task],
        visual_seed_generator=fixed_seed_generator(visual_seed),
        dynamics_seed_generator=uniform_seed_generator(0, 1_000_000),
        height=height,
        width=width,
        camera_id=0,
        frame_skip=frame_skip,
        channels_first=not channels_last,
        vary=vary,
    )
    test_env = DMC_Remastered_Env(
        task_builder=ALL_ENVS[domain][task],
        visual_seed_generator=fixed_seed_generator(visual_seed),
        dynamics_seed_generator=uniform_seed_generator(0, 1_000_000),
        height=height,
        width=width,
        camera_id=0,
        frame_skip=frame_skip,
        channels_first=not channels_last,
        vary=vary,
    )
    train_env = FrameStack(train_env, frame_stack)
    test_env = FrameStack(test_env, frame_stack)
    return train_env, test_env


def visual_generalization(
    domain,
    task,
    num_levels,
    frame_stack=3,
    height=84,
    width=84,
    frame_skip=1,
    channels_last=False,
    vary=DMCR_VARY,
):

    random_start = random.randint(1, 1_000_000)
    train_env = DMC_Remastered_Env(
        task_builder=ALL_ENVS[domain][task],
        visual_seed_generator=uniform_seed_generator(
            random_start, random_start + num_levels
        ),
        dynamics_seed_generator=uniform_seed_generator(0, 1_000_000),
        height=height,
        width=width,
        camera_id=0,
        frame_skip=frame_skip,
        channels_first=not channels_last,
        vary=vary,
    )
    test_env = DMC_Remastered_Env(
        task_builder=ALL_ENVS[domain][task],
        visual_seed_generator=uniform_seed_generator(1, 1_000_000),
        dynamics_seed_generator=uniform_seed_generator(1, 1_000_00),
        height=height,
        width=width,
        camera_id=0,
        frame_skip=frame_skip,
        channels_first=not channels_last,
        vary=vary,
    )
    train_env = FrameStack(train_env, frame_stack)
    test_env = FrameStack(test_env, frame_stack)
    return train_env, test_env
