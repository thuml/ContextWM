import collections
import functools
import logging
import os
import pathlib
import re
import sys
import warnings

try:
    import rich.traceback
    rich.traceback.install()
except ImportError:
    pass

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # tensorboard
logging.getLogger().setLevel("ERROR")
warnings.filterwarnings("ignore", ".*box bound precision lowered.*")

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
import ruamel.yaml as yaml
import torch
import random

import wmlib
import wmlib.envs as envs
import wmlib.agents as agents
import wmlib.utils as utils


def main():

    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent.parent / "configs" / "apv_finetuning.yaml").read_text()
    )
    parsed, remaining = utils.Flags(configs=["defaults"]).parse(known_only=True)
    config = utils.Config(configs["defaults"])
    for name in parsed.configs:
        config = config.update(configs[name])
    config = utils.Flags(config).parse(remaining)

    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    config.save(logdir / "config.yaml")
    print(config, "\n")
    print("Logdir", logdir)
    if config.load_logdir != "none":
        load_logdir = pathlib.Path(config.load_logdir).expanduser()
        print("Loading Logdir", load_logdir)

    utils.snapshot_src(".", logdir / "src", ".gitignore")

    message = "No GPU found. To actually train on CPU remove this assert."
    assert torch.cuda.is_available(), message  # FIXME
    assert config.precision in (16, 32), config.precision
    if config.precision == 16:
        wmlib.ENABLE_FP16 = True  # enable fp16 here since only cuda can use fp16
        print("setting fp16")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device != "cpu":
        torch.set_num_threads(1)

    # reproducibility
    seed = config.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # no apparent impact on speed
    torch.backends.cudnn.benchmark = True  # faster, increases memory though.., no impact on seed

    train_replay = wmlib.Replay(logdir / "train_episodes", seed=seed, **config.replay)
    eval_replay = wmlib.Replay(logdir / "eval_episodes", seed=seed, **dict(
        capacity=config.replay.capacity // 10,
        minlen=config.dataset.length,
        maxlen=config.dataset.length))
    step = utils.Counter(train_replay.stats["total_steps"])
    outputs = [
        utils.TerminalOutput(),
        utils.JSONLOutput(logdir),
        utils.TensorBoardOutputPytorch(logdir),
    ]
    logger = utils.Logger(step, outputs, multiplier=config.action_repeat)
    metrics = collections.defaultdict(list)

    should_train = utils.Every(config.train_every)
    should_log = utils.Every(config.log_every)
    should_video_train = utils.Every(config.eval_every)
    should_video_eval = utils.Every(config.eval_every)
    should_expl = utils.Until(config.expl_until // config.action_repeat)

    # save experiment used config
    with open(logdir / "used_config.yaml", "w") as f:
        f.write("## command line input:\n## " + " ".join(sys.argv) + "\n##########\n\n")
        yaml.dump(config, f)

    def make_env(mode):
        suite, task = config.task.split("_", 1)
        if suite == "dmc":
            env = envs.DMC(
                task, config.action_repeat, config.render_size, config.dmc_camera)
            env = envs.NormalizeAction(env)
        elif suite == "metaworld":
            task = "-".join(task.split("_"))
            env = envs.MetaWorld(
                task,
                config.seed,
                config.action_repeat,
                config.render_size,
                config.camera,
            )
            env = envs.NormalizeAction(env)
        elif suite == "carla":
            env = envs.Carla(ports=[config.carla_port, config.carla_port + 10],
                             fix_weather=task, frame_skip=config.action_repeat, **config.carla)
            env = envs.NormalizeAction(env)
        elif suite == "dmcr":
            env = envs.DMCRemastered(task, config.action_repeat, config.render_size,
                                     config.dmc_camera, config.dmcr_vary)
            env = envs.NormalizeAction(env)
        else:
            raise NotImplementedError(suite)
        env = envs.TimeLimit(env, config.time_limit)
        return env

    def per_episode(ep, mode):
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        if "metaworld" in config.task or 'rlbench' in config.task:
            success = float(np.sum(ep["success"]) >= 1.0)
            print(
                f"{mode.title()} episode has {float(success)} success, {length} steps and return {score:.1f}."
            )
            logger.scalar(f"{mode}_success", float(success))
        elif "carla" in config.task:
            print(ep["dist_s"].shape, ep["collision_cost"].shape, ep["steer_cost"].shape)
            dist_s = float(np.max(ep["dist_s"]))
            collision_cost = float(np.min(ep["collision_cost"]))
            steer_cost = float(np.min(ep["steer_cost"]))
            centering_cost = float(np.min(ep["centering_cost"]))
            print(
                f"{mode.title()} episode max dist_s is {float(dist_s)}, {length} steps and return {score:.1f}."
            )
            print(f"{mode.title()} episode has {float(collision_cost)} collision cost, {float(steer_cost)} steer cost and {float(centering_cost)} centering cost.")
            logger.scalar(f"{mode}_dist_s", float(dist_s))
            logger.scalar(f"{mode}_collision_cost", float(collision_cost))
            logger.scalar(f"{mode}_steer_cost", float(steer_cost))
            logger.scalar(f"{mode}_centering_cost", float(centering_cost))
        else:
            print(f"{mode.title()} episode has {length} steps and return {score:.1f}.")
        logger.scalar(f"{mode}_return", score)
        logger.scalar(f"{mode}_length", length)
        for key, value in ep.items():
            if re.match(config.log_keys_sum, key):
                logger.scalar(f"sum_{mode}_{key}", ep[key].sum())
            if re.match(config.log_keys_mean, key):
                logger.scalar(f"mean_{mode}_{key}", ep[key].mean())
            if re.match(config.log_keys_max, key):
                logger.scalar(f"max_{mode}_{key}", ep[key].max(0).mean())
        should = {"train": should_video_train, "eval": should_video_eval}[mode]
        if should(step):
            for key in config.log_keys_video:
                logger.video(f"{mode}_policy_{key}", ep[key])
        replay = dict(train=train_replay, eval=eval_replay)[mode]
        logger.add(replay.stats, prefix=mode)
        if mode != 'eval':  # NOTE: to aggregate eval results at last
            logger.write()

    print("Create envs.")
    is_carla = config.task.split("_", 1)[0] == 'carla'
    num_eval_envs = min(config.envs, config.eval_eps)
    # only one env for carla
    if is_carla:
        assert config.envs == 1 and num_eval_envs == 1
    if config.envs_parallel == "none":
        train_envs = [make_env("train") for _ in range(config.envs)]
        eval_envs = [make_env("eval") for _ in range(num_eval_envs)]
    else:
        make_async_env = lambda mode: envs.Async(
            functools.partial(make_env, mode), config.envs_parallel)
        train_envs = [make_async_env("train") for _ in range(config.envs)]
        eval_envs = [make_async_env("eval") for _ in range(num_eval_envs)]
    act_space = train_envs[0].act_space
    obs_space = train_envs[0].obs_space
    train_driver = wmlib.Driver(train_envs, device)
    train_driver.on_episode(lambda ep: per_episode(ep, mode="train"))
    train_driver.on_step(lambda tran, worker: step.increment())
    train_driver.on_step(train_replay.add_step)
    train_driver.on_reset(train_replay.add_step)
    eval_driver = wmlib.Driver(eval_envs, device)
    eval_driver.on_episode(lambda ep: per_episode(ep, mode="eval"))
    eval_driver.on_episode(eval_replay.add_episode)

    prefill = max(0, config.prefill - train_replay.stats["total_steps"])
    if prefill:
        print(f"Prefill dataset ({prefill} steps).")
        random_agent = agents.RandomAgent(act_space)
        train_driver(random_agent, steps=prefill, episodes=1)
        eval_driver(random_agent, episodes=1)
        train_driver.reset()
        eval_driver.reset()

    print("Create agent.")
    train_dataset = iter(train_replay.dataset(**config.dataset))
    report_dataset = iter(train_replay.dataset(**config.dataset))
    eval_dataset = iter(eval_replay.dataset(pin_memory=False, **config.dataset))

    def next_batch(iter, fp16=True):
        # casts to fp16 and cuda
        dtype = torch.float16 if wmlib.ENABLE_FP16 and fp16 else torch.float32  # only on cuda
        out = {k: v.to(device=device, dtype=dtype) for k, v in next(iter).items()}
        return out

    # the agent needs 1. init modules 2. go to device 3. set optimizer
    agnt = agents.APV_Finetune(config, obs_space, act_space, step)
    agnt = agnt.to(device)
    agnt.init_optimizers()

    train_agent = wmlib.CarryOverState(agnt.train)
    train_agent(next_batch(train_dataset))  # do initial benchmarking pass
    torch.cuda.empty_cache()  # clear cudnn bechmarking cache
    if (logdir / "variables.pt").exists():
        print("Load agent.")
        agnt.load_state_dict(torch.load(logdir / "variables.pt"))
    else:
        if config.load_logdir != "none":
            if "af_rssm" in config.load_modules:
                print(agnt.wm.af_rssm.load_state_dict(torch.load(load_logdir / "rssm_variables.pt"), strict=config.load_strict))
                print("Load af_rssm.")
            if "encoder" in config.load_modules:
                print(agnt.wm.encoder.load_state_dict(torch.load(
                    load_logdir / "encoder_variables.pt"), strict=config.load_strict))
                print("Load encoder.")
            if "decoder" in config.load_modules:
                print(agnt.wm.heads["decoder"].load_state_dict(torch.load(
                    load_logdir / "decoder_variables.pt"), strict=config.load_strict))
                print("Load decoder.")
        print("Pretrain agent.")
        for _ in range(config.pretrain):
            train_agent(next_batch(train_dataset))
    train_policy = lambda *args: agnt.policy(
        *args, mode="explore" if should_expl(step) else "train")
    eval_policy = lambda *args: agnt.policy(*args, mode="eval")

    def train_step(tran, worker):
        if should_train(step):
            for _ in range(config.train_steps):
                mets = train_agent(next_batch(train_dataset))
                [metrics[key].append(value) for key, value in mets.items()]
        if should_log(step):
            for name, values in metrics.items():
                logger.scalar(name, np.array(values, np.float64).mean())
                metrics[name].clear()
            logger.add(agnt.report(next_batch(report_dataset)), prefix="train")
            logger.write(fps=True)
    train_driver.on_step(train_step)

    try:
        while step < config.steps:
            logger.write()
            print("Start evaluation.")
            logger.add(agnt.report(next_batch(eval_dataset)), prefix="eval")
            eval_driver(eval_policy, episodes=config.eval_eps)
            logger.write()  # NOTE: to aggregate eval results

            if is_carla and int(step) % 50000 < 5000:
                torch.save(agnt.state_dict(), logdir / ("variables_s" + str(int(step)) + ".pt"))

            if config.stop_steps != -1 and step >= config.stop_steps:
                break
            else:
                print("Start training.")
                train_driver(train_policy, steps=config.eval_every)
                torch.save(agnt.state_dict(), logdir / "variables.pt")
    except KeyboardInterrupt:
        print("Keyboard Interrupt - saving agent")
        torch.save(agnt.state_dict(), logdir / "variables.pt")
    except Exception as e:
        print("Training Error:", e)
        torch.save(agnt.state_dict(), logdir / "variables_error.pt")
        raise e
    finally:
        for env in train_envs + eval_envs:
            try:
                env.finish()
            except Exception:
                try:
                    env.close()
                except Exception:
                    pass

    torch.save(agnt.state_dict(), logdir / "variables.pt")


if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    main()
