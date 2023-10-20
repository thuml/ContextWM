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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
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
import wmlib.datasets as datasets


def main():

    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent.parent / "configs" / "apv_pretraining.yaml").read_text()
    )
    parsed, remaining = utils.Flags(configs=["defaults"]).parse(known_only=True)
    config = utils.Config(configs["defaults"])
    for name in parsed.configs:
        config = config.update(configs[name])
    config = utils.Flags(config).parse(remaining)

    logdir = pathlib.Path(config.logdir).expanduser()
    load_logdir = pathlib.Path(config.load_logdir).expanduser()
    load_model_dir = pathlib.Path(config.load_model_dir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    config.save(logdir / "config.yaml")
    print(config, "\n")
    print("Logdir", logdir)
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

    if config['dataset_type'] == 'replay':
        train_replay = wmlib.ReplayWithoutAction(
            logdir / "train_episodes",
            load_directory=load_logdir / "train_episodes",
            seed=seed,
            **config.replay
        )
    elif config['dataset_type'] == 'something':
        somethingv2_dataset = datasets.SomethingV2(
            root_path=config['video_dir'],
            list_file=f'data/somethingv2/{config.video_list}.txt',
            segment_len=config['replay']['minlen'],
            manual_labels=config['manual_labels'],
        )
        train_replay = datasets.DummyReplay(somethingv2_dataset)

        if config.eval_video_list != 'none':
            somethingv2_eval_dataset = datasets.SomethingV2(
                root_path=config['video_dir'],
                list_file=f'data/somethingv2/{config.eval_video_list}.txt',
                segment_len=config['replay']['minlen'],
                manual_labels=config['manual_labels'],
            )
            eval_replay = datasets.DummyReplay(somethingv2_eval_dataset)
    elif config['dataset_type'] == 'human':
        human_dataset = datasets.Human(
            root_path=config['video_dir'],
            list_file=f'data/human36m/{config.video_list}.txt',
            segment_len=config['replay']['minlen'],
        )
        train_replay = datasets.DummyReplay(human_dataset)
    elif config['dataset_type'] == 'ytb':
        ytb_dataset = datasets.YoutubeDriving(
            root_path=config['video_dir'],
            list_file=f'data/ytb_driving/{config.video_list}.txt',
            segment_len=config['replay']['minlen'],
        )
        train_replay = datasets.DummyReplay(ytb_dataset)
    elif config['dataset_type'] == 'mixture':
        somethingv2_dataset = datasets.SomethingV2(
            root_path=config['video_dirs'][0],
            list_file=f'data/somethingv2/{config.video_lists[0]}.txt',
            segment_len=config['replay']['minlen'],
            manual_labels=config['manual_labels'],
        )
        human_dataset = datasets.Human(
            root_path=config['video_dirs'][1],
            list_file=f'data/human36m/{config.video_lists[1]}.txt',
            segment_len=config['replay']['minlen'],
        )
        ytb_dataset = datasets.YoutubeDriving(
            root_path=config['video_dirs'][2],
            list_file=f'data/ytb_driving/{config.video_lists[2]}.txt',
            segment_len=config['replay']['minlen'],
        )
        mixture_dataset = datasets.Mixture([somethingv2_dataset, human_dataset, ytb_dataset])
        train_replay = datasets.DummyReplay(mixture_dataset)
    else:
        raise NotImplementedError

    step = utils.Counter(train_replay.stats["total_steps"])
    outputs = [
        utils.TerminalOutput(),
        utils.JSONLOutput(logdir),
        utils.TensorBoardOutputPytorch(logdir),
    ]
    logger = utils.Logger(step, outputs, multiplier=config.action_repeat)
    metrics = collections.defaultdict(list)

    should_log = utils.Every(config.log_every)
    should_video = utils.Every(config.video_every)
    should_save = utils.Every(config.eval_every)

    def make_env(mode):
        suite, task = config.task.split("_", 1)
        if suite == "dmc":
            env = envs.DMC(
                task, config.action_repeat, config.render_size, config.dmc_camera
            )
            env = envs.NormalizeAction(env)
        elif suite == "atari":
            env = envs.Atari(
                task, config.action_repeat, config.render_size, config.atari_grayscale
            )
            env = envs.OneHotAction(env)
        elif suite == "crafter":
            assert config.action_repeat == 1
            outdir = logdir / "crafter" if mode == "train" else None
            reward = bool(["noreward", "reward"].index(task)) or mode == "eval"
            env = envs.Crafter(outdir, reward)
            env = envs.OneHotAction(env)
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
        elif suite == "dmcgb":
            env = envs.DMCGeneralization(task, config.action_repeat, config.render_size, config.dmc_camera, config.dmcgb_mode,
                                         config.seed)
            env = envs.NormalizeAction(env)
        elif suite == "dmcr":
            env = envs.DMCRemastered(task, config.action_repeat, config.render_size,
                                     config.dmc_camera, config.dmcr_vary)
            env = envs.NormalizeAction(env)
        else:
            raise NotImplementedError(suite)
        env = envs.TimeLimit(env, config.time_limit)
        return env

    print("Create envs.")
    env = make_env("train")
    act_space, obs_space = env.act_space, env.obs_space

    print("Create agent.")
    train_dataset = iter(train_replay.dataset(**config.dataset))
    report_dataset = iter(train_replay.dataset(**config.dataset))
    if config.eval_video_list != 'none':
        eval_dataset = iter(eval_replay.dataset(**config.dataset))

    def next_batch(iter, fp16=True):
        # casts to fp16 and cuda
        dtype = torch.float16 if wmlib.ENABLE_FP16 and fp16 else torch.float32  # only on cuda
        out = {k: v.to(device=device, dtype=dtype) for k, v in next(iter).items()}
        return out

    # the agent needs 1. init modules 2. go to device 3. set optimizer
    agnt = agents.APV_Pretrain(config, obs_space, act_space, step)
    agnt = agnt.to(device)
    agnt.init_optimizers()

    train_agent = wmlib.CarryOverState(agnt.train)
    train_agent(next_batch(train_dataset))  # do initial benchmarking pass
    torch.cuda.empty_cache()  # clear cudnn bechmarking cache
    if (logdir / "variables.pt").exists():
        print("Load agent.")
        print(agnt.load_state_dict(torch.load(logdir / "variables.pt")))
        # agnt.load_state_dict(torch.load(logdir / "variables.pt"), strict=False)
    if (load_model_dir / "variables.pt").exists():
        print("Load agent.")
        print(agnt.load_state_dict(torch.load(load_model_dir / "variables.pt")))
        # agnt.load_state_dict(torch.load(logdir / "variables.pt"), strict=False)

    def save_models(suffix=''):
        agnt.save_model(logdir, suffix)

    print("Train a video prediction model.")
    # for step in tqdm(range(step, config.steps), total=config.steps, initial=step):
    for _ in range(int(step.value), int(config.steps)):
        mets = train_agent(next_batch(train_dataset))
        [metrics[key].append(value) for key, value in mets.items()]
        step.increment()

        if should_log(step):
            for name, values in metrics.items():
                try:
                    logger.scalar(name, np.array(values, np.float64).mean())
                except TypeError:
                    # FIXME to be compatible with TF 2.3
                    if name.endswith('loss_scale'):
                        logger.scalar(name, values[0]().numpy().item())
                metrics[name].clear()
            if should_video(step):
                logger.add(agnt.report(next_batch(report_dataset), recon=True), prefix="train")
            logger.write(fps=True)

        if should_save(step):
            save_models()
            if config.save_all_models and int(step) % 50000 == 1:
                save_models('_s' + str(int(step)))

            if config.eval_video_list != 'none':
                mets = agnt.eval(next_batch(eval_dataset))[1]
                for name, values in mets.items():
                    if name.endswith('loss'):
                        logger.scalar('eval/' + name, np.array(values, np.float64).mean())
                logger.add(agnt.report(next_batch(eval_dataset), recon=True), prefix="val")
                logger.write(fps=True)

    env.close()
    save_models()


if __name__ == "__main__":
    main()
