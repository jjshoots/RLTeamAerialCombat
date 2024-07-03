from __future__ import annotations

import math
from pathlib import Path
from signal import SIGINT, signal

import torch
from wingman import Wingman
from wingman.utils import cpuize, gpuize, shutdown_handler

from dogfighter.models.bases import BaseActor
from dogfighter.models.mlp.mlp_bases import MlpObservation
from env_interaction_utils import env_collect_to_memory, env_evaluate
from setup_utils import (setup_algorithm, setup_replay_buffer,
                         setup_single_environment, setup_vector_environment)


def train(wm: Wingman) -> None:
    # pull the config out of wingman
    cfg = wm.cfg

    # setup envs, alg, replay buffer
    train_env = setup_vector_environment(wm)
    eval_env = setup_vector_environment(wm)
    alg = setup_algorithm(wm)
    memory = setup_replay_buffer(wm)

    # logging metrics
    wm.log["epoch"] = 0
    wm.log["eval_perf"] = -math.inf
    wm.log["max_eval_perf"] = -math.inf
    wm.log["mean_episode_length"] = 0.0
    next_eval_step = 0

    """START TRAINING"""
    while memory.count <= cfg.total_steps:
        print("\n\n")
        print(f"New epoch @ {memory.count} / {cfg.total_steps} total transitions.")
        wm.log["epoch"] += 1

        """ENVIRONMENT ROLLOUT"""
        memory = env_collect_to_memory(
            actor=alg.actor,
            vec_env=train_env,
            device=wm.device,
            memory=memory,
            random_actions=memory.count <= cfg.exploration_steps,
            num_steps=cfg.vec_env_steps_per_epoch,
        )
        wm.log["buffer_size"] = memory.__len__()
        wm.log["num_transitions"] = memory.count

        """TRAINING RUN"""
        print(
            f"Training epoch {wm.log['epoch']}, Replay Buffer Capacity {memory.count} / {memory.mem_size}"
        )
        update_info = alg.update(
            device=wm.device,
            memory=memory,
            batch_size=cfg.batch_size,
            num_gradient_steps=cfg.model_updates_per_epoch,
        )
        wm.log.update(update_info)

        """EVALUATE POLICY"""
        if memory.count >= next_eval_step:
            wm.log["eval_perf"], wm.log["mean_episode_length"] = env_evaluate(
                actor=alg.actor,
                device=wm.device,
                vec_env=eval_env,
                num_episodes=cfg.eval_num_episodes,
            )
            wm.log["max_eval_perf"] = max(
                [float(wm.log["max_eval_perf"]), float(wm.log["eval_perf"])]
            )
            next_eval_step = (
                int(memory.count / cfg.eval_steps_ratio) + 1
            ) * cfg.eval_steps_ratio
            print(f"Eval score: {wm.log['eval_perf']}")

        """WANDB"""
        # save weights
        to_update, model_file, _ = wm.checkpoint(
            loss=-float(wm.log["eval_perf"]), step=wm.log["num_transitions"]
        )
        if to_update:
            torch.save(alg.state_dict(), model_file)


def render_gif(wm: Wingman, actor: BaseActor | None) -> Path:
    import imageio.v3 as iio

    frames = []

    # setup the environment and actor
    env = setup_single_environment(wm)
    actor = actor or setup_algorithm(wm).actor

    term, trunc = False, False
    obs, info = env.reset()

    # step for one episode
    while not term and not trunc:
        # get an action from the actor
        policy_observation = MlpObservation(obs=gpuize(obs, wm.device).unsqueeze(0))
        act = actor.infer(*actor(policy_observation))

        # convert the action to cpu, and remove the batch dim
        act = cpuize(act.squeeze(0))

        # step the transition
        next_obs, rew, term, trunc, info = env.step(act)

        # new observation is the next observation
        obs = next_obs

        # for gif
        frames.append(env.render())

    gif_path = Path("/tmp") / Path(
        "gif"
        # "".join(random.choices(string.ascii_letters + string.digits, k=8))
    ).with_suffix(".gif")

    iio.imwrite(
        gif_path,
        frames,
        fps=30,
    )

    return gif_path.absolute()


if __name__ == "__main__":
    signal(SIGINT, shutdown_handler)
    wm = Wingman(config_yaml="./configs/quadx_pole_waypoints_config.yaml")
    # wm = Wingman(config_yaml="./configs/quadx_pole_balance_config.yaml")
    # wm = Wingman(config_yaml="./configs/quadx_waypoints_config.yaml")

    if wm.cfg.train:
        train(wm)
    elif wm.cfg.eval:
        wm.cfg.num_envs = 1 if wm.cfg.display else wm.cfg.num_envs
        wm.log["eval_perf"], wm.log["mean_episode_length"] = env_evaluate(
            vec_env=setup_vector_environment(wm),
            actor=setup_algorithm(wm).actor,
            device=wm.device,
            num_episodes=wm.cfg.eval_num_episodes,
        )
    elif wm.cfg.render:
        print(render_gif(wm=wm, actor=None))
    else:
        print("So this is life now.")
