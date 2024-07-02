from __future__ import annotations

import math
import random
import string
from pathlib import Path
from signal import SIGINT, signal

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import record_episode_statistics, rescale_action
from PyFlyt import gym_envs  # noqa
from PyFlyt.gym_envs import FlattenWaypointEnv  # noqa
from tqdm import tqdm
from wingman import ReplayBuffer, Wingman
from wingman.utils import cpuize, gpuize, shutdown_handler

from dogfighter.algorithms import CCGE
from dogfighter.algorithms.ccge import CCGEParams
from dogfighter.models.bases import BaseActor
from dogfighter.models.mlp import MlpActor, MlpEnvParams, MlpQUEnsemble
from dogfighter.models.mlp.mlp_bases import MlpModelParams, MlpObservation


def train(wm: Wingman) -> None:
    # pull the config out of wingman
    cfg = wm.cfg

    # setup environment and algorithm
    vec_env = setup_vector_environment(wm)
    alg = setup_algorithm(wm)

    # setup replay buffer
    memory = ReplayBuffer(
        mem_size=cfg.buffer_size,
        mode="torch",
        device=wm.device,
        store_on_device=True,
    )

    # logging metrics
    wm.log["epoch"] = 0
    wm.log["eval_perf"] = -math.inf
    wm.log["max_eval_perf"] = -math.inf
    next_eval_step = 0

    """START TRAINING"""
    while memory.count <= cfg.total_steps:
        print("\n\n")
        print(f"New epoch @ {memory.count} / {cfg.total_steps} total transitions.")
        wm.log["epoch"] += 1

        """ENVIRONMENT ROLLOUT"""
        alg.eval()
        alg.zero_grad()
        with torch.no_grad():
            obs, info = vec_env.reset()
            exclusion_mask = np.zeros(vec_env.num_envs, dtype=bool)

            # step for one episode
            print("Collecting transitions...")
            for i in range(cfg.vec_env_steps_per_epoch):
                # compute an action depending on whether we're exploring or not
                if memory.count < cfg.exploration_steps:
                    # sample an action from the env
                    act = vec_env.action_space.sample()
                else:
                    # get an action from the actor
                    # this is a tensor
                    policy_observation = MlpObservation(obs=gpuize(obs, wm.device))
                    act, _ = alg.actor.sample(*alg.actor(policy_observation))

                    # convert the action to cpu
                    act = cpuize(act)

                # step the transition
                next_obs, rew, term, trunc, info = vec_env.step(act)

                # get the mask of indexes of outputs we want to exclude

                # store stuff in mem
                memory.push(
                    [
                        obs[exclusion_mask, ...],
                        next_obs[exclusion_mask, ...],
                        act[exclusion_mask, ...],
                        np.expand_dims(rew, axis=-1)[exclusion_mask, ...],
                        np.expand_dims(term, axis=-1)[exclusion_mask, ...],
                    ],
                    bulk=True,
                    random_rollover=cfg.random_rollover,
                )

                # new observation is the next observation
                obs = next_obs
                exclusion_mask = ~term & ~trunc

        """TRAINING RUN"""
        print(
            f"Training epoch {wm.log['epoch']}, Replay Buffer Capacity {memory.count} / {memory.mem_size}"
        )
        alg.train()
        for stuff in tqdm(
            memory.iter_sample(
                batch_size=cfg.batch_size,
                num_iter=cfg.model_updates_per_epoch,
            ),
            total=cfg.model_updates_per_epoch,
        ):
            # unpack batches
            obs = MlpObservation(obs=gpuize(stuff[0], wm.device))
            next_obs = MlpObservation(obs=gpuize(stuff[1], wm.device))
            act = gpuize(stuff[2], wm.device)
            rew = gpuize(stuff[3], wm.device)
            term = gpuize(stuff[4], wm.device)

            # take a gradient step
            update_info = alg.update(
                obs=obs, act=act, next_obs=next_obs, term=term, rew=rew
            )
            wm.log.update(update_info)

        """EVALUATE POLICY"""
        if memory.count >= next_eval_step:
            next_eval_step = (
                int(memory.count / cfg.eval_steps_ratio) + 1
            ) * cfg.eval_steps_ratio
            wm.log["eval_perf"] = evaluate(wm=wm, actor=alg.actor)
            wm.log["max_eval_perf"] = max(
                [float(wm.log["max_eval_perf"]), float(wm.log["eval_perf"])]
            )
            print(f"Eval score: {wm.log['eval_perf']}")

        """WANDB"""
        wm.log["num_transitions"] = memory.count
        wm.log["buffer_size"] = memory.__len__()

        # save weights
        to_update, model_file, _ = wm.checkpoint(
            loss=-float(wm.log["eval_perf"]), step=wm.log["num_transitions"]
        )
        if to_update:
            torch.save(alg.state_dict(), model_file)


def evaluate(wm: Wingman, actor: BaseActor | None) -> float:
    # setup the environment and actor
    env = setup_single_environment(wm)
    actor = actor or setup_algorithm(wm).actor

    # start the evaluation loops
    cumulative_rewards: list[float] = []
    for _ in range(wm.cfg.eval_num_episodes):
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

        cumulative_rewards.append(info["episode"]["r"][0])

    return float(np.mean(cumulative_rewards))


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
        "".join(random.choices(string.ascii_letters + string.digits, k=8))
    ).with_suffix(".gif")

    iio.imwrite(
        gif_path,
        frames,
        fps=30,
    )

    return gif_path.absolute()


def setup_vector_environment(wm: Wingman) -> AsyncVectorEnv:
    return AsyncVectorEnv(
        [
            lambda i=i: setup_single_environment(wm, for_vector=True)
            for i in range(wm.cfg.num_envs)
        ]
    )


def setup_single_environment(wm: Wingman, for_vector: bool = False) -> gym.Env:
    # define one env
    env = gym.make(
        wm.cfg.env_name,
        render_mode="human" if wm.cfg.display or wm.cfg.render else None,
        flight_mode=-1,
    )
    env = rescale_action.RescaleAction(env, min_action=-1.0, max_action=1.0)

    # wrap in flatten if needed
    if "waypoint" in wm.cfg.env_name.lower():
        env = FlattenWaypointEnv(env, context_length=1)

    # recording wrapper
    if not for_vector:
        env = record_episode_statistics.RecordEpisodeStatistics(env)

    # record observation space shape
    wm.cfg.obs_size = env.observation_space.shape[0]  # pyright: ignore[reportOptionalSubscript]
    wm.cfg.act_size = env.action_space.shape[0]  # pyright: ignore[reportOptionalSubscript]

    return env


def setup_algorithm(wm: Wingman) -> CCGE:
    # define some params
    env_params = MlpEnvParams(
        obs_size=wm.cfg.obs_size,
        act_size=wm.cfg.act_size,
    )
    model_params = MlpModelParams(
        qu_num_ensemble=wm.cfg.qu_num_ensemble,
        embed_dim=wm.cfg.embed_dim,
    )
    algorithm_params = CCGEParams()

    # define the algorithm, conditionally jit
    alg = CCGE(
        actor_type=MlpActor,
        critic_type=MlpQUEnsemble,
        env_params=env_params,
        model_params=model_params,
        algorithm_params=algorithm_params,
        device=torch.device(wm.device),
    )
    if not wm.cfg.debug:
        torch.compile(alg)

    # get latest weight files
    has_weights, model_file, _ = wm.get_weight_files()
    if has_weights:
        # load the model
        alg.load_state_dict(
            torch.load(model_file, map_location=torch.device(wm.cfg.device))
        )

    return alg


if __name__ == "__main__":
    signal(SIGINT, shutdown_handler)
    wm = Wingman(config_yaml="./configs/quadx_pole_waypoints_config.yaml")

    if wm.cfg.train:
        train(wm)
    elif wm.cfg.eval:
        evaluate(wm=wm, actor=None)
    elif wm.cfg.render:
        print(render_gif(wm=wm, actor=None))
    else:
        print("So this is life now.")
