from __future__ import annotations

import math
from signal import SIGINT, signal

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import record_episode_statistics
from wingman import ReplayBuffer, Wingman
from wingman.utils import cpuize, gpuize, shutdown_handler

from dogfighter.algorithms import CCGE
from dogfighter.algorithms.ccge import CCGEParams
from dogfighter.models.bases import BaseActor
from dogfighter.models.mlp import MlpActor, MlpEnvParams, MlpQUEnsemble
from dogfighter.models.mlp.mlp_bases import MlpModelParams


def train(wm: Wingman) -> None:
    # pull the config out of wingman
    cfg = wm.cfg

    # setup environment and algorithm
    env = setup_environment(wm)
    alg = setup_algorithm(wm)

    # setup replay buffer
    memory = ReplayBuffer(cfg.buffer_size)

    # logging metrics
    wm.log["epoch"] = 0
    wm.log["eval_perf"] = -math.inf
    wm.log["max_eval_perf"] = -math.inf
    next_eval_step = 0

    """START TRAINING"""
    while memory.count <= cfg.total_steps:
        wm.log["epoch"] += 1

        """EVALUATE POLICY"""
        if memory.count >= next_eval_step:
            next_eval_step = (
                int(memory.count / cfg.eval_steps_ratio) + 1
            ) * cfg.eval_steps_ratio
            wm.log["eval_perf"] = evaluate(wm=wm, actor=alg.actor, env=env)
            wm.log["max_eval_perf"] = max(
                [float(wm.log["max_eval_perf"]), float(wm.log["eval_perf"])]
            )

        """ENVIRONMENT ROLLOUT"""
        alg.eval()
        alg.zero_grad()

        with torch.no_grad():
            term, trunc = False, False
            obs, info = env.reset()

            # step for one episode
            while not term and not trunc:
                # compute an action depending on whether we're exploring or not
                if memory.count < cfg.exploration_steps:
                    # sample an action from the env
                    act = env.action_space.sample()
                else:
                    # get an action from the actor
                    # this is a tensor
                    act, _ = alg.actor.sample(
                        *alg.actor(gpuize(obs, wm.device).unsqueeze(0))
                    )

                    # convert the action to cpu, and remove the batch dim
                    act = cpuize(act.squeeze(0))

                # step the transition
                next_obs, rew, term, trunc, info = env.step(act)
                rew = float(rew)

                # store stuff in mem
                memory.push(
                    [obs, next_obs, rew, term, trunc],
                    random_rollover=cfg.random_rollover,
                )

                # new observation is the next observation
                obs = next_obs

            # for logging
            wm.log["episode_cumulative_reward"] = info["episode"]["r"][0]


def evaluate(wm: Wingman, actor: BaseActor | None) -> float:
    # setup the environment and actor
    env = setup_environment(wm)
    actor = actor or setup_algorithm(wm).actor

    # start the evaluation loops
    cumulative_rewards: list[float] = []
    for _ in range(wm.cfg.eval_num_episodes):
        term, trunc = False, False
        obs, info = env.reset()

        # step for one episode
        while not term and not trunc:
            # get an action from the actor
            # this is a tensor
            act, _ = actor.sample(*actor(gpuize(obs, wm.device).unsqueeze(0)))

            # convert the action to cpu, and remove the batch dim
            act = cpuize(act.squeeze(0))

            # step the transition
            next_obs, rew, term, trunc, info = env.step(act)

            # new observation is the next observation
            obs = next_obs

        cumulative_rewards.append(info["episode"]["r"][0])

    return float(np.mean(cumulative_rewards))


def setup_environment(wm: Wingman) -> gym.Env:
    env = gym.make(wm.cfg.env_name)
    wm.cfg.obs_size = env.observation_space.shape[0]  # pyright: ignore[reportOptionalSubscript]
    wm.cfg.act_size = env.action_space.shape[0]  # pyright: ignore[reportOptionalSubscript]

    # wrap some wrappers
    env = record_episode_statistics.RecordEpisodeStatistics(env)

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

    # define the algorithm
    return CCGE(
        actor_type=MlpActor,
        critic_type=MlpQUEnsemble,
        env_params=env_params,
        model_params=model_params,
        algorithm_params=algorithm_params,
        device=torch.device(wm.device),
        jit=not wm.cfg.debug,
    )


if __name__ == "__main__":
    signal(SIGINT, shutdown_handler)
    wm = Wingman(config_yaml="./config.yaml")

    train(wm)
