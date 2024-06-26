from __future__ import annotations

import math
from signal import SIGINT, signal

import gymnasium as gym
import torch
from wingman import ReplayBuffer, Wingman
from wingman.utils import shutdown_handler

from dogfighter.algorithms import CCGE
from dogfighter.algorithms.ccge import CCGEParams
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

        # TODO: evaluation script

        """ENVIRONMENT ROLLOUT"""
        alg.eval()
        alg.zero_grad()

        with torch.no_grad():
            term, trunc = False, False
            obs, _ = env.reset()

            # step for one episode
            while not term and not trunc:
                # compute an action depending on whether we're exploring or not
                if memory.count < cfg.exploration_steps:
                    act = env.action_space.sample()
                else:
                    # TODO: maybe GPU this?
                    t_obs = torch.tensor(obs)
                    act = alg.actor(t_obs.unsqueeze(0)).squeeze(0)
                    act = act.numpy()

                # step the transition
                next_obs, rew, term, trunc, _ = env.step(act)
                rew = float(rew)

                # store stuff in mem
                memory.push(
                    [obs, next_obs, rew, term, trunc],
                    random_rollover=cfg.random_rollover,
                )

                # new observation is the next observation
                obs = next_obs


def setup_environment(wm: Wingman) -> gym.Env:
    env = gym.make(wm.cfg.env_name)
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

    # define the algorithm
    return CCGE(
        actor_type=MlpActor,
        critic_type=MlpQUEnsemble,
        env_params=env_params,
        model_params=model_params,
        algorithm_params=algorithm_params,
        jit=not wm.cfg.debug,
    )


if __name__ == "__main__":
    signal(SIGINT, shutdown_handler)
    wm = Wingman(config_yaml="./config.yaml")

    train(wm)
