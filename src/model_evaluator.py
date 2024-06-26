import gymnasium as gym
import numpy as np
from wingman import Wingman
from wingman.utils import cpuize, gpuize

from dogfighter.models.bases import BaseActor


def evaluate_model(wm: Wingman, actor: BaseActor, env: gym.Env) -> float:
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

    return np.mean(cumulative_rewards)
