from __future__ import annotations

import torch
from wingman import Wingman
from wingman.replay_buffer import ReplayBuffer

from dogfighter.algorithms.ccge import CCGEConfig
from dogfighter.bases.base_algorithm import Algorithm
from dogfighter.bases.base_replay_buffer import ReplayBufferConfig
from dogfighter.models.mlp.mlp_actor import MlpActorConfig
from dogfighter.models.mlp.mlp_qu_network import MlpQUNetworkConfig


def setup_replay_buffer(wm: Wingman) -> ReplayBuffer:
    memory = ReplayBufferConfig(
        mem_size=wm.cfg.buffer_size,
        mode=wm.cfg.replay_buffer_mode,
        device=str(wm.device),
        store_on_device=wm.cfg.replay_buffer_store_on_device,
        random_rollover=wm.cfg.random_rollover,
    ).instantiate()

    return memory


def setup_algorithm(wm: Wingman) -> Algorithm:
    alg = CCGEConfig(
        device=str(wm.device),
        actor_config=MlpActorConfig(
            obs_size=wm.cfg.obs_size,
            act_size=wm.cfg.act_size,
            embed_dim=wm.cfg.embed_dim,
        ),
        qu_config=MlpQUNetworkConfig(
            obs_size=wm.cfg.obs_size,
            act_size=wm.cfg.act_size,
            embed_dim=wm.cfg.embed_dim,
        ),
        target_entropy=(-wm.cfg.act_size),
    ).instantiate()

    if not wm.cfg.mode.debug:
        torch.compile(alg)

    # get latest weight files
    has_weights, model_file, _ = wm.get_weight_files()
    if has_weights:
        # load the model
        alg.load_state_dict(
            torch.load(model_file, map_location=torch.device(wm.device))
        )

    return alg
