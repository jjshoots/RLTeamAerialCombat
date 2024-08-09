from __future__ import annotations

import torch
from wingman import Wingman
from wingman.replay_buffer import ReplayBuffer

from dogfighter.algorithms.ccge import CCGEConfig
from dogfighter.bases.base_algorithm import Algorithm
from dogfighter.bases.base_replay_buffer import ReplayBufferConfig
from dogfighter.models.mlp.mlp_actor import MlpActorConfig
from dogfighter.models.mlp.mlp_qu_network import MlpQUNetworkConfig
from dogfighter.models.transformer.prelndecoder_actor import \
    PreLNDecoderActorConfig
from dogfighter.models.transformer.prelndecoder_qu_network import \
    PreLNDecoderQUNetworkConfig


def setup_replay_buffer(wm: Wingman) -> ReplayBuffer:
    return ReplayBufferConfig(
        mem_size=wm.cfg.replay_buffer.mem_size,
        mode=wm.cfg.replay_buffer.mode,
        device=str(wm.device),
        use_dict_wrapper=wm.cfg.replay_buffer.use_dict_wrapper,
        store_on_device=wm.cfg.replay_buffer.store_on_device,
        random_rollover=wm.cfg.replay_buffer.random_rollover,
    ).instantiate()


def setup_algorithm(wm: Wingman) -> Algorithm:
    if wm.cfg.algorithm.variant == "mlp":
        actor_config = MlpActorConfig(
            obs_size=wm.cfg.algorithm.obs_size,
            act_size=wm.cfg.algorithm.act_size,
            embed_dim=wm.cfg.algorithm.actor.embed_dim,
        )
        qu_config = MlpQUNetworkConfig(
            obs_size=wm.cfg.algorithm.obs_size,
            act_size=wm.cfg.algorithm.act_size,
            embed_dim=wm.cfg.algorithm.critic.embed_dim,
        )
    elif wm.cfg.algorithm.variant == "transformer":
        actor_config = PreLNDecoderActorConfig(
            src_size=wm.cfg.algorithm.src_size,
            tgt_size=wm.cfg.algorithm.tgt_size,
            act_size=wm.cfg.algorithm.act_size,
            embed_dim=wm.cfg.algorithm.actor.embed_dim,
            ff_dim=wm.cfg.algorithm.actor.ff_dim,
            num_tgt_context=wm.cfg.algorithm.actor.num_tgt_context,
            num_att_heads=wm.cfg.algorithm.actor.num_att_head,
            num_layers=wm.cfg.algorithm.critic.num_layers,
        )
        qu_config = PreLNDecoderQUNetworkConfig(
            src_size=wm.cfg.algorithm.src_size,
            tgt_size=wm.cfg.algorithm.tgt_size,
            act_size=wm.cfg.algorithm.act_size,
            embed_dim=wm.cfg.algorithm.critic.embed_dim,
            ff_dim=wm.cfg.algorithm.critic.ff_dim,
            num_tgt_context=wm.cfg.algorithm.actor.num_tgt_context,
            num_att_heads=wm.cfg.algorithm.critic.num_att_head,
            num_layers=wm.cfg.algorithm.critic.num_layers,
        )
    else:
        raise NotImplementedError

    alg = CCGEConfig(
        device=str(wm.device),
        actor_config=actor_config,
        qu_config=qu_config,
        qu_num_ensemble=wm.cfg.algorithm.qu_num_ensemble,
        batch_size=wm.cfg.algorithm.batch_size,
        grad_steps_per_update=wm.cfg.algorithm.grad_steps_per_update,
        actor_learning_rate=wm.cfg.algorithm.actor_learning_rate,
        critic_learning_rate=wm.cfg.algorithm.critic_learning_rate,
        alpha_learning_rate=wm.cfg.algorithm.alpha_learning_rate,
        tune_entropy=wm.cfg.algorithm.tune_entropy,
        target_entropy=(-wm.cfg.algorithm.act_size)
        * wm.cfg.algorithm.target_entropy_gain,
        learn_uncertainty=wm.cfg.algorithm.learn_uncertainty,
        discount_factor=wm.cfg.algorithm.discount_factor,
        actor_update_ratio=wm.cfg.algorithm.actor_update_ratio,
        critic_update_ratio=wm.cfg.algorithm.critic_update_ratio,
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
