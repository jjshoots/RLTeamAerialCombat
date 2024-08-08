from __future__ import annotations

import torch
from wingman import Wingman
from wingman.replay_buffer import ReplayBuffer

from dogfighter.algorithms.ccge import CCGEConfig
from dogfighter.bases.base_algorithm import Algorithm
from dogfighter.bases.base_replay_buffer import ReplayBufferConfig
from dogfighter.models.mlp.mlp_actor import MlpActorConfig
from dogfighter.models.mlp.mlp_qu_network import MlpQUNetworkConfig
from dogfighter.models.transformer.transformer_actor import \
    TransformerActorConfig
from dogfighter.models.transformer.transformer_qu_network import \
    TransformerQUNetworkConfig


def setup_replay_buffer(wm: Wingman) -> ReplayBuffer:
    memory = ReplayBufferConfig(
        mem_size=wm.cfg.replay_buffer.size,
        mode=wm.cfg.replay_buffer.mode,
        device=str(wm.device),
        store_on_device=wm.cfg.replay_buffer.store_on_device,
        random_rollover=wm.cfg.replay_buffer.random_rollover,
    ).instantiate()

    return memory


def setup_algorithm(wm: Wingman) -> Algorithm:
    if wm.cfg.model.variant == "mlp":
        actor_config = MlpActorConfig(
            obs_size=wm.cfg.model.obs_size,
            act_size=wm.cfg.model.act_size,
            embed_dim=wm.cfg.actor.embed_dim,
        )
        qu_config = MlpQUNetworkConfig(
            obs_size=wm.cfg.model.obs_size,
            act_size=wm.cfg.model.act_size,
            embed_dim=wm.cfg.model.critic.embed_dim,
        )
    elif wm.cfg.model.variant == "transformer":
        actor_config = TransformerActorConfig(
            src_size=wm.cfg.model.actor.src_size,
            tgt_size=wm.cfg.model.actor.tgt_size,
            act_size=wm.cfg.model.act_size,
            embed_dim=wm.cfg.model.actor.embed_dim,
            ff_dim=wm.cfg.model.actor.ff_dim,
            num_att_heads=wm.cfg.model.actor.num_att_head,
            num_encode_layers=wm.cfg.model.actor.num_encode_layers,
            num_decode_layers=wm.cfg.model.actor.num_decode_layers,
        )
        qu_config = TransformerQUNetworkConfig(
            src_size=wm.cfg.model.critic.src_size,
            tgt_size=wm.cfg.model.critic.tgt_size,
            act_size=wm.cfg.model.act_size,
            embed_dim=wm.cfg.model.critic.embed_dim,
            ff_dim=wm.cfg.model.critic.ff_dim,
            num_att_heads=wm.cfg.model.critic.num_att_head,
            num_encode_layers=wm.cfg.model.critic.num_encode_layers,
            num_decode_layers=wm.cfg.model.critic.num_decode_layers,
        )
    else:
        raise NotImplementedError

    alg = CCGEConfig(
        device=str(wm.device),
        actor_config=actor_config,
        qu_config=qu_config,
        qu_num_ensemble=wm.cfg.model.qu_num_ensemble,
        actor_learning_rate=wm.cfg.model.actor_learning_rate,
        critic_learning_rate=wm.cfg.model.critic_learning_rate,
        alpha_learning_rate=wm.cfg.model.alpha_learning_rate,
        tune_entropy=wm.cfg.model.tune_entropy,
        target_entropy=(-wm.cfg.model.act_size) * wm.cfg.model.target_entropy_gain,
        learn_uncertainty=wm.cfg.model.learn_uncertainty,
        discount_factor=wm.cfg.model.discount_factor,
        actor_update_ratio=wm.cfg.model.actor_update_ratio,
        critic_update_ratio=wm.cfg.model.critic_update_ratio,
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
