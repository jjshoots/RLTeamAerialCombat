from __future__ import annotations

from wingman import Wingman

from dogfighter.algorithms.ccge import CCGEConfig
from dogfighter.bases.base_algorithm import AlgorithmConfig
from dogfighter.models.mlp.mlp_actor import MlpActorConfig
from dogfighter.models.mlp.mlp_qu_network import MlpQUNetworkConfig
from dogfighter.models.mlp.simba_actor import SimbaActorConfig
from dogfighter.models.mlp.simba_qu_network import SimbaQUNetworkConfig
from dogfighter.models.transformer.basic_merge_actor import \
    BasicMergeActorConfig
from dogfighter.models.transformer.basic_merge_qu_network import \
    BasicMergeQUNetworkConfig
from dogfighter.models.transformer.prelndecoder_actor import \
    PreLNDecoderActorConfig
from dogfighter.models.transformer.prelndecoder_qu_network import \
    PreLNDecoderQUNetworkConfig


def get_algorithm_config(wm: Wingman) -> AlgorithmConfig:
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
    elif wm.cfg.algorithm.variant == "simba":
        actor_config = SimbaActorConfig(
            obs_size=wm.cfg.algorithm.obs_size,
            act_size=wm.cfg.algorithm.act_size,
            embed_dim=wm.cfg.algorithm.actor.embed_dim,
            num_blocks=wm.cfg.algorithm.actor.num_blocks,
        )
        qu_config = SimbaQUNetworkConfig(
            obs_size=wm.cfg.algorithm.obs_size,
            act_size=wm.cfg.algorithm.act_size,
            embed_dim=wm.cfg.algorithm.critic.embed_dim,
            num_blocks=wm.cfg.algorithm.critic.num_blocks,
        )
    elif wm.cfg.algorithm.variant == "transformer":
        # actor_config = BasicMergeActorConfig(
        #     src_size=wm.cfg.algorithm.src_size,
        #     tgt_size=wm.cfg.algorithm.tgt_size,
        #     act_size=wm.cfg.algorithm.act_size,
        #     embed_dim=wm.cfg.algorithm.actor.embed_dim,
        # )
        # qu_config = BasicMergeQUNetworkConfig(
        #     src_size=wm.cfg.algorithm.src_size,
        #     tgt_size=wm.cfg.algorithm.tgt_size,
        #     act_size=wm.cfg.algorithm.act_size,
        #     embed_dim=wm.cfg.algorithm.critic.embed_dim,
        # )
        actor_config = PreLNDecoderActorConfig(
            src_size=wm.cfg.algorithm.src_size,
            tgt_size=wm.cfg.algorithm.tgt_size,
            act_size=wm.cfg.algorithm.act_size,
            embed_dim=wm.cfg.algorithm.actor.embed_dim,
            ff_dim=wm.cfg.algorithm.actor.ff_dim,
            num_att_heads=wm.cfg.algorithm.actor.num_att_head,
            num_layers=wm.cfg.algorithm.critic.num_layers,
        )
        qu_config = PreLNDecoderQUNetworkConfig(
            src_size=wm.cfg.algorithm.src_size,
            tgt_size=wm.cfg.algorithm.tgt_size,
            act_size=wm.cfg.algorithm.act_size,
            embed_dim=wm.cfg.algorithm.critic.embed_dim,
            ff_dim=wm.cfg.algorithm.critic.ff_dim,
            num_att_heads=wm.cfg.algorithm.critic.num_att_head,
            num_layers=wm.cfg.algorithm.critic.num_layers,
        )
    else:
        raise NotImplementedError

    alg_config = CCGEConfig(
        compile=(not wm.cfg.mode.debug),
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
    )

    return alg_config
