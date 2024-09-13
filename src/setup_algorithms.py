from wingman import Wingman

from dogfighter.algorithms import KnownAlgorithmConfigs
from dogfighter.algorithms.ccge import CCGEConfig
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
from dogfighter.models.transformer.transformer_actor import \
    TransformerActorConfig
from dogfighter.models.transformer.transformer_qu_network import \
    TransformerQUNetworkConfig


def get_algorithm_config(wm: Wingman) -> KnownAlgorithmConfigs:
    if wm.cfg.algorithm.variant == "ccge":
        if wm.cfg.algorithm.actor.variant == "mlp":
            actor_config = MlpActorConfig(
                obs_size=wm.cfg.algorithm.obs_size,
                act_size=wm.cfg.algorithm.act_size,
                embed_dim=wm.cfg.algorithm.actor.embed_dim,
            )
        elif wm.cfg.algorithm.actor.variant == "simba":
            actor_config = SimbaActorConfig(
                obs_size=wm.cfg.algorithm.obs_size,
                act_size=wm.cfg.algorithm.act_size,
                embed_dim=wm.cfg.algorithm.actor.embed_dim,
                num_blocks=wm.cfg.algorithm.actor.num_blocks,
            )
        elif wm.cfg.algorithm.actor.variant == "basic_merge":
            actor_config = BasicMergeActorConfig(
                src_size=wm.cfg.algorithm.src_size,
                tgt_size=wm.cfg.algorithm.tgt_size,
                act_size=wm.cfg.algorithm.act_size,
                embed_dim=wm.cfg.algorithm.actor.embed_dim,
            )
        elif wm.cfg.algorithm.actor.variant == "pre_ln_decoder":
            actor_config = PreLNDecoderActorConfig(
                src_size=wm.cfg.algorithm.src_size,
                tgt_size=wm.cfg.algorithm.tgt_size,
                act_size=wm.cfg.algorithm.act_size,
                embed_dim=wm.cfg.algorithm.actor.embed_dim,
                ff_dim=wm.cfg.algorithm.actor.ff_dim,
                num_att_heads=wm.cfg.algorithm.actor.num_att_head,
                num_layers=wm.cfg.algorithm.critic.num_layers,
            )
        elif wm.cfg.algorithm.actor.variant == "transformer":
            actor_config = TransformerActorConfig(
                src_size=wm.cfg.algorithm.src_size,
                tgt_size=wm.cfg.algorithm.tgt_size,
                act_size=wm.cfg.algorithm.act_size,
                embed_dim=wm.cfg.algorithm.actor.embed_dim,
                ff_dim=wm.cfg.algorithm.actor.ff_dim,
                num_att_heads=wm.cfg.algorithm.actor.num_att_head,
                num_encode_layers=wm.cfg.algorithm.critic.num_encode_layers,
                num_decode_layers=wm.cfg.algorithm.critic.num_decode_layers,
            )
        else:
            raise NotImplementedError

        if wm.cfg.algorithm.critic.variant == "mlp":
            qu_config = MlpQUNetworkConfig(
                obs_size=wm.cfg.algorithm.obs_size,
                act_size=wm.cfg.algorithm.act_size,
                embed_dim=wm.cfg.algorithm.critic.embed_dim,
            )
        elif wm.cfg.algorithm.critic.variant == "simba":
            qu_config = SimbaQUNetworkConfig(
                obs_size=wm.cfg.algorithm.obs_size,
                act_size=wm.cfg.algorithm.act_size,
                embed_dim=wm.cfg.algorithm.critic.embed_dim,
                num_blocks=wm.cfg.algorithm.critic.num_blocks,
            )
        elif wm.cfg.algorithm.critic.variant == "basic_merge":
            qu_config = BasicMergeQUNetworkConfig(
                src_size=wm.cfg.algorithm.src_size,
                tgt_size=wm.cfg.algorithm.tgt_size,
                act_size=wm.cfg.algorithm.act_size,
                embed_dim=wm.cfg.algorithm.critic.embed_dim,
            )
        elif wm.cfg.algorithm.critic.variant == "pre_ln_decoder":
            qu_config = PreLNDecoderQUNetworkConfig(
                src_size=wm.cfg.algorithm.src_size,
                tgt_size=wm.cfg.algorithm.tgt_size,
                act_size=wm.cfg.algorithm.act_size,
                embed_dim=wm.cfg.algorithm.critic.embed_dim,
                ff_dim=wm.cfg.algorithm.critic.ff_dim,
                num_att_heads=wm.cfg.algorithm.critic.num_att_head,
                num_layers=wm.cfg.algorithm.critic.num_layers,
            )
        elif wm.cfg.algorithm.actor.variant == "transformer":
            qu_config = TransformerQUNetworkConfig(
                src_size=wm.cfg.algorithm.src_size,
                tgt_size=wm.cfg.algorithm.tgt_size,
                act_size=wm.cfg.algorithm.act_size,
                embed_dim=wm.cfg.algorithm.actor.embed_dim,
                ff_dim=wm.cfg.algorithm.actor.ff_dim,
                num_att_heads=wm.cfg.algorithm.actor.num_att_head,
                num_encode_layers=wm.cfg.algorithm.critic.num_encode_layers,
                num_decode_layers=wm.cfg.algorithm.critic.num_decode_layers,
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
            target_smoothing_coefficient=wm.cfg.algorithm.target_smoothing_coefficient,
            tune_entropy=wm.cfg.algorithm.tune_entropy,
            target_entropy=(
                -wm.cfg.algorithm.act_size * wm.cfg.algorithm.target_entropy_gain
            ),
            learn_uncertainty=wm.cfg.algorithm.learn_uncertainty,
            discount_factor=wm.cfg.algorithm.discount_factor,
            actor_update_ratio=wm.cfg.algorithm.actor_update_ratio,
            critic_update_ratio=wm.cfg.algorithm.critic_update_ratio,
        )
    else:
        raise NotImplementedError

    return alg_config
