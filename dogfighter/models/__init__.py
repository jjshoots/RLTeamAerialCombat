from typing import Union

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
from dogfighter.models.transformer.transformer_actor import TransformerActorConfig
from dogfighter.models.transformer.transformer_qu_network import \
    TransformerQUNetworkConfig

KnownActorConfigs = Union[
    MlpActorConfig,
    SimbaActorConfig,
    BasicMergeActorConfig,
    PreLNDecoderActorConfig,
    TransformerActorConfig,
]

KnownQUNetworkConfigs = Union[
    MlpQUNetworkConfig,
    SimbaQUNetworkConfig,
    BasicMergeQUNetworkConfig,
    PreLNDecoderQUNetworkConfig,
    TransformerQUNetworkConfig,
]
