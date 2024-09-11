from typing import Union

from dogfighter.env_interactors.mlp_ma_env_interactor import \
    MLPMAEnvInteractorConfig
from dogfighter.env_interactors.mlp_sa_vec_env_interactor import \
    MLPSAEnvInteractorConfig
from dogfighter.env_interactors.transformer_ma_env_interactor import \
    TransformerMAEnvInteractorConfig

KnownInteractorConfigs = Union[
    MLPMAEnvInteractorConfig,
    MLPSAEnvInteractorConfig,
    TransformerMAEnvInteractorConfig,
]
