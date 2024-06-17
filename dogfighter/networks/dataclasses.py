from pydantic import BaseModel, Field, StrictFloat, StrictInt


class EnvParams(BaseModel):
    obs_size: StrictInt
    att_size: StrictInt
    act_size: StrictInt


class ModelParams(BaseModel):
    qu_num_ensemble: StrictInt = Field(2)
    embed_dim: StrictInt = Field(128)
    att_inner_dim: StrictInt = Field(256)
    att_num_heads: StrictInt = Field(2)
    att_num_encoder_layers: StrictInt = Field(2)
    att_num_decoder_layers: StrictInt = Field(2)


class LearningParams(BaseModel):
    learning_rate: StrictFloat = Field(0.003)
    alpha_learning_rate: StrictFloat = Field(0.01)
    target_entropy: None | StrictFloat = Field(None)
    discount_factor: StrictFloat = Field(0.99)
    actor_update_ratio: StrictInt = Field(1)
    critic_update_ratio: StrictInt = Field(1)
