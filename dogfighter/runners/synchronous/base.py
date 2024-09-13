from pydantic import BaseModel


class SynchronousRunnerSettings(BaseModel):
    """SynchronousRunnerSettings."""

    transitions_max: int
    transitions_per_epoch: int
    transitions_num_exploration: int
    transitions_min_for_train: int
    transitions_eval_frequency: int
    eval_num_episodes: int
