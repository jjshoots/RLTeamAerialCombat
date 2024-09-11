from pydantic import BaseModel


class SynchronousRunnerSettings(BaseModel):
    """SynchronousRunnerSettings."""

    max_transitions: int
    transitions_per_epoch: int
    transitions_num_exploration: int
    transitions_min_for_train: int

    eval_num_episodes: int
    eval_transitions_frequency: int
