from __future__ import annotations

from signal import SIGINT, signal

from wingman import Wingman
from wingman.utils import shutdown_handler

from ma_env_interaction_utils import ma_env_evaluate
from setup_utils import (setup_algorithm, setup_ma_environment,
                         setup_vector_environment)
from train_utils import train
from vec_env_interaction_utils import vec_env_evaluate, vec_env_render_gif

if __name__ == "__main__":
    signal(SIGINT, shutdown_handler)
    # wm = Wingman(config_yaml="./configs/quad_dogfight_config.yaml")
    wm = Wingman(config_yaml="./configs/dual_dogfight_config.yaml")

    if wm.cfg.train:
        train(wm)

    elif wm.cfg.display:
        if wm.cfg.env_type == "ma_env":
            info = ma_env_evaluate(
                env=setup_ma_environment(wm),
                actor=setup_algorithm(wm).actor,
                num_episodes=1,
            )
        elif wm.cfg.env_type == "vec_env":
            wm.cfg.num_envs = 1 if wm.cfg.display else wm.cfg.num_envs
            wm.log["eval_perf"], wm.log["mean_episode_length"] = vec_env_evaluate(
                env=setup_vector_environment(wm),
                actor=setup_algorithm(wm).actor,
                num_episodes=wm.cfg.eval_num_episodes,
            )
        else:
            raise ValueError(
                f"Expected only 'vec_env' and 'ma_env' for env_type, got '{wm.cfg.env_type}'."
            )

    elif wm.cfg.render:
        if wm.cfg.env_type == "vec_env":
            print(vec_env_render_gif(wm=wm))

    else:
        print("So this is life now.")
