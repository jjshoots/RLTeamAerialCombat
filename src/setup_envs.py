from wingman import Wingman

from dogfighter.envs.ma_envs import KnownMAEnvConfigs
from dogfighter.envs.ma_envs.pyflyt_ma_envs import PyFlytMAEnvConfig
from dogfighter.envs.sa_envs import KnownSAEnvConfigs
from dogfighter.envs.sa_envs.dmc_sa_env import DMCSAEnvConfig
from dogfighter.envs.sa_envs.pyflyt_sa_envs import PyFlytSAEnvConfig


def get_mlp_sa_env_config(wm: Wingman) -> KnownSAEnvConfigs:
    if wm.cfg.env.id.startswith("PyFlyt"):
        env_config = PyFlytSAEnvConfig(
            env_id=wm.cfg.env.id,
            render_mode="human" if wm.cfg.mode.display or wm.cfg.mode.render else None,
            env_kwargs=vars(wm.cfg.env.kwargs) if hasattr(wm.cfg.env, "kwargs") else {},
        )
    elif wm.cfg.env.id.startswith("dm_control"):
        env_config = DMCSAEnvConfig(
            env_id=wm.cfg.env.id,
            render_mode="human" if wm.cfg.mode.display or wm.cfg.mode.render else None,
            env_kwargs=vars(wm.cfg.env.kwargs) if hasattr(wm.cfg.env, "kwargs") else {},
        )
    else:
        raise NotImplementedError(wm.cfg.env.id)

    # record observation and action space shapes if needed
    _obs = getattr(wm.cfg.algorithm, "obs_size", None)
    _act = getattr(wm.cfg.algorithm, "act_size", None)
    if not _obs or not _act:
        dummy_env = env_config.instantiate()
        if not _obs:
            wm.cfg.algorithm.obs_size = dummy_env.observation_space.shape[0]  # pyright: ignore[reportOptionalSubscript]
        if not _act:
            wm.cfg.algorithm.act_size = dummy_env.action_space.shape[0]  # pyright: ignore[reportOptionalSubscript]
        dummy_env.close()

    return env_config


def get_mlp_ma_env_config(wm: Wingman) -> KnownMAEnvConfigs:
    env_config = PyFlytMAEnvConfig(
        env_id=wm.cfg.env.id,
        render_mode="human" if wm.cfg.mode.display or wm.cfg.mode.render else None,
        env_kwargs=vars(wm.cfg.env.kwargs) if hasattr(wm.cfg.env, "kwargs") else {},
    )

    # record observation and action space shapes if needed
    _obs = getattr(wm.cfg.algorithm, "obs_size", None)
    _act = getattr(wm.cfg.algorithm, "act_size", None)
    if not _obs or not _act:
        dummy_env = env_config.instantiate()
        if not _obs:
            wm.cfg.algorithm.obs_size = dummy_env.observation_space(0).shape[0]  # pyright: ignore[reportOptionalSubscript]
        if not _act:
            wm.cfg.algorithm.act_size = dummy_env.action_space(0).shape[0]  # pyright: ignore[reportOptionalSubscript]
        dummy_env.close()

    return env_config


def get_transformer_ma_env_config(wm: Wingman) -> KnownMAEnvConfigs:
    env_config = PyFlytMAEnvConfig(
        env_id=wm.cfg.env.id,
        render_mode="human" if wm.cfg.mode.display or wm.cfg.mode.render else None,
        env_kwargs=vars(wm.cfg.env.kwargs) if hasattr(wm.cfg.env, "kwargs") else {},
    )

    # record observation and action space shapes
    _src = getattr(wm.cfg.algorithm, "src_size", None)
    _tgt = getattr(wm.cfg.algorithm, "tgt_size", None)
    _act = getattr(wm.cfg.algorithm, "act_size", None)
    if not _src or not _tgt or not _act:
        dummy_env = env_config.instantiate()
        if not _src:
            wm.cfg.algorithm.src_size = dummy_env.observation_space(0)[  # pyright: ignore[reportIndexIssue]
                "src"
            ].feature_space.shape[0]
        if not _tgt:
            wm.cfg.algorithm.tgt_size = dummy_env.observation_space(0)[  # pyright: ignore[reportIndexIssue]
                "tgt"
            ].feature_space.shape[0]
        if not _act:
            wm.cfg.algorithm.act_size = dummy_env.action_space(0).shape[0]  # pyright: ignore[reportOptionalSubscript]
        dummy_env.close()

    return env_config
