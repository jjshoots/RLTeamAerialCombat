import argparse
import json
from signal import SIGINT, signal

from wingman import Wingman
from wingman.utils import shutdown_handler

from dogfighter.runners.asynchronous.base import AsynchronousRunnerSettings
from dogfighter.runners.asynchronous.trainer import run_train
from dogfighter.runners.asynchronous.workers import (run_collection,
                                                     run_evaluation)
from dogfighter.runners.base import ConfigStack

signal(SIGINT, shutdown_handler)


def run_asynchronous(
    wm: Wingman,
    configs: ConfigStack,
) -> None:
    # assert some things
    assert isinstance(configs.runner_settings, AsynchronousRunnerSettings)

    # depending on the mode, run eval or collect
    if wm.cfg.runner.mode == "trainer":
        run_train(wm=wm, configs=configs)
    elif wm.cfg.runner.mode == "worker":
        if wm.cfg.runner.worker.task == "collect":
            run_collection(configs=configs)
        elif wm.cfg.runner.worker.task == "eval":
            run_evaluation(configs=configs)
        else:
            raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_file", required=True)
    task_file = parser.parse_args().task_file

    # read in the current task file
    with open(task_file, "r") as f:
        configs = ConfigStack(**json.load(f))

    # assert some things
    assert isinstance(configs.runner_settings, AsynchronousRunnerSettings)

    if configs.runner_settings.mode == "worker":
        if configs.runner_settings.worker_settings.task == "collect":
            run_collection(configs=configs)
        elif configs.runner_settings.worker_settings.task == "eval":
            run_evaluation(configs=configs)
        else:
            raise NotImplementedError
