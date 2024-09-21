import argparse
import json
from signal import SIGINT, signal

from wingman import Wingman
from wingman.utils import shutdown_handler

from dogfighter.runners.asynchronous.base import AsynchronousRunnerSettings
from dogfighter.runners.asynchronous.trainer import run_train
from dogfighter.runners.asynchronous.workers import run_collection, run_evaluation
from dogfighter.runners.base import ConfigStack
from dogfighter.runners.utils import AtomicFileWriter

signal(SIGINT, shutdown_handler)


def run_asynchronous(
    wm: Wingman,
    configs: ConfigStack,
) -> None:
    """THIS ENTRY POINT IS FOR TRAINER ONLY!!!"""
    # assert some things
    assert isinstance(configs.runner_settings, AsynchronousRunnerSettings)
    run_train(wm=wm, configs=configs)


if __name__ == "__main__":
    """THIS ENTRY POINT IS FOR WORKERS ONLY!!!"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--task_type", required=True)
    parser.add_argument("--actor_weights_path", required=True)
    parser.add_argument("--result_output_path", required=True)
    args = parser.parse_args()

    # read in the current task file
    with open(args.config_file, "r") as f:
        configs = ConfigStack(**json.load(f))
    assert isinstance(configs.runner_settings, AsynchronousRunnerSettings)

    if args.task_type == "collect":
        result =run_collection(
            configs=configs,
            actor_weights_path=args.actor_weights_path,
        )
    elif args.task_type == "eval":
        result = run_evaluation(
            configs=configs,
            actor_weights_path=args.actor_weights_path,
        )
    else:
        raise NotImplementedError

    # dump the pointer to disk
    assert args.result_output_path is not None
    with AtomicFileWriter(args.result_output_path) as f:
        with open(f, "w") as fw:
            json.dump(result.model_dump(), fw)
