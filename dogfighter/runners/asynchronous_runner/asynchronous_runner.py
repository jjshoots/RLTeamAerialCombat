import argparse

from wingman import Wingman

from dogfighter.runners.asynchronous_runner.workers import (run_collection,
                                                            run_evaluation)


def main() -> None:
    # grab the config_path from the CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    args, _ = parser.parse_known_args()
    config_yaml = args.config_path
    wm = Wingman(config_yaml=config_yaml)

    # assert that we are async
    assert wm.cfg.runner.variant == "async"

    # depending on the mode, run eval or collect
    if wm.cfg.runner.mode == "trainer":
        pass
    elif wm.cfg.runner.mode == "worker":
        if wm.cfg.runner.worker.task == "collect":
            run_collection(wm)
        elif wm.cfg.runner.worker.task == "eval":
            run_evaluation(wm)
        else:
            raise NotImplementedError
