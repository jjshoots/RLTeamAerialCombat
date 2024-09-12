from dataclasses import dataclass, field
import json
from multiprocessing import Manager, Process
from multiprocessing.managers import ListProxy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Generator

from pydantic import BaseModel, StrictBool, StrictFloat, StrictStr
from dogfighter.runners.asynchronous.base import AsynchronousRunnerSettings, CollectionResult, EvaluationResult, WorkerTaskType
from dogfighter.runners.base import ConfigStack
from uuid import uuid4


class _RunningTask(BaseModel):
    """_RunningTask."""

    task_type: WorkerTaskType
    task_file: StrictStr
    result_output_path: StrictStr


@dataclass
class TaskDispatcherConfig:

    config_stack: ConfigStack
    kill_on_error: StrictBool = True
    loop_interval_seconds: StrictFloat = 1.0
    _work_dir_reference: tempfile.TemporaryDirectory[str] = field(default_factory=tempfile.TemporaryDirectory)


    def __del__(self) -> None:
        """Cleanup the working directory."""
        self._work_dir_reference.cleanup()

    @property
    def work_dir(self) -> str:
        return self._work_dir_reference.name

    @property
    def actor_weights_path(self) -> str:
        """This is where the actor weights should be saved. It may not actually exist here."""
        return f"{self.work_dir}/actor_weights.pth"

    def instantiate(self) -> "TaskDispatcher":
        """Starts the TaskDispatcher in a separate process from the main one."""
        return TaskDispatcher(self)


class TaskDispatcher:
    """A dispatcher that handles submitting collect and eval jobs asynchronously."""
    def __init__(self, config: TaskDispatcherConfig) -> None:
        """A dispatcher that handles submitting collect and eval jobs asynchronously."""
        runner_settings = config.config_stack.runner_settings
        assert isinstance(runner_settings, AsynchronousRunnerSettings)

        # some constants
        self.config = config
        self.actor_weights_path = config.actor_weights_path
        self._config_stack = config.config_stack
        self._kill_on_error = config.kill_on_error
        self._max_workers = runner_settings.max_workers
        self._loop_interval_seconds = config.loop_interval_seconds
        self._work_dir = config.work_dir

        # runtime variables
        self._num_requested_evals: int = 0
        self._running_processes: dict[subprocess.Popen, _RunningTask] = dict()

        # special case for completed tasks since we need to query it
        self._manager = Manager()
        self._completed_tasks: ListProxy[CollectionResult | EvaluationResult] = self._manager.list()

        Process(target=self._start).start()

    @property
    def completed_tasks(self) -> Generator[CollectionResult | EvaluationResult, None, None]:
        """List of completed tasks. Once the task is queried, it is deleted."""
        while self._completed_tasks:
            yield self._completed_tasks.pop(0)

    @property
    def _active_actor_weights_path(self) -> str:
        """This may be '' or an actual path, depending on whether the weights exist."""
        if os.path.exists(self.config.actor_weights_path):
            return self.config.actor_weights_path
        else:
            return ""

    def queue_eval(self) -> None:
        """Queues an eval task."""
        self._num_requested_evals += 1

    def _start(self) -> None:
        """Starts the task dispatcher in an infinite loop.

        This first checks for any completed jobs.
        Then, it dispatches tasks

        It first checks if there are idle workers.
        If there are, it tries to schedule eval tasks first.
        If there are no eval tasks, it fills up the rest with collect tasks.
        """
        while True:
            self._query_processes()

            while len(self._running_processes) < self._max_workers:
                if self._num_requested_evals > 0:
                    self._submit_process(mode=WorkerTaskType.EVAL)
                    self._num_requested_evals -= 1

                self._submit_process(mode=WorkerTaskType.COLLECT)

            time.sleep(self._loop_interval_seconds)

    def _submit_process(self, mode: WorkerTaskType) -> None:
        """Submits a collect or eval task and appends the task to `self._running_tasks`."""
        # define the place we define the task and submit results
        task_file = f"{self._work_dir}/{uuid4()}_task.json"
        result_output_path = f"{self._work_dir}/{uuid4()}_result.json"

        # patch the config
        task_config = self._config_stack.model_dump()
        task_config["runner_settings"]["mode"] = "worker"
        task_config["runner_settings"]["worker"]["io"]["actor_weights_path"] = (
            self.actor_weights_path if os.path.exists(self.actor_weights_path) else ""
        )
        task_config["runner_settings"]["worker"]["io"]["result_output_path"] = (
            result_output_path
        )
        if mode == WorkerTaskType.COLLECT:
            task_config["runner_settings"]["worker"]["task"] = "collect"
        elif mode == WorkerTaskType.EVAL:
            task_config["runner_settings"]["worker"]["task"] = "eval"
        else:
            raise NotImplementedError

        # dump the file to disk and run the command
        with open(task_file, "w") as f:
            f.write(json.dumps(task_config))

        command = []
        command.append(f"{sys.executable}")
        command.append(str(Path(__file__).parent / "runner.py"))
        command.append("--task_file")
        command.append(task_file)

        # Run the command and add to our queue
        self._running_processes[
            subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        ] = _RunningTask(
            task_type=mode,
            task_file=task_file,
            result_output_path=result_output_path,
        )

    def _query_processes(self) -> None:
        """Queries each running task and parses the outputs.

        The outputs are in `self.completed_tasks` as either:
        - CollectionResult
        - EvaluationResult

        """
        # for cleanup later
        done_tasks: list[subprocess.Popen] = []

        # iterate through each task and handle done items
        for process, task in self._running_processes.items():
            result = process.poll()

            # if no result, the task is still running
            if result is None:
                continue

            # if no error, record the output and break
            if result == 0:
                with open(task.result_output_path, "r") as f:
                    if task.task_type == WorkerTaskType.COLLECT:
                        self._completed_tasks.append(
                            CollectionResult(**json.load(f))
                        )
                    elif task.task_type == WorkerTaskType.EVAL:
                        self._completed_tasks.append(
                            EvaluationResult(**json.load(f))
                        )
                    else:
                        raise NotImplementedError

                # cleanup
                os.remove(task.task_file)
                os.remove(task.result_output_path)
                done_tasks.append(process)
                continue

            # if error, handle it
            else:
                stdout = process.stdout
                stderr = process.stdout
                if stdout is not None:
                    stdout = stdout.read().decode()
                if stderr is not None:
                    stderr = stderr.read().decode()
                printout = (
                    "Subprocess has failed!\n\n"
                    f"{stdout}\n\n"
                    f"{stderr}"
                )

                if self._kill_on_error:
                    raise subprocess.SubprocessError(printout)
                else:
                    print(printout)

        # cleanup done items
        for task in done_tasks:
            self._running_processes.pop(task)
