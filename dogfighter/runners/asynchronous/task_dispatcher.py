import json
import os
import subprocess
import sys
import tempfile
import time
from multiprocessing import Manager, Process
from multiprocessing.managers import ListProxy
from pathlib import Path
from typing import Generator
from uuid import uuid4

from pydantic import BaseModel, StrictStr

from dogfighter.runners.asynchronous.base import (
    AsynchronousRunnerSettings,
    CollectionResult,
    EvaluationResult,
    ResultDefinition,
    TaskType,
)
from dogfighter.runners.base import ConfigStack


class _RunningTask(BaseModel):
    """_RunningTask."""

    task_type: TaskType
    result_output_path: StrictStr


class TaskDispatcher:
    """A dispatcher that handles submitting collect and eval jobs asynchronously."""

    def __init__(
        self,
        config_stack: ConfigStack,
        kill_on_error: bool = False,
        loop_interval_seconds: float = 1.0,
    ) -> None:
        """A dispatcher that handles submitting collect and eval jobs asynchronously."""
        assert isinstance(config_stack.runner_settings, AsynchronousRunnerSettings)

        # some constants
        self._config_stack = config_stack
        self._kill_on_error = kill_on_error
        self._max_workers = config_stack.runner_settings.max_workers
        self._max_queued_evals = config_stack.runner_settings.max_queued_evals
        self._loop_interval_seconds = loop_interval_seconds

        # runtime paths
        self._work_dir_reference = tempfile.TemporaryDirectory()
        self._work_dir = self._work_dir_reference.name
        self._config_stack_path = f"{self._work_dir}/config_stack.json"
        with open(self._config_stack_path, "w") as f:
            f.write(self._config_stack.model_dump_json())
        self._actor_weights_path = f"{self._work_dir}/actor_weights.pth"

        # runtime variables
        self._running_processes: dict[subprocess.Popen, _RunningTask] = dict()

        # special case for variables that must be shared between processes
        self._manager = Manager()
        self._num_queued_evals = self._manager.Value("i", 0)
        self._completed_tasks: ListProxy[ResultDefinition] = self._manager.list()

        # start the main loop
        Process(target=self._start).start()

    def __del__(self) -> None:
        """Cleanup the working directory."""
        self._work_dir_reference.cleanup()

    @property
    def actor_weights_path(self) -> str:
        return self._actor_weights_path

    @property
    def completed_tasks(
        self,
    ) -> Generator[ResultDefinition, None, None]:
        """List of completed tasks. Once the task is queried, it is deleted."""
        with self._manager.Lock():
            while self._completed_tasks:
                yield self._completed_tasks.pop(0)

    def queue_eval(self) -> None:
        """Queues an eval task."""
        with self._manager.Lock():
            self._num_queued_evals.value = max(
                self._max_queued_evals,
                self._num_queued_evals.value + 1,
            )

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
                # queue as many evals as we need
                with self._manager.Lock():
                    if self._num_queued_evals.value > 0:
                        self._submit_process(mode=TaskType.EVAL)
                        self._num_queued_evals.value -= 1

                # queue collects for the rest of available slots
                self._submit_process(mode=TaskType.COLLECT)

            time.sleep(self._loop_interval_seconds)

    def _submit_process(self, mode: TaskType) -> None:
        """Submits a collect or eval task and appends the task to `self._running_tasks`."""
        result_output_path = f"{self._work_dir}/{uuid4()}_result.json"

        command = []
        command.append(f"{sys.executable}")
        command.append(str(Path(__file__).parent / "runner.py"))
        command.append("--config_file")
        command.append(self._config_stack_path)
        command.append("--task_type")
        command.append("collect" if mode == TaskType.COLLECT else "eval")
        command.append("--actor_weights_path")
        command.append(self._actor_weights_path)
        command.append("--result_output_path")
        command.append(result_output_path)

        # Run the command and add to our queue
        self._running_processes[
            subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        ] = _RunningTask(
            task_type=mode,
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
            # if no result, the task is still running
            if process.poll() is None:
                continue

            # if no error, record the output and break
            if process.returncode == 0:
                with open(task.result_output_path, "r") as f, self._manager.Lock():
                    if task.task_type == TaskType.COLLECT:
                        self._completed_tasks.append(CollectionResult(**json.load(f)))
                    elif task.task_type == TaskType.EVAL:
                        self._completed_tasks.append(EvaluationResult(**json.load(f)))
                    else:
                        raise NotImplementedError

            # if error, handle it
            else:
                stdout, stderr = process.communicate()
                printout = "Subprocess has failed!\n\n" f"{stdout}\n\n" f"{stderr}"

                if self._kill_on_error:
                    raise subprocess.SubprocessError(printout)
                else:
                    print(printout)

            # cleanup
            os.remove(task.result_output_path)
            done_tasks.append(process)

        # cleanup done items
        for task in done_tasks:
            self._running_processes.pop(task)
