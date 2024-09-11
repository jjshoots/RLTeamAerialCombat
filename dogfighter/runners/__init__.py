from typing import Union

from dogfighter.runners.asynchronous.base import AsynchronousRunnerSettings
from dogfighter.runners.synchronous.base import SynchronousRunnerSettings

KnownRunnerSettings = Union[
    SynchronousRunnerSettings,
    AsynchronousRunnerSettings,
]
