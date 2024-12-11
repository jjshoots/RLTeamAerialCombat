from collections import defaultdict

import numpy as np


class VecEnvInfoAggregator:
    """VecEnvInfoAggregator.

    For aggregating infos from a Gymnasium vector environment.
    By default, the info is a dictionary of np arrays.
    Each `key` in the info will be accompanied by a similar `_{key_name}` item in the info dictionary serving as a mask.
    The mask is `True` where the element in the `info[key]` contains a valid value.

    See https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/vector/vector_env.py#L275
    """

    def __init__(self):
        """__init__."""
        self.info_list: list[dict] = []

    def add_info(self, info: dict[str, np.ndarray]) -> None:
        """add_info.

        Args:
            info (dict[str, np.ndarray]): info

        Returns:
            None:
        """
        self.info_list.append(info)

    def aggregate_info(self) -> tuple[dict[str, float], dict[str, int]]:
        """aggregate_info.

        Args:

        Returns:
            tuple[dict[str, float], dict[str, int]]: averaged infos, info key counts
        """
        info_accumulator = defaultdict(lambda: 0.0)
        info_counts = defaultdict(lambda: 0)

        for info in self.info_list:
            for k in info:
                k: str

                # ignore info mask key
                if k.startswith("_"):
                    continue

                # count the k
                info_counts[k] += np.sum(info[f"_{k}"])

                # sum over valid items in the info
                info_accumulator[k] += np.sum(info[k] * info[f"_{k}"])

        # compute the average over the infos
        info_averages: dict[str, float] = dict()
        for k in info_accumulator:
            info_averages[k] = info_accumulator[k] / info_counts[k]

        return info_averages, info_counts
