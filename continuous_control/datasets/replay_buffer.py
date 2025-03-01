import os
import pickle
from typing import Optional

import gym
import numpy as np

from continuous_control.datasets.dataset import Batch, Dataset


class ReplayBuffer(Dataset):
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_dim: int,
        capacity: int,
        n_parts: int = 4,
    ):

        observations = np.empty(
            (capacity, *observation_space.shape), dtype=observation_space.dtype
        )
        actions = np.empty((capacity, action_dim), dtype=np.float32)
        rewards = np.empty((capacity,), dtype=np.float32)
        masks = np.empty((capacity,), dtype=np.float32)
        dones_float = np.empty((capacity,), dtype=np.float32)
        next_observations = np.empty(
            (capacity, *observation_space.shape), dtype=observation_space.dtype
        )
        super().__init__(
            observations=observations,
            actions=actions,
            rewards=rewards,
            masks=masks,
            dones_float=dones_float,
            next_observations=next_observations,
            size=0,
        )

        self.size = 0

        self.insert_index = 0
        self.capacity = capacity

        # for saving the buffer
        self.n_parts = n_parts
        assert self.capacity % self.n_parts == 0

    def sample_last_k(self, k: int) -> Batch:
        assert self.size == self.capacity, "Need to be full for this."
        return Batch(
            observations=self.observations[-k:],
            actions=self.actions[-k:],
            rewards=self.rewards[-k:],
            next_observations=self.next_observations[-k:],
            masks=self.masks[-k:],
        )

    def initialize_with_dataset(self, dataset: Dataset, num_samples: Optional[int]):
        assert (
            self.insert_index == 0
        ), "Can insert a batch online in an empty replay buffer."

        dataset_size = len(dataset.observations)

        if num_samples is None:
            num_samples = dataset_size
        else:
            num_samples = min(dataset_size, num_samples)
        assert (
            self.capacity >= num_samples
        ), "Dataset cannot be larger than the replay buffer capacity."

        if num_samples < dataset_size:
            perm = np.random.permutation(dataset_size)
            indices = perm[:num_samples]
        else:
            indices = np.arange(num_samples)

        self.observations[:num_samples] = dataset.observations[indices]
        self.actions[:num_samples] = dataset.actions[indices]
        self.rewards[:num_samples] = dataset.rewards[indices]
        self.masks[:num_samples] = dataset.masks[indices]
        self.dones_float[:num_samples] = dataset.dones_float[indices]
        self.next_observations[:num_samples] = dataset.next_observations[indices]

        self.insert_index = num_samples
        self.size = num_samples

    def insert(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        mask: float,
        done_float: float,
        next_observation: np.ndarray,
    ):
        self.observations[self.insert_index] = observation
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        self.masks[self.insert_index] = mask
        self.dones_float[self.insert_index] = done_float
        self.next_observations[self.insert_index] = next_observation

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def save_so_far(self, data_dir: str):
        # because of memory limits, we will dump the buffer (SO FAR) into multiple files.
        os.makedirs(data_dir, exist_ok=True)

        assert (
            self.size != self.capacity
        ), "we will only be using this when the buffer isn't full."
        assert (
            self.size == self.insert_index
        ), "yet another sanity check to make sure the buffer is actually not full."

        assert self.size % self.n_parts == 0

        chunk_size = self.size // self.n_parts

        for i in range(self.n_parts):
            data_chunk = [
                self.observations[i * chunk_size : (i + 1) * chunk_size],
                self.actions[i * chunk_size : (i + 1) * chunk_size],
                self.rewards[i * chunk_size : (i + 1) * chunk_size],
                self.masks[i * chunk_size : (i + 1) * chunk_size],
                self.dones_float[i * chunk_size : (i + 1) * chunk_size],
                self.next_observations[i * chunk_size : (i + 1) * chunk_size],
            ]

            # set up the proper data path chunk.
            data_chunk_name = f"chunk_{i}_before_full.pkl"
            with open(os.path.join(data_dir, data_chunk_name), "wb") as f:
                pickle.dump(data_chunk, f)

    def save(self, data_dir: str, round: int = 1):
        # because of memory limits, we will dump the buffer into multiple files
        os.makedirs(data_dir, exist_ok=True)
        chunk_size = self.capacity // self.n_parts

        for i in range(self.n_parts):
            data_chunk = [
                self.observations[i * chunk_size : (i + 1) * chunk_size],
                self.actions[i * chunk_size : (i + 1) * chunk_size],
                self.rewards[i * chunk_size : (i + 1) * chunk_size],
                self.masks[i * chunk_size : (i + 1) * chunk_size],
                self.dones_float[i * chunk_size : (i + 1) * chunk_size],
                self.next_observations[i * chunk_size : (i + 1) * chunk_size],
            ]

            # set up the proper data path chunk.
            data_chunk_name = f"chunk_{i}_when_full_round_{round}.pkl"
            with open(os.path.join(data_dir, data_chunk_name), "wb") as f:
                pickle.dump(data_chunk, f)

    def load(self, data_dir: str):
        chunk_size = self.capacity // self.n_parts
        total_size = 0

        # assert (
        #     len(os.listdir(data_dir)) == self.n_parts
        # ), "weird if not true, although we need to set params right here to confirm it"

        # we have a set of the following:
        # - `chunk_X_before_full_round_X.pkl` -- this we can just sort normally, as this only happens once.
        # - `chunk_Y_when_full_round_X.pkl`

        chunk_paths_before_full = sorted(
            [path for path in os.listdir(data_dir) if "before" in path]
        )
        chunk_paths_when_full = []

        # sort chunk_paths_when_full
        def round_number(path: str) -> int:
            start_idx = path.index("round_")
            end_idx = path.index(".")
            return int(path[start_idx + 6 : end_idx])

        max_rounds = max([round_number(path) for path in os.listdir(data_dir)])
        for round in range(1, max_rounds + 1):
            paths_for_round = sorted(
                [path for path in os.listdir(data_dir) if round_number(path) == round]
            )
            chunk_paths_when_full += paths_for_round

        sorted_chunk_paths = chunk_paths_before_full + chunk_paths_when_full

        curr_idx = 0
        for _, chunk_path in enumerate(sorted_chunk_paths):
            data_path_chunk = os.path.join(data_dir, chunk_path)
            data_chunk = pickle.load(open(data_path_chunk, "rb"))
            total_size += len(data_chunk[0])

            chunk_size = len(data_chunk[0])
            (
                self.observations[curr_idx : curr_idx + chunk_size],
                self.actions[curr_idx : curr_idx + chunk_size],
                self.rewards[curr_idx : curr_idx + chunk_size],
                self.masks[curr_idx : curr_idx + chunk_size],
                self.dones_float[curr_idx : curr_idx + chunk_size],
                self.next_observations[curr_idx : curr_idx + chunk_size],
            ) = data_chunk

            curr_idx += chunk_size

        if self.capacity != total_size:
            print(
                f"WARNING: buffer capacity does not match size of loaded data! Total size is {total_size}, buffer capacity is {self.capacity}."
            )

        self.insert_index = 0
        self.size = total_size
