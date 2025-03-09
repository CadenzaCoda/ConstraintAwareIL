import copy
from abc import ABC, abstractmethod
from collections import deque, defaultdict
from pathlib import Path
from typing import Callable, List, TypeVar, Type, Dict
import numpy as np
import scipy

# from FADS.density import KDE
# from FADS.sampling import DS
import FADS

from mpclab_common.pytypes import VehicleState
from scipy.spatial import ConvexHull, Delaunay, QhullError
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm

from carla_gym.controllers.barc_lmpc import LMPCWrapper
from utils import pytorch_util as ptu
import os
from loguru import logger


class EfficientReplayBuffer(Dataset, ABC):
    _replay_buffer_name = None

    @property
    def replay_buffer_name(self):
        return self._replay_buffer_name or self.__class__.__name__

    def __init__(self,
                 maxsize=1_000_000, transform=None, random_eviction: bool = True, constants: dict = None,
                 lazy_init=True, name=None):
        """A more memory efficient implementation of the replay buffer with numpy arrays.
        Overwrite _fetch to determine how the dataset interacts with dataloaders.
        """
        super().__init__()
        self.maxsize = maxsize
        self.fields = {}
        self.constants = constants if constants is not None else {}
        self.transform = transform if transform is not None else {}
        self.random_eviction = random_eviction
        self.lazy_init = lazy_init
        self._replay_buffer = name

        self.left = 0
        self.right = 0
        self.size = 0

        self.initialized = False

    def __len__(self):
        return self.size

    def __getitem__(self, index_ext):
        assert index_ext in range(self.size), "Index out of range"
        return self._fetch((index_ext + self.left) % self.maxsize)

    def _fetch(self, index):
        """Note: index is the absolute index in the arrays. """
        data = {}
        for k, v in self.fields.items():
            if k in self.transform:
                # logger.debug(f"Applying transformation for {k}.")
                data[k] = self.transform[k](v[index])
            else:
                data[k] = v[index]
        # data = {k: self.transform[k](v[index]) if k in self.transform else v[index] for k, v in self.fields.items()}
        return {**data, **self.constants}

    def add_frame(self, obs, rews, terminated, truncated, info, **kwargs):
        self.append(batched=False,
                    rewards=rews,
                    **obs, **kwargs)

    def initialize(self, batched: bool, size: int = None, **kwargs):
        if self.initialized:
            return
        if self.lazy_init:
            init_func = lambda shape, dtype: np.empty(shape, dtype)
        else:
            init_func = lambda shape, dtype: np.full(shape, 1, dtype)
        if batched:
            for attr, data in kwargs.items():
                self.fields[attr] = init_func((self.maxsize, *data.shape[1:]), dtype=data.dtype)
                # self.fields[attr] = np.empty((self.maxsize, *data.shape[1:]), dtype=type(data))
        else:
            for attr, data in kwargs.items():
                if np.isscalar(data):
                    # self.fields[attr] = np.empty((self.maxsize,), dtype=type(data))
                    self.fields[attr] = init_func((self.maxsize,), dtype=type(data))
                else:
                    # self.fields[attr] = np.empty((self.maxsize, *data.shape), dtype=data.dtype)
                    self.fields[attr] = init_func((self.maxsize, *data.shape), dtype=data.dtype)
        self.initialized = True

    def clear(self):
        self.left = self.right = self.size = 0

    def preprocess(self):
        pass

    def absorb(self, other: 'EfficientReplayBuffer'):
        if not other.initialized:
            return
        self.append(batched=True,
                    size=other.size,
                    **other.consolidate())
        other.clear()

    def get_traj_len(self, batched, size=None, **kwargs):
        if not batched:
            return 1
        traj_len = size
        for data in kwargs.values():
            if traj_len is None:
                traj_len = data.shape[0]
            else:
                assert traj_len == data.shape[0]
        return traj_len

    def get_idxs_for_new_data(self, traj_len) -> np.ndarray:
        """
        Determine the index for the new data. Evict existing data if the insertion leads to overflowing.

        @param traj_len: Length of the data to be inserted.
        @return: An numpy array indicating where the new data should be inserted.
        """
        index = (np.arange(traj_len) + self.right) % self.maxsize  # Determine the index to put the new data first.
        self.right = (self.right + traj_len) % self.maxsize  # Move the right cursor further right.

        n_overflow = self.size + traj_len - self.maxsize
        if n_overflow > 0:
            # If the insertion causes overflow, the elements on the leftmost side of the replay buffer gets overwritten.
            if self.random_eviction:
                # Randomly select n elements and swap them with the ones at the front, so they are evicted instead.
                self.move_random_n_elements_to_front(n=n_overflow)
            # Move the left cursor to indicate the new starting point of the buffer.
            self.left = (self.left + n_overflow) % self.maxsize
            self.size = self.maxsize  # The replay buffer will surely be full after this.
        else:
            self.size += traj_len
        return index

    def append(self, batched: bool, size=None, **kwargs):
        self.initialize(batched, **kwargs)
        traj_len = self.get_traj_len(batched, **kwargs)
        index = self.get_idxs_for_new_data(traj_len)
        for attr, data in kwargs.items():
            self.fields[attr][index] = data
        return traj_len

    def pop(self, idx=None) -> Dict[str, np.ndarray]:
        """
        idx: the external index. Must be in [0, size).
        Note: This implementation doesn't retain the original order!
        """
        # if idx is None:
        #     return self.popback(1)
        if isinstance(idx, int):
            idx = [idx]
        idx = list(set(idx))
        assert all(0 <= x < self.size for x in idx)
        idx = (np.asarray(idx) + self.left) % self.maxsize
        last_k_idx = (self.right - np.arange(len(idx)) - 1) % self.maxsize
        for attr, data in self.fields.items():
            self.fields[attr][[idx, last_k_idx]] = data[[last_k_idx, idx]]
        self.right = (self.right - len(idx)) % self.maxsize
        self.size -= len(idx)
        return {attr: data[last_k_idx] for attr, data in self.fields.items()}

    def popback(self, n) -> None:
        self.right = (self.right - min(n, len(self))) % self.maxsize
        self.size = max(self.size - n, 0)

    def move_random_n_elements_to_front(self, n: int):
        """
        Move <n_evictions> random entries to the front (for eviction).

        @param n: Number of random elements to move to front.
        @return:
        """
        if n <= 0:
            return
        index = (np.random.randint(0, self.size, (n, )) + self.left) % self.maxsize
        other_index = (np.arange(n) + self.left) % self.maxsize

        for attr in self.fields:
            self.fields[attr][[index, other_index]] = self.fields[attr][[other_index, index]]

    def consolidate(self) -> Dict[str, np.ndarray]:
        if self.left + self.size <= self.maxsize:
            data = {k: v[self.left: self.right] for k, v in self.fields.items()}
        else:
            data = {k: np.concatenate((v[self.left:], v[:self.right]), axis=0) for k, v in self.fields.items()}
        # self.left, self.right = 0, self.size
        # for field in self.fields:
        #     self.fields[field][:self.size] = data[field]
        return data

    def export(self, path=None, name=None):
        path = path or Path(__file__).resolve().parent.parent / 'data'
        path = Path(path)
        os.makedirs(path, exist_ok=True)
        name = f"{self.replay_buffer_name}{f'_{name}' if name else ''}"
        logger.debug("Saving Replay Buffer...")

        np.savez_compressed(path / f"{name}.npz",
                            size=self.size,
                            **self.consolidate())

        logger.debug("Replay Buffer Saved!")

    def load(self, path=None, name=None):
        path = path or Path(__file__).resolve().parent.parent / 'data'
        path = Path(path)
        name = f"{self.replay_buffer_name}{f'_{name}' if name else ''}"
        # path = path or Path(os.path.expanduser('~/Documents/vision_racing_barc/data'))
        # name = name or self.replay_buffer_name
        if not os.path.exists(path / f"{name}.npz"):
            logger.warning("Replay buffer save file not found!")
            return
        logger.debug("Loading replay buffer...")

        data = np.load(path / f"{name}.npz")
        self.size = self.left = self.right = 0
        # self.size = self.right = data['size']
        # self.left = 0

        self.initialize(batched=True, **data)
        self.append(batched=True, **data)

    def dataloader(self, batch_size: int = 64, shuffle: bool = True, num_workers: int = 0, manifest=None) -> DataLoader:
        if manifest is None:
            raise ValueError("Must specify fields to fetch.")

        def collate_fn(batch):
            """
            Custom collate function to extract only the requested features and labels.
            """
            ret = []
            for blocks in manifest:
                ret.append([ptu.from_numpy(np.stack([item[field] for item in batch], axis=0)) for field in blocks])
            return ret

        return DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          drop_last=True,
                          collate_fn=collate_fn)

    def sample_latest_data(self, traj_len) -> Dict[str, np.ndarray]:
        if traj_len > self.size:
            raise IndexError(f'Current replay buffer size is {self.size}, but {traj_len} examples are requested.')
        idx = (self.right - np.arange(traj_len, 0, -1)) % self.maxsize
        return {k: v[idx] for k, v in self.fields.items()}

    def sample_random_data(self, traj_len) -> Dict[str, np.ndarray]:
        if traj_len > self.size:
            raise IndexError(f'Current replay buffer size is {self.size}, but {traj_len} examples are requested.')
        idx = (np.random.randint(0, self.size, traj_len) + self.left) % self.maxsize
        return {k: v[idx] for k, v in self.fields.items()}

    def is_in_knn_convex_hull(self, query, fields, k: int, threshold=0.5):
        """
        The query must be batched, and have the same dimension as the corresponding field.
        """
        nbr = NearestNeighbors(n_neighbors=k, algorithm='auto')
        # Prepare the data
        if self.left + self.size == self.right:
            data = np.concatenate([self.fields[field][self.left:self.right] for field in fields], axis=-1)
        else:
            data = np.concatenate(
                [np.concatenate((self.fields[field][self.left:], self.fields[field][:self.right]), axis=0) for field in
                 fields], axis=-1)
        # Efficiently subsample data using random choice without full permutation
        n_samples = min(len(data), 32768)
        if n_samples < len(data):
            # indices = np.random.choice(len(data), size=n_samples, replace=False)
            fads = FADS.FADS(data)
            indices = fads.DS(n_samples)
            # data = data[indices]
        else:
            data = data.copy()  # Ensure original data isn't modified

        # Estimate density (kernel density estimation)

        # probabilities = 1 / density_estimator.scores  # Inverse density weights

        # Select 1,000 uniform subsamples
        # ds_subsample = DS(data, n=8192, probabilities=probabilities).select()

        nbr.fit(data)
        # Correctly retrieve up to k neighbors within threshold for each query
        dists, indices = nbr.radius_neighbors(query, radius=threshold, return_distance=True, sort_results=True)
        # Truncate each query's neighbors to k closest
        # indices = [idx[:k] for idx in indices]
        ret = []
        for q, idx in tqdm(zip(query, indices), total=len(indices), desc='Self-labeling'):
            if len(idx) < q.shape[0] + 1:
                ret.append(False)
                continue
            k_nearest_points = data[idx]
            try:
                hull = ConvexHull(k_nearest_points, qhull_options='QJ Pp')
            except QhullError:
                ret.append(False)
                continue
            if hull.vertices.shape[0] < 8:
                ret.append(False)
                continue
            # Check if q is inside the convex hull using equations
            # Compute dot product of q with each equation's normal and add offset
            vals = np.dot(hull.equations[:, :-1], q) + hull.equations[:, -1]
            # Allow a small tolerance for numerical precision
            is_inside = np.all(vals <= 1e-8)
            ret.append(is_inside)
        return np.asarray(ret)

    # def __is_in_knn_convex_hull(self, query, fields, k: int, threshold=1.):
    #     """
    #     The query must be batched, and have the same dimension as the corresponding field.
    #     """
    #     nbr = NearestNeighbors(n_neighbors=k, algorithm='auto')
    #     if self.left + self.size == self.right:
    #         data = np.concatenate([self.fields[field][self.left:self.right] for field in fields], axis=-1)
    #     else:
    #         data = np.concatenate(
    #             [np.concatenate((self.fields[field][self.left:], self.fields[field][:self.right]), axis=0) for field in
    #              fields], axis=-1)
    #     data = np.random.permutation(data)[:1024]  # Limit the size for comparison. Use randomization to keep the distribution approximately the same.
    #     nbr.fit(data)
    #     # dists, indices = nbr.kneighbors(query)  # dists, indices: (Q, k)
    #     _, indices = nbr.radius_neighbors(query, radius=threshold, return_distance=True, sort_results=True)[:k]
    #     # indices = indices[:k]  # Limiting the size of the neighbors.
    #     ret = []
    #     for q, idx in zip(query, indices):
    #         # k_nearest_points = data[idx][dist <= threshold]
    #         k_nearest_points = data[idx]
    #         if k_nearest_points.shape[0] < q.shape[0] + 1:
    #             ret.append(False)
    #             continue
    #         hull = ConvexHull(k_nearest_points, qhull_options='QJ Pp')
    #         if hull.vertices.shape[0] < 8:
    #             ret.append(False)
    #             continue
    #         delaunay = Delaunay(k_nearest_points[hull.vertices], qhull_options='QJ Pp')
    #         ret.append(delaunay.find_simplex(q) >= 0)
    #     return np.asarray(ret)


class EfficientReplayBufferPN(EfficientReplayBuffer):
    def __init__(self, maxsize: int = 1_000_000, transform=None, random_eviction: bool = True,
                 lazy_init=True):
        self.D_pos = EfficientReplayBuffer(maxsize=maxsize, transform=transform, random_eviction=random_eviction,
                                           constants={'safe': np.array([1.], dtype=np.float32)},
                                           lazy_init=lazy_init, name='D_pos')
        self.D_neg = EfficientReplayBuffer(maxsize=maxsize, transform=transform, random_eviction=random_eviction,
                                           constants={'safe': np.array([0.1], dtype=np.float32)},
                                           lazy_init=lazy_init, name='D_neg')
        self.buffer = EfficientReplayBuffer(maxsize=2048, random_eviction=False, lazy_init=lazy_init)
        # self.buffer2 = EfficientReplayBuffer(maxsize=16384, random_eviction=False)

    def add_frame(self, obs, rews, terminated, truncated, info, **kwargs):
        if terminated:
            self.D_pos.absorb(self.buffer)
            # self.buffer2.absorb(self.buffer)
        elif truncated:
            self.D_neg.absorb(self.buffer)
            # self.D_neg.absorb(self.buffer2)
        self.buffer.append(batched=False,
                           rewards=rews,
                           **obs, **kwargs)

    def clear_buffer(self):
        self.buffer.clear()
        # self.buffer2.clear()

    def __len__(self):
        return len(self.D_pos) + len(self.D_neg)

    def __getitem__(self, idx):
        if idx < len(self.D_pos):
            return self.D_pos[idx]
        return self.D_neg[idx - len(self.D_pos)]

    def popback(self, n) -> None:
        return self.buffer.popback(n)

    def pop(self, idx=None) -> None:
        raise NotImplementedError

    def export(self, path=None, name=None):
        self.D_pos.export(path=path, name=f"{name}_pos")
        self.D_neg.export(path=path, name=f"{name}_neg")

    def load(self, path=None, name=None):
        self.D_pos.load(path=path, name=f"{name}_pos")
        self.D_neg.load(path=path, name=f"{name}_neg")

    def preprocess(self, rho=1.):
        self.buffer.clear()
        if not self.D_neg.initialized or len(self.D_neg) < 16:
            return
        mask = self.D_pos.is_in_knn_convex_hull(self.D_neg.consolidate()['state'], ['state'], k=100, threshold=rho)
        projected_safe = self.D_neg.pop(np.where(mask)[0])
        # self.D_pos.append(batched=True, size=np.sum(mask), **projected_safe))
        logger.debug(f"Ditched {np.sum(mask)} examples from D_neg. {len(self.D_neg)} left uncertain. {len(self.D_pos)} known safe.")
        return projected_safe


class EfficientReplayBufferPN_nopreprocess(EfficientReplayBufferPN):
    def preprocess(self):
        return


class LMPCPredictor(LMPCWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lmpc_params = dict(
            N=15,
            n_ss_pts=96,
            n_ss_its=8,
        )


class EfficientReplayBufferPN_LMPC(EfficientReplayBuffer):
    def __init__(self, maxsize: int = 1_000_000, random_eviction: bool = True, expert=None):
        self.D_pos = EfficientReplayBuffer(maxsize=maxsize, random_eviction=random_eviction,
                                           constants={'safe': np.array([0.9], dtype=np.float32)})
        self.D_neg = EfficientReplayBuffer(maxsize=maxsize, random_eviction=random_eviction,
                                           constants={'safe': np.array([0.1], dtype=np.float32)})
        self.buffer = EfficientReplayBuffer(maxsize=16384, random_eviction=random_eviction)
        self.expert = expert

        # self.lap_data = []
        # self.last_lap_data = None

    # def _process_lmpc_data(self, lap_no):
    #     # try:
    #     # q_data, u_data, lap_c2g = self.lmpc._process_lap_data(self.lap_data)
    #     q_data, u_data = zip(*self.lap_data)
    #     q_data = np.asarray(q_data)
    #     u_data = np.asarray(u_data)
    #     lap_c2g = np.zeros((q_data.shape[0],))
    #     # except IndexError as e:
    #     #     return
    #     self.lmpc.lmpc_controller.add_iter_data(q_data, u_data)
    #     self.lmpc.lmpc_controller.add_safe_set_data(q_data, u_data, lap_c2g)

        # if self.last_lap_data is not None:
            # If this is not the first lap, consider the current lap as the extension of the previous lap
            # and append to the safe set as additional data.
            # last_lap_end = self.last_lap_data[-1].t
            # for ld in copy.deepcopy(self.lap_data):
                # ld.p.s += self.lmpc.track_obj.track_length
                # self.last_lap_data.append(ld)
            # q_data, u_data, lap_c2g = self.lmpc._process_lap_data(self.last_lap_data, lap_end=last_lap_end)
            # self.lmpc.lmpc_controller.add_safe_set_data(q_data, u_data, lap_c2g, iter_idx=lap_no - 1)
        # self.last_lap_data = copy.deepcopy(self.lap_data)
        # self.lap_data = []

    def add_frame(self, obs, rews, terminated, truncated, info, **kwargs):
        if terminated:
            self.D_pos.absorb(self.buffer)
            # self._process_lmpc_data(len(self.lap_data))
        elif truncated:
            self.D_neg.absorb(self.buffer)
        self.buffer.append(batched=False,
                           rewards=rews,
                           **obs, **kwargs)
        # self.lap_data.append((obs['state'], kwargs['closed_loop_action']))

    def clear_buffer(self):
        self.buffer.clear()
        # self.lmpc.clear_lap_data()
        # self.lap_data = []
        # self.last_lap_data = None

    def __len__(self):
        return len(self.D_pos) + len(self.D_neg)

    def __getitem__(self, idx):
        if idx < len(self.D_pos):
            return self.D_pos[idx]
        return self.D_neg[idx - len(self.D_pos)]

    def popback(self, n) -> None:
        return self.buffer.popback(n)

    def pop(self, idx=None) -> None:
        raise NotImplementedError

    def export(self, path=None, name=None):
        self.D_pos.export(path=path, name=f"{name}_pos")
        self.D_neg.export(path=path, name=f"{name}_neg")

    def load(self, path=None, name=None):
        self.D_pos.load(path=path, name=f"{name}_pos")
        self.D_neg.load(path=path, name=f"{name}_neg")

    def preprocess(self):
        self.clear_buffer()
        # self.buffer.clear()
        if not self.D_neg.initialized:
            return
        neg_data = self.D_neg.consolidate()['state']
        mask = []
        status = defaultdict(lambda: 0)
        for q in tqdm(neg_data, desc="Self-labeling"):
            _state = VehicleState()
            _state.v.v_long, _state.v.v_tran, _state.w.w_psi, _state.x.x, _state.x.y, _state.e.psi = q
            self.expert.reset(options={'vehicle_state': _state})
            info = self.expert.step(_state)
            x_tran_pred = self.expert.controller.q_pred[:, 5]
            # e_psi_pred = self.lmpc.lmpc_controller.q_pred[:, 3]
            # status[info['status']] += 1
            mask.append((np.abs(x_tran_pred) <= self.expert.track_obj.half_width).all())
            # mask.append(info['success'])
        # mask = self.D_pos.is_in_knn_convex_hull(self.D_neg.consolidate()['state'], ['state'], k=16)
        projected_safe = self.D_neg.pop(np.where(mask)[0])
        # self.D_pos.append(batched=True, size=np.sum(mask), **projected_safe)
        logger.debug(f"Status in self-relabeling: {dict(status)}")
        logger.debug(f"Ditched {np.sum(mask)} examples from D_neg. {len(self.D_neg)} left uncertain. {len(self.D_pos)} known safe.")


class EfficientReplayBufferSA(EfficientReplayBuffer):
    def dataloader(self, batch_size: int = 64, shuffle: bool = True, num_workers: int = 0, manifest=None) -> DataLoader:
        return super().dataloader(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                  manifest=[['states'], ['actions']])


class EfficientReplayBufferSCA(EfficientReplayBuffer):
    def dataloader(self, batch_size: int = 64, shuffle: bool = True, num_workers: int = 0, manifest=None) -> DataLoader:
        return super().dataloader(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                  manifest=[['states', 'conds'], ['actions']])

# class EfficientReplayBufferSA(EfficientReplayBuffer):
#     def _fetch(self, index):
#         return self.fields['states'][index], self.fields['actions'][index]
#
#     def dataloader(self, batch_size: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
#         def collate_fn(data):
#             states, actions = zip(*data)
#             states = ptu.from_numpy(np.asarray(states)).float()
#             actions = ptu.from_numpy(np.asarray(actions)).float()
#             return states, actions
#
#         return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers)
#
#
# class EfficientReplayBufferSCA(EfficientReplayBuffer):
#     def add_frame(self, ob, ac, rew=None, done=None, info=None):
#         state = ob['state']
#         cond = np.array([info['avg_lap_speed']])
#         self.append(batched=False,
#                     states=state,
#                     actions=ac,
#                     conds=cond,
#                     rews=rew,
#                     dones=done,)
#
#     def _fetch(self, index):
#         return self.fields['states'][index], self.fields['conds'][index], self.fields['actions'][index]
#
#     def dataloader(self, batch_size: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
#         def collate_fn(data):
#             states, conds, actions = zip(*data)
#             states = ptu.from_numpy(np.asarray(states)).float()
#             conds = ptu.from_numpy(np.asarray(conds)).float()
#             actions = ptu.from_numpy(np.asarray(actions)).float()
#             return states, conds, actions
#
#         return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers)


# Create a new Dataset class:
# All information in batch-first numpy arrays from the same trajectory as elements in deques.
# Data augmentations that affect labels (e.g. horizontal flipping) need to be done for the entire trajectory.
# The __len__ method returns number of trajectories in the dataset.
# Memory management is achieved by keeping track of number of examples in the dataset.
#
