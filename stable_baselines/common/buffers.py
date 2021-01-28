import random
from typing import Optional, List, Union

import numpy as np

from stable_baselines.common.segment_tree import SumSegmentTree, MinSegmentTree
from stable_baselines.common.vec_env import VecNormalize


class ReplayBuffer(object):
    __name__ = "ReplayBuffer"
    def __init__(self, size: int, extra_data_names=()):
        """
        Implements a ring buffer (FIFO).

        :param size: (int)  Max number of transitions to store in the buffer. When the buffer overflows the old
            memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0
        self._extra_data_names = sorted(extra_data_names)

    def __len__(self) -> int:
        return len(self._storage)

    @property
    def storage(self):
        """[(Union[np.ndarray, int], Union[np.ndarray, int], float, Union[np.ndarray, int], bool)]: content of the replay buffer"""
        return self._storage

    @property
    def buffer_size(self) -> int:
        """float: Max capacity of the buffer"""
        return self._maxsize

    def can_sample(self, n_samples: int) -> bool:
        """
        Check if n_samples samples can be sampled
        from the buffer.

        :param n_samples: (int)
        :return: (bool)
        """
        return len(self) >= n_samples

    def is_full(self) -> int:
        """
        Check whether the replay buffer is full or not.

        :return: (bool)
        """
        return len(self) == self.buffer_size

    def add(self, obs_t, action, reward, obs_tp1, done, *extra_data, **extra_data_kwargs):
        """
        add a new transition to the buffer

        :param obs_t: (Union[np.ndarray, int]) the last observation
        :param action: (Union[np.ndarray, int]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Union[np.ndarray, int]) the current observation
        :param done: (bool) is the episode done
        """
        data = (obs_t, action, reward, obs_tp1, done, *extra_data, *[extra_data_kwargs[k] for k in sorted(extra_data_kwargs)])

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def extend(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new batch of transitions to the buffer

        :param obs_t: (Union[Tuple[Union[np.ndarray, int]], np.ndarray]) the last batch of observations
        :param action: (Union[Tuple[Union[np.ndarray, int]]], np.ndarray]) the batch of actions
        :param reward: (Union[Tuple[float], np.ndarray]) the batch of the rewards of the transition
        :param obs_tp1: (Union[Tuple[Union[np.ndarray, int]], np.ndarray]) the current batch of observations
        :param done: (Union[Tuple[bool], np.ndarray]) terminal status of the batch

        Note: uses the same names as .add to keep compatibility with named argument passing
                but expects iterables and arrays with more than 1 dimensions
        """
        for data in zip(obs_t, action, reward, obs_tp1, done):
            if self._next_idx >= len(self._storage):
                self._storage.append(data)
            else:
                self._storage[self._next_idx] = data
            self._next_idx = (self._next_idx + 1) % self._maxsize

    @staticmethod
    def _normalize_obs(obs: np.ndarray,
                       env: Optional[VecNormalize] = None) -> np.ndarray:
        """
        Helper for normalizing the observation.
        """
        if env is not None:
            return env.normalize_obs(obs)
        return obs

    @staticmethod
    def _normalize_reward(reward: np.ndarray,
                          env: Optional[VecNormalize] = None) -> np.ndarray:
        """
        Helper for normalizing the reward.
        """
        if env is not None:
            return env.normalize_reward(reward)
        return reward

    def _encode_sample(self, idxes: Union[List[int], np.ndarray], env: Optional[VecNormalize] = None):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        extra_data = {name: [] for name in self._extra_data_names}
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, *extra_timestep_data = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)

            for data_i, extra_data_name in enumerate(self._extra_data_names):
                data = extra_timestep_data[data_i]
                if np.ndim(data) == 0:
                    extra_data[extra_data_name].append(data)
                else:
                    extra_data[extra_data_name].append(np.array(data, copy=False))

        extra_data = {k: np.array(v) for k, v in extra_data.items()}
        if "bootstrap" in self._extra_data_names:
            dones = 1 - extra_data.pop("bootstrap").astype(np.float32)

        return self._normalize_obs(np.array(obses_t), env), np.array(actions), \
               self._normalize_reward(np.array(rewards), env), self._normalize_obs(np.array(obses_tp1), env), \
                np.array(dones), extra_data

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None, **_kwargs):
        """
        Sample a batch of experiences.

        :param batch_size: (int) How many transitions to sample.
        :param env: (Optional[VecNormalize]) associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes, env=env)


# TODO: scan/"burn in"
class RecurrentReplayBuffer(ReplayBuffer):
    __name__ = "RecurrentReplayBuffer"

    def __init__(self, size, sequence_length=1, scan_length=0, extra_data_names=(), rnn_inputs=(), her_k=4):
        super().__init__(size)
        self._sample_cycle = 0
        self.her_k = her_k
        self._extra_data_names = sorted(extra_data_names)
        self._data_name_to_idx = {"obs": 0, "action": 1, "reward": 2, "obs_tp1": 3, "done": 4,
                                  **{name: 5 + i for i, name in enumerate(self._extra_data_names)}}
        self._current_episode_data = []
        self.sequence_length = sequence_length
        assert self.sequence_length >= 1
        self.scan_length = scan_length
        self._rnn_inputs = rnn_inputs
        assert self.scan_length == 0 or len(self._rnn_inputs) > 0
        self._is_full = False

    def add(self, obs_t, action, reward, obs_tp1, done, *extra_data, **extra_data_kwargs):
        if self.her_k > 0:
            obs_t = [obs_t]
            obs_tp1 = [obs_tp1]
            reward = [reward]
        data = [obs_t, action, reward, obs_tp1, done, *extra_data, *[extra_data_kwargs[k] for k in sorted(extra_data_kwargs)]]  # Data needs to be mutable
        self._current_episode_data.append(data)
        self._sample_cycle += 1
        if done:
            self.store_episode()

    def store_episode(self):
        if len(self._current_episode_data) >= self.sequence_length + self.scan_length:
            if self._sample_cycle >= self.buffer_size:
                self._next_idx = 0
                self._sample_cycle = 0
                self._is_full = True
            if not self._is_full:
                self._storage.append(self._current_episode_data)
            else:
                try:
                    self._storage[self._next_idx] = self._current_episode_data 
                except IndexError:
                    self._storage.append(self._current_episode_data)
                self._next_idx += 1
        else:
            if self.her_k > 0:
                self._sample_cycle -= sum([len(t[0]) for t in self._current_episode_data])
            else:
                self._sample_cycle -= len(self._current_episode_data)
        self._current_episode_data = []

    def add_her(self, obs, obs_tp1, reward, timestep, ep_index=None):
        assert self.her_k > 0
        if ep_index is not None:
            episode_data = self._storage[ep_index]
        else:
            episode_data = self._current_episode_data
        episode_data[timestep][0].append(obs)
        episode_data[timestep][2].append(reward)
        episode_data[timestep][3].append(obs_tp1)
        self._sample_cycle += 1

    def sample(self, batch_size, sequence_length=None, **_kwargs):
        if sequence_length is None:
            sequence_length = self.sequence_length
        assert batch_size % sequence_length == 0

        ep_idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size // sequence_length)]
        ep_ts = [random.randint(self.scan_length, len(self._storage[ep_i]) - 1 - (sequence_length - 1)) for ep_i in
                 ep_idxes]
        extra_data = {name: [] for name in self._extra_data_names}
        extra_data.update({"scan_{}".format(name): [] for name in self._rnn_inputs})

        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i, ep_i in enumerate(ep_idxes):
            ep_data = self.storage[ep_i]
            ep_t = ep_ts[i]
            if self.her_k > 0:
                her_idx = random.randint(0, self.her_k + 2)
            for scan_t in range(ep_t - self.scan_length, ep_t):
                for scan_data_name in self._rnn_inputs:
                    data = ep_data[scan_t][self._data_name_to_idx[scan_data_name]]
                    if self.her_k > 0 and self._data_name_to_idx[scan_data_name] in [self._data_name_to_idx[n] for n in
                                                                                     ["obs", "reward", "obs_tp1"]]:
                        data = data[0]
                    extra_data["scan_{}".format(scan_data_name)].append(data)
            for seq_i in range(sequence_length):
                obs_t, action, reward, obs_tp1, done, *extra_timestep_data = ep_data[ep_t + seq_i]
                if self.her_k > 0:
                    try:  # TODO: fix indexing with last timestep data not having her data
                        obs_t, obs_tp1, reward = obs_t[her_idx], obs_tp1[her_idx], reward[her_idx]
                    except IndexError:
                        obs_t, obs_tp1, reward = obs_t[0], obs_tp1[0], reward[0]
                obses_t.append(np.array(obs_t, copy=False))
                actions.append(np.array(action, copy=False))
                rewards.append(reward)
                obses_tp1.append(np.array(obs_tp1, copy=False))
                dones.append(done)
                for data_i, extra_data_name in enumerate(self._extra_data_names):
                    if "state" in extra_data_name:
                        if seq_i > 0:  # For RNN states only get the first state in the sequence (or in the scan)
                            continue
                        data = ep_data[ep_t + seq_i - self.scan_length][self._data_name_to_idx[extra_data_name]]
                    else:
                        data = extra_timestep_data[data_i]
                    if np.ndim(data) == 0:
                        extra_data[extra_data_name].append(data)
                    else:
                        extra_data[extra_data_name].append(np.array(data, copy=False))

        extra_data = {k: np.array(v) for k, v in extra_data.items()}
        extra_data["state_idxs"] = list(zip(ep_idxes, [t + sequence_length for t in ep_ts]))
        if self.scan_length > 0:
            extra_data["state_idxs_scan"] = list(zip(ep_idxes, ep_ts))

        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones), extra_data

    def update_state(self, idxs, data):
        for i, (ep_idx, t) in enumerate(idxs):
            try:
                for state_name, state_val in data.items():
                    self.storage[ep_idx][t][self._data_name_to_idx[state_name]] = state_val[i, :]
            except IndexError:  # Hidden state computed for last sample in episode, doesnt belong to any sample
                pass

    def __len__(self):  # TODO: consider if this is important enough to do right
        return max(self._sample_cycle - ((len(self._current_episode_data) - 1) * (1 + self.her_k) + 1), 0) \
            if not self._is_full else self.buffer_size

    def is_full(self):
        return self._is_full


# TODO: maybe add support for episode constant data
class EpisodicRecurrentReplayBuffer(ReplayBuffer):
    __name__ = "EpisodicRecurrentReplayBuffer"

    def __init__(self, size, episode_length, sequence_length=10, extra_data_names=()):
        super().__init__(size // episode_length)
        self._current_episode_data = []
        # self._episode_data = []  # Data which is constant within episode
        self._extra_data_names = sorted(extra_data_names)
        self._data_name_to_idx = {"obs": 0, "action": 1, "reward": 2, "obs_tp1": 3, "done": 4}
        self._data_name_to_idx.update({name: i + 5 for i, name in enumerate(self._extra_data_names)})
        self._sequence_length = sequence_length  # TODO: add scan length and assert is multiple of sample_consecutive_max

    def add(self, obs_t, action, reward, obs_tp1, done, *extra_data):
        self._current_episode_data.append(
            [obs_t, action, reward, obs_tp1, done, *extra_data])  # List to support updating states etc.

        if done:
            self.store_episode()

    def store_episode(self):
        if len(self._current_episode_data) == 0:
            return

        if self._next_idx >= len(self._storage):
            self._storage.append(self._current_episode_data)
        else:
            self._storage[self._next_idx] = self._current_episode_data
        self._next_idx = (self._next_idx + 1) % self._maxsize
        self._current_episode_data = []

    def sample(self, batch_size, sequence_length=None):
        if sequence_length is None:
            sequence_length = self._sequence_length
        samples_left = batch_size
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        extra_data = [[] for i in range(len(self._extra_data_names))]
        state_idxs = []
        while samples_left > 0:
            ep_idx = np.random.randint(0, len(self._storage) - 1)
            ep_data = self._storage[ep_idx]
            ep_start_idx = np.random.randint(0, max(len(ep_data) - sequence_length, 1))
            ep_data = ep_data[ep_start_idx:ep_start_idx + sequence_length + 1]
            state_idxs.append((ep_idx, ep_start_idx + sequence_length))
            if len(ep_data) > samples_left:
                ep_data = ep_data[:samples_left]

            for j, timestep in enumerate(ep_data):
                obs_t, action, reward, obs_tp1, done, *extra_timestep_data = timestep
                obses_t.append(np.array(obs_t, copy=False))
                actions.append(np.array(action, copy=False))
                rewards.append(reward)
                obses_tp1.append(np.array(obs_tp1, copy=False))
                dones.append(done)
                for data_i, data in enumerate(extra_timestep_data):
                    if np.ndim(data) == 0:
                        extra_data[data_i].append(data)
                    else:
                        extra_data[data_i].append(np.array(data, copy=False))

            samples_left -= len(ep_data)

            assert samples_left >= 0

        extra_data_dict = {name: np.array(extra_data[i]) for i, name in enumerate(self._extra_data_names)}
        extra_data_dict["reset"] = np.zeros(shape=(batch_size,))  # np.array(resets)
        extra_data_dict["state"] = extra_data_dict["state"][::sequence_length]
        extra_data_dict["state_idxs"] = state_idxs

        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(
            dones), extra_data_dict

    def update_state(self, idxs, data):
        for i, (ep_idx, t) in enumerate(idxs):
            if isinstance(data, list):
                self.storage[ep_idx][t][self._data_name_to_idx["pi_state"]] = data[0][i, :]
                self.storage[ep_idx][t][self._data_name_to_idx["qf1_state"]] = data[1][i, :]
                self.storage[ep_idx][t][self._data_name_to_idx["qf2_state"]] = data[2][i, :]
            else:
                self.storage[ep_idx][t][self._data_name_to_idx["state"]] = data[i, :]

    def __len__(self):
        if len(self.storage) > 1:
            return sum([len(episode) for episode in self._storage])
        else:
            return 0


class DRRecurrentReplayBuffer(ReplayBuffer):
    __name__ = "DRRecurrentReplayBuffer"

    def __init__(self, size, episode_max_len, scan_length, her_k=4):
        self.her_k = her_k
        super().__init__(size)
        self._scan_length = scan_length
        self._maxsize = self._maxsize // episode_max_len
        self._episode_my = []

    def add(self, obs_t, action, reward, obs_tp1, done, goal, my=None):
        assert not (done and my is None)

        if self.her_k > 0:
            goal = [goal]
            reward = [reward]
        data = (obs_t, action, reward, obs_tp1, done, goal)
        self._current_episode_data.append(data)
        if done:
            if self._next_idx >= len(self._storage):
                self._storage.append(self._current_episode_data)
                self._episode_my.append(my)
            else:
                self._storage[self._next_idx] = self._current_episode_data
                self._episode_my[self._next_idx] = my
            self._next_idx = (self._next_idx + 1) % self._maxsize
            self._current_episode_data = []

    def add_her(self, goal, reward, timestep, ep_index=None):
        assert self.her_k > 0
        if ep_index is not None:
            episode_data = self._storage[ep_index]
        else:
            episode_data = self._current_episode_data
        episode_data[timestep][5].append(goal)
        episode_data[timestep][2].append(reward)

    def sample(self, batch_size, **_kwargs):
        if self.her_k > 0:
            num_episodes = len(self._storage)
            ep_idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
            ep_ts = [
                random.randint(self._scan_length * (1 + self.her_k), (len(self._storage[ep_i]) - 1) * (1 + self.her_k))
                for ep_i in ep_idxes]  # - self._optim_length)
            return self._encode_sample(ep_idxes, ep_ts)
        else:
            return super(DRRecurrentReplayBuffer, self).sample(batch_size)

    def _encode_sample(self, ep_idxes, ep_ts):
        obses_t, actions, rewards, obses_tp1, dones, goals, mys, hists_o, hists_a = [], [], [], [], [], [], [], [], []
        for i, ep_i in enumerate(ep_idxes):
            if self.her_k > 0:
                ep_t = int(ep_ts[i] / (self.her_k + 1))
            else:
                ep_t = ep_ts[i]
            ep_data = self._storage[ep_i]
            obs_t, action, reward, obs_tp1, done, goal = ep_data[ep_t]
            if self.her_k > 0:
                goal = goal[ep_ts[i] - ep_t * (self.her_k + 1)]
                reward = reward[ep_ts[i] - ep_t * (self.her_k + 1)]
            if self._scan_length > 0:
                ep_scan_start = ep_t - self._scan_length if ep_t - self._scan_length >= 0 else 0
                hist_o, hist_a = [], []
                for hist_i in range(ep_scan_start, ep_t):
                    hist_o.append(np.array(ep_data[hist_i][0]))
                    if hist_i > 0:
                        hist_a.append(np.array(ep_data[hist_i - 1][1]))
                    else:
                        hist_a.append(np.zeros(shape=(len(ep_data[0][1]),)))
                hist_o.append(np.array(obs_t))
                hist_a.append(np.array(ep_data[ep_t - 1][1]))
            else:
                hist_o = [obs_t]
                hist_a = [ep_data[ep_t - 1][1]]
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            hists_o.extend(hist_o)
            hists_a.extend(hist_a)
            goals.append(np.array(goal, copy=False))
            mys.append(np.array(self._episode_my[ep_i], copy=False))
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones), {
            "goal": np.array(goals), "obs_rnn": np.array(hists_o), "action_prev": np.array(hists_a),
            "my": np.array(mys)}


class PrioritizedReplayBuffer(ReplayBuffer):
    __name__ = "PrioritizedReplayBuffer"

    def __init__(self, size, alpha):
        """
        Create Prioritized Replay buffer.

        See Also ReplayBuffer.__init__

        :param size: (int) Max number of transitions to store in the buffer. When the buffer overflows the old memories
            are dropped.
        :param alpha: (float) how much prioritization is used (0 - no prioritization, 1 - full prioritization)
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new transition to the buffer

        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        """
        idx = self._next_idx
        super().add(obs_t, action, reward, obs_tp1, done)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def extend(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new batch of transitions to the buffer

        :param obs_t: (Union[Tuple[Union[np.ndarray, int]], np.ndarray]) the last batch of observations
        :param action: (Union[Tuple[Union[np.ndarray, int]]], np.ndarray]) the batch of actions
        :param reward: (Union[Tuple[float], np.ndarray]) the batch of the rewards of the transition
        :param obs_tp1: (Union[Tuple[Union[np.ndarray, int]], np.ndarray]) the current batch of observations
        :param done: (Union[Tuple[bool], np.ndarray]) terminal status of the batch

        Note: uses the same names as .add to keep compatibility with named argument passing
            but expects iterables and arrays with more than 1 dimensions
        """
        idx = self._next_idx
        super().extend(obs_t, action, reward, obs_tp1, done)
        while idx != self._next_idx:
            self._it_sum[idx] = self._max_priority ** self._alpha
            self._it_min[idx] = self._max_priority ** self._alpha
            idx = (idx + 1) % self._maxsize

    def _sample_proportional(self, batch_size):
        mass = []
        total = self._it_sum.sum(0, len(self._storage) - 1)
        # TODO(szymon): should we ensure no repeats?
        mass = np.random.random(size=batch_size) * total
        idx = self._it_sum.find_prefixsum_idx(mass)
        return idx

    def sample(self, batch_size: int, beta: float = 0, env: Optional[VecNormalize] = None):
        """
        Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.

        :param batch_size: (int) How many transitions to sample.
        :param beta: (float) To what degree to use importance weights (0 - no corrections, 1 - full correction)
        :param env: (Optional[VecNormalize]) associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
            - weights: (numpy float) Array of shape (batch_size,) and dtype np.float32 denoting importance weight of
                each sampled transition
            - idxes: (numpy int) Array of shape (batch_size,) and dtype np.int32 idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)
        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)
        p_sample = self._it_sum[idxes] / self._it_sum.sum()
        weights = (p_sample * len(self._storage)) ** (-beta) / max_weight
        encoded_sample = self._encode_sample(idxes, env=env)
        return tuple(list(encoded_sample) + {"is_weights": weights, "idxs": idxes})

    def update_priorities(self, idxes, priorities):
        """
        Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        :param idxes: ([int]) List of idxes of sampled transitions
        :param priorities: ([float]) List of updated priorities corresponding to transitions at the sampled idxes
            denoted by variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        assert np.min(priorities) > 0
        assert np.min(idxes) >= 0
        assert np.max(idxes) < len(self.storage)
        self._it_sum[idxes] = priorities ** self._alpha
        self._it_min[idxes] = priorities ** self._alpha
        self._max_priority = max(self._max_priority, np.max(priorities))


class DiscrepancyReplayBuffer(ReplayBuffer):
    def __init__(self, size, scorer):
        """
        Create Prioritized Replay buffer.

        See Also ReplayBuffer.__init__

        :param size: (int) Max number of transitions to store in the buffer. When the buffer overflows the old memories
            are dropped.
        :param alpha: (float) how much prioritization is used (0 - no prioritization, 1 - full prioritization)
        """
        super(DiscrepancyReplayBuffer, self).__init__(size)
        self.scores = []
        self.scorer = scorer
        self.min_score = None
        self.max_score = None

    def add(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new transition to the buffer

        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        """
        idx = self._next_idx

        score = self.scorer(np.expand_dims(obs_tp1, axis=0))[0][0]
        if self.min_score is None or score < self.min_score:
            self.min_score = score
        if self.max_score is None or score > self.max_score:
            self.max_score = score
        if self._next_idx >= len(self._storage):
            self.scores.append(score)
        else:
            self.scores[idx] = score

        super().add(obs_t, action, reward, obs_tp1, done)
        self.storage[-1] += (self.scorer(np.expand_dims(obs_tp1, axis=0)),)

    def sample(self, batch_size, **_kwargs):
        """
        Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.

        :param batch_size: (int) How many transitions to sample.
        :param beta: (float) To what degree to use importance weights (0 - no corrections, 1 - full correction)
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
            - weights: (numpy float) Array of shape (batch_size,) and dtype np.float32 denoting importance weight of
                each sampled transition
            - idxes: (numpy int) Array of shape (batch_size,) and dtype np.int32 idexes in buffer of sampled experiences
        """
        if not self.can_sample(batch_size):
            return self._encode_sample(list(range(len(self))))

        scores = self._scale_scores(np.array(self.scores))
        idxs = np.random.choice(np.arange(len(scores)), size=(batch_size,), p=scores / np.sum(scores), replace=False)

        return self._encode_sample(idxs)

    def update_priorities(self):
        """
        Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        :param idxes: ([int]) List of idxes of sampled transitions
        :param priorities: ([float]) List of updated priorities corresponding to transitions at the sampled idxes
            denoted by variable `idxes`.
        """
        scores = self.scorer([transition[0] for transition in self.storage])[:, 0]
        for i, transition in enumerate(self.storage):
            transition[-1] = scores[i]

    def _scale_scores(self, vals):
        return (vals - self.min_score) / (self.max_score - self.min_score) * (1 - 0.1) + 0.1


class StableReplayBuffer(ReplayBuffer):
    __name__ = "StableReplayBuffer"

    def __init__(self, size):
        """
        Create Prioritized Replay buffer.

        See Also ReplayBuffer.__init__

        :param size: (int) Max number of transitions to store in the buffer. When the buffer overflows the old memories
            are dropped.
        :param alpha: (float) how much prioritization is used (0 - no prioritization, 1 - full prioritization)
        """
        super(StableReplayBuffer, self).__init__(size)
        self.scores = []
        self.lower_clip = None
        self.upper_clip = None

    def add(self, obs_t, action, reward, obs_tp1, done, score=None):
        """
        add a new transition to the buffer

        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        """
        idx = self._next_idx

        if self._next_idx >= len(self._storage):
            self.scores.append(score)
        else:
            self.scores[idx] = score

        super().add(obs_t, action, reward, obs_tp1, done)

    def sample(self, batch_size, **_kwargs):
        """
        Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.

        :param batch_size: (int) How many transitions to sample.
        :param beta: (float) To what degree to use importance weights (0 - no corrections, 1 - full correction)
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
            - weights: (numpy float) Array of shape (batch_size,) and dtype np.float32 denoting importance weight of
                each sampled transition
            - idxes: (numpy int) Array of shape (batch_size,) and dtype np.int32 idexes in buffer of sampled experiences
        """
        if not self.can_sample(batch_size):
            return self._encode_sample(list(range(len(self))))

        scores = np.array(self.scores)
        scores = np.clip(scores, np.percentile(scores, 10), np.percentile(scores, 90))
        idxs = np.random.choice(np.arange(len(self.scores)), size=(batch_size,), p=scores / np.sum(scores),
                                replace=False)

        return self._encode_sample(idxs)

    def update_priorities(self):
        """
        Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        :param idxes: ([int]) List of idxes of sampled transitions
        :param priorities: ([float]) List of updated priorities corresponding to transitions at the sampled idxes
            denoted by variable `idxes`.
        """
        self.lower_clip = np.percentile(self.scores, 10)
        self.upper_clip = np.percentile(self.scores, 90)

