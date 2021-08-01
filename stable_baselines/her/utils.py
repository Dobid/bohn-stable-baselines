from collections import OrderedDict

from stable_baselines.common.running_mean_std import RunningMeanStdSerial, RunningMeanStd
import pickle

import numpy as np
from gym import spaces

# Important: gym mixes up ordered and unordered keys
# and the Dict space may return a different order of keys that the actual one
KEY_ORDER = ['observation', 'achieved_goal', 'desired_goal']


class HERGoalEnvWrapper(object):
    """
    A wrapper that allow to use dict observation space (coming from GoalEnv) with
    the RL algorithms.
    It assumes that all the spaces of the dict space are of the same type.

    :param env: (gym.GoalEnv)
    """

    def __init__(self, env, norm=False, norm_reward=False, gamma=0.99, clip_obs=10, update_every=8):  # TODO: sync with other wrappers (probably has to be done outside this class)
        super(HERGoalEnvWrapper, self).__init__()
        self._env = env
        self.metadata = self._env.metadata
        self.action_space = env.action_space
        self.spaces = [env.observation_space[key] for key in KEY_ORDER]

        self.multi_step_obs = len(env.observation_space.spaces['observation'].shape) > 1
        # Check that all spaces are of the same type
        # (current limitation of the wrapper)
        space_types = [type(env.observation_space.spaces[key]) for key in KEY_ORDER]
        assert len(set(space_types)) == 1, "The spaces for goal and observation"\
                                           " must be of the same type"

        if isinstance(self.spaces[0], spaces.Discrete):
            self.obs_dim = 1
            self.goal_dim = 1
        else:
            goal_space_shape = env.observation_space.spaces['achieved_goal'].shape
            self.obs_dim = env.observation_space.spaces['observation'].shape[-1]
            self.goal_dim = goal_space_shape[-1]

            #if len(goal_space_shape) == 2:
            #    assert goal_space_shape[1] == 1, "Only 1D observation spaces are supported yet"
            #else:
            #    assert len(goal_space_shape) == 1, "Only 1D observation spaces are supported yet"

        if isinstance(self.spaces[0], spaces.MultiBinary):
            total_dim = self.obs_dim + 2 * self.goal_dim
            self.observation_space = spaces.MultiBinary(total_dim)

        elif isinstance(self.spaces[0], spaces.Box):
            if self.multi_step_obs:
                lows = np.concatenate([space.low for space in self.spaces], axis=1)
                highs = np.concatenate([space.high for space in self.spaces], axis=1)
            else:
                lows = np.concatenate([space.low for space in self.spaces])
                highs = np.concatenate([space.high for space in self.spaces])
            self.observation_space = spaces.Box(lows, highs, dtype=np.float32)

        elif isinstance(self.spaces[0], spaces.Discrete):
            dimensions = [env.observation_space.spaces[key].n for key in KEY_ORDER]
            self.observation_space = spaces.MultiDiscrete(dimensions)

        else:
            raise NotImplementedError("{} space is not supported".format(type(self.spaces[0])))

        self.norm = norm
        self.norm_reward = norm_reward
        self.training = True
        self.orig_obs = None
        self.epsilon = 1e-5
        self.gamma = gamma

        self.update_every = update_every

        self.disabled = False
        if norm:
            obs_norm_shape = [self.observation_space.shape[-1]]
            obs_norm_shape[-1] -= self.goal_dim
            self.obs_rms = RunningMeanStd(shape=obs_norm_shape)
            self.ret_rms = RunningMeanStd(shape=())
            if norm_reward:
                self.ret_rms = RunningMeanStd(shape=(), update_every=update_every)
            self.clip_obs = clip_obs
            self.ep_obs_data = []

    def convert_dict_to_obs(self, obs_dict):
        """
        :param obs_dict: (dict<np.ndarray>)
        :return: (np.ndarray)
        """
        # Note: achieved goal is not removed from the observation
        # this is helpful to have a revertible transformation
        if isinstance(self.observation_space, spaces.MultiDiscrete):
            # Special case for multidiscrete
            return np.concatenate([[int(obs_dict[key])] for key in KEY_ORDER])
        axis = 1 if self.multi_step_obs else 0
        return np.concatenate([obs_dict[key] for key in KEY_ORDER], axis=axis)

    def convert_obs_to_dict(self, observations):
        """
        Inverse operation of convert_dict_to_obs

        :param observations: (np.ndarray)
        :return: (OrderedDict<np.ndarray>)
        """
        if self.multi_step_obs:
            return OrderedDict([
                ('observation', observations[:, :self.obs_dim]),
                ('achieved_goal', observations[:, self.obs_dim:self.obs_dim + self.goal_dim]),
                ('desired_goal', observations[:, self.obs_dim + self.goal_dim:]),
            ])
        else:
            return OrderedDict([
                ('observation', observations[:self.obs_dim]),
                ('achieved_goal', observations[self.obs_dim:self.obs_dim + self.goal_dim]),
                ('desired_goal', observations[self.obs_dim + self.goal_dim:]),
            ])

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = self.convert_dict_to_obs(obs)
        if self.norm:
            self.orig_obs = np.copy(obs)
            obs = self.normalize_observation(obs, update=True)
            if self.norm_reward:
                reward = self.normalize_reward(reward, update=True)
            if len(self.ep_obs_data) > self.update_every:
                self.obs_rms.update(np.stack(self.ep_obs_data, axis=0))
                self.ep_obs_data = []
        return obs, reward, done, info

    def normalize_observation(self, obs, update):
        """
        :param obs: (numpy tensor)
        """
        if self.norm:
            is_dict = False
            if isinstance(obs, dict):
                is_dict = True
                obs = self.convert_dict_to_obs(obs)
            if self.training and update:
                if len(obs.shape) == 2:
                    self.ep_obs_data.append(obs[0, :-self.goal_dim])
                else:
                    self.ep_obs_data.append(obs[:-self.goal_dim])
            if self.obs_rms.count > 1:
                obs[..., :-self.goal_dim] = np.clip((obs[..., :-self.goal_dim] - self.obs_rms.mean) /
                                               np.sqrt(self.obs_rms.var + self.epsilon),
                                               -self.clip_obs, self.clip_obs)
                obs[..., -self.goal_dim:] = np.clip((obs[..., -self.goal_dim:] - self.obs_rms.mean[..., -self.goal_dim:]) /
                                               np.sqrt(self.obs_rms.var[..., -self.goal_dim:] + self.epsilon),
                                               -self.clip_obs, self.clip_obs)

            if is_dict:
                return self.convert_obs_to_dict(obs)
            else:
                return obs
        else:
            return obs

    def unnormalize_observation(self, obs):
        if self.norm:
            is_dict = False
            if isinstance(obs, dict):
                is_dict = True
                obs = self.convert_dict_to_obs(obs)
            obs = obs * np.sqrt(self.obs_rms.var + self.epsilon) + self.obs_rms.mean
            if is_dict:
                return self.convert_obs_to_dict(obs)
            else:
                return obs
        else:
            return obs

    def get_original_obs(self):
        """
        returns the unnormalized observation

        :return: (numpy float)
        """
        return self.orig_obs

    def normalize_reward(self, reward: np.ndarray, update=True) -> np.ndarray:
        """
        Normalize rewards using this VecNormalize's rewards statistics.
        Calling this method does not update statistics.
        """
        reward = reward / np.sqrt(self.ret_rms.var + self.epsilon)
        # if self.clip_reward is not None:
        #    reward = np.clip(reward, -self.clip_reward, self.clip_reward)
        if update:
            self._update_reward(reward)
        return reward

    def _update_reward(self, reward: np.ndarray) -> None:
        """Update reward normalization statistics."""
        self.ret = self.ret * self.gamma + reward
        self.ret_rms.update(np.array([self.ret]))

    def seed(self, seed=None):
        return self._env.seed(seed)

    def reset(self, *args, **kwargs):
        obs = self.convert_dict_to_obs(self._env.reset(*args, **kwargs))
        if self.norm:
            self.orig_obs = np.copy(obs)
            obs = self.normalize_observation(obs, update=True)
            self.ret = 0

        return obs

    def compute_reward(self, achieved_goal, desired_goal, info):
        if self.norm:
            achieved_goal = achieved_goal * np.sqrt(self.obs_rms.var[-self.goal_dim:] + self.epsilon) + \
                            self.obs_rms.mean[-self.goal_dim:]
            desired_goal = desired_goal * np.sqrt(self.obs_rms.var[-self.goal_dim:] + self.epsilon) + \
                            self.obs_rms.mean[-self.goal_dim:]

        rew = self._env.compute_reward(achieved_goal, desired_goal, info)
        if self.norm_reward:
            if isinstance(rew, list):
                for i in range(len(rew)):
                    rew[i] = self.normalize_reward(rew[i], update=False)
            else:
                rew = self.normalize_reward(rew, update=False)
        return rew

    def render(self, mode='human', **kwargs):
        return self._env.render(mode, **kwargs)

    def get_env_parameters(self, *args, **kwargs):
        return self._env.get_env_parameters(*args, **kwargs)

    def close(self):
        return self._env.close()

    def __getattr__(self, item):
        if item in self.__dict__:
            return getattr(self, item)
        else:
            return getattr(self._env, item)

    def save_running_average(self, path, suffix=None):
        """
        :param path: (str) path to log dir
        :param suffix: (str) suffix to file
        """
        file_names = ['obs_rms', 'ret_rms']
        if suffix is not None:
            file_names = [f + suffix for f in file_names]
        for rms, name in zip([self.obs_rms, self.ret_rms], file_names):
            with open("{}/{}.pkl".format(path, name), 'wb') as file_handler:
                pickle.dump(rms, file_handler)

    def load_running_average(self, path, suffix=None):
        """
        :param path: (str) path to log dir
        :param suffix: (str) suffix to file
        """

        file_names = ['obs_rms', 'ret_rms']
        for name in file_names:
            open_name = name
            if suffix is not None:
                open_name += suffix
            with open("{}/{}.pkl".format(path, open_name), 'rb') as file_handler:
                setattr(self, name, pickle.load(file_handler))


