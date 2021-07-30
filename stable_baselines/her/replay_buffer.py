import copy
from enum import Enum
import scipy.signal

import numpy as np

from stable_baselines.common.buffers import StableReplayBuffer


class GoalSelectionStrategy(Enum):
    """
    The strategies for selecting new goals when
    creating artificial transitions.
    """
    # Select a goal that was achieved
    # after the current step, in the same episode
    FUTURE = 0
    # Select the goal that was achieved
    # at the end of the episode
    FINAL = 1
    # Select a goal that was achieved in the episode
    EPISODE = 2
    # Select a goal that was achieved
    # at some point in the training procedure
    # (and that is present in the replay buffer)
    RANDOM = 3
    # Select a stable goal that was achieved (stable as in achieved in many consecutive steps)
    # after the current step, in the same episode
    FUTURE_STABLE = 4
    HORIZON = 5


# For convenience
# that way, we can use string to select a strategy
KEY_TO_GOAL_STRATEGY = {
    "future_stable": GoalSelectionStrategy.FUTURE_STABLE,
    'future': GoalSelectionStrategy.FUTURE,
    'final': GoalSelectionStrategy.FINAL,
    'episode': GoalSelectionStrategy.EPISODE,
    'random': GoalSelectionStrategy.RANDOM,
    "horizon": GoalSelectionStrategy.HORIZON
}


class HindsightExperienceReplayWrapper(object):
    """
    Wrapper around a replay buffer in order to use HER.
    This implementation is inspired by to the one found in https://github.com/NervanaSystems/coach/.

    :param replay_buffer: (ReplayBuffer)
    :param n_sampled_goal: (int) The number of artificial transitions to generate for each actual transition
    :param goal_selection_strategy: (GoalSelectionStrategy) The method that will be used to generate
        the goals for the artificial transitions.
    :param wrapped_env: (HERGoalEnvWrapper) the GoalEnv wrapped using HERGoalEnvWrapper,
        that enables to convert observation to dict, and vice versa
    """
    __name__ = "HindsightExperienceReplayWrapper"

    def __init__(self, replay_buffer, n_sampled_goal, goal_selection_strategy, wrapped_env, her_starts=0):
        super(HindsightExperienceReplayWrapper, self).__init__()

        assert isinstance(goal_selection_strategy, GoalSelectionStrategy), "Invalid goal selection strategy," \
                                                                           "please use one of {}".format(
            list(GoalSelectionStrategy))

        self.n_sampled_goal = n_sampled_goal
        self.goal_selection_strategy = goal_selection_strategy
        self.env = wrapped_env
        self.num_envs = getattr(self.env, "num_envs", 1)
        # Buffer for storing transitions of the current episode
        self.episode_transitions = [[] for i in range(self.num_envs)]
        if self.goal_selection_strategy == GoalSelectionStrategy.FUTURE_STABLE:
            self.stable_indices = []
            self.stable_max_change = np.array([0.025, 0.025, 0.025])#np.array([5, 5, 5]) * self.env.env.simulator.dt
            #self.legal_goal_low, self.legal_goal_high = self.env.env.get_goal_limits()
        self.replay_buffer = replay_buffer
        self.her_starts = her_starts
        self.use_her = False
        self.require_change = True
        self.reward_transformation = None
        if hasattr(self.env, "get_attr"):
            self.multi_step_obs = self.env.get_attr("multi_step_obs")
        else:
            self.multi_step_obs = getattr(self.env, "multi_step_obs", False)

        if "Recurrent" in replay_buffer.__name__:
            self.recurrent = True
        else:
            self.recurrent = False

    def add(self, obs_t, action, reward, obs_tp1, done, env_i=0, **extra_data):
        """
        add a new transition to the buffer

        :param obs_t: (np.ndarray) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (np.ndarray) the new observation
        :param done: (bool) is the episode done
        """
        assert self.replay_buffer is not None
        # Update current episode buffer
        self.episode_transitions[env_i].append((obs_t, action, reward, obs_tp1, done, *[extra_data[k] for k in sorted(extra_data)]))
        if self.goal_selection_strategy == GoalSelectionStrategy.FUTURE_STABLE:
            # Store information about typical change in achieved goal (should consider if desired goal also changes)
            pass
        if done:
            # Add transitions (and imagined ones) to buffer only when an episode is over
            self._store_episode(env_i)
            # Reset episode buffer
            self.episode_transitions[env_i] = []

    def extend(self, *args, **kwargs):
        for env_i in range(self.num_envs):
            self.add(*args, **kwargs, env_i=env_i)

    def sample(self, *args, **kwargs):
        batch = self.replay_buffer.sample(*args, **kwargs)
        if self.replay_buffer.__name__ == "RecurrentReplayBuffer" and self.replay_buffer.scan_length > 0 and False:
            # TODO: i think it is guaranteed that there are always scan_length instances but im not sure
            obs, next_obs = batch[0], batch[3]
            sampled_goals = obs[::self.replay_buffer.scan_length + 1, -self.env.goal_dim:]
            sampled_goals = np.repeat(sampled_goals, self.replay_buffer.scan_length + 1, axis=0)
            obs[:, -self.env.goal_dim:] = sampled_goals
            next_obs[:, -self.env.goal_dim:] = sampled_goals
            batch[0], batch[3] = obs, next_obs

        return batch

    def can_sample(self, n_samples):
        """
        Check if n_samples samples can be sampled
        from the buffer.

        :param n_samples: (int)
        :return: (bool)
        """
        return self.replay_buffer.can_sample(n_samples)

    def __len__(self):
        return len(self.replay_buffer)

    def _sample_achieved_goal(self, episode_transitions, transition_idx):
        """
        Sample an achieved goal according to the sampling strategy.

        :param episode_transitions: ([tuple]) a list of all the transitions in the current episode
        :param transition_idx: (int) the transition to start sampling from
        :return: (np.ndarray) an achieved goal
        """
        if self.goal_selection_strategy == GoalSelectionStrategy.HORIZON:
            selected_idx = np.random.randint(transition_idx + 1,
                                             min(int(transition_idx + 2 + 0.05 * len(episode_transitions)),
                                                 len(episode_transitions)))
            selected_transition = episode_transitions[selected_idx]
        elif self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
            # Sample a goal that was observed in the same episode after the current step
            selected_idx = np.random.randint(transition_idx + 1, len(episode_transitions))
            selected_transition = episode_transitions[selected_idx]
        elif self.goal_selection_strategy == GoalSelectionStrategy.FUTURE_STABLE:
            weights = 1 / (self.stable_indices[self.stable_indices > transition_idx] - transition_idx + 1)
            selected_idx = np.random.choice(self.stable_indices[self.stable_indices > transition_idx], p=weights / np.sum(weights))
            selected_transition = episode_transitions[selected_idx]
        elif self.goal_selection_strategy == GoalSelectionStrategy.FINAL:
            # Choose the goal achieved at the end of the episode
            selected_transition = episode_transitions[-1]
        elif self.goal_selection_strategy == GoalSelectionStrategy.EPISODE:
            # Random goal achieved during the episode
            selected_idx = np.random.choice(np.arange(len(episode_transitions)))
            selected_transition = episode_transitions[selected_idx]
        elif self.goal_selection_strategy == GoalSelectionStrategy.RANDOM:
            # Random goal achieved, from the entire replay buffer
            selected_idx = np.random.choice(np.arange(len(self.replay_buffer)))
            selected_transition = self.replay_buffer.storage[selected_idx]
        else:
            raise ValueError("Invalid goal selection strategy,"
                             "please use one of {}".format(list(GoalSelectionStrategy)))
        return self.env.convert_obs_to_dict(selected_transition[0])['achieved_goal']

    def _sample_achieved_goals(self, episode_transitions, transition_idx):
        """
        Sample a batch of achieved goals according to the sampling strategy.

        :param episode_transitions: ([tuple]) list of the transitions in the current episode
        :param transition_idx: (int) the transition to start sampling from
        :return: (np.ndarray) an achieved goal
        """
        return [
            self._sample_achieved_goal(episode_transitions, transition_idx)
            for _ in range(self.n_sampled_goal)
        ]

    def _store_episode(self, env_i=0):
        """
        Sample artificial goals and store transition of the current
        episode in the replay buffer.
        This method is called only after each end of episode.
        """
        # For each transition in the last episode,
        # create a set of artificial transitions
        if self.replay_buffer.__name__ == "StableReplayBuffer" or self.goal_selection_strategy == GoalSelectionStrategy.FUTURE_STABLE:
            achieved_goal_changes = []
            achieved_goals = []
            for transition_idx, transition in enumerate(self.episode_transitions[env_i]):
                obs_t, action, reward, obs_tp1, done = transition
                obs_dict, next_obs_dict = map(self.env.convert_obs_to_dict, (obs_t, obs_tp1))
                if self.env.multi_dimensional_obs:
                    change = next_obs_dict["achieved_goal"][0, :] - obs_dict["achieved_goal"][0, :] - (
                            next_obs_dict["desired_goal"][0, :] - obs_dict["desired_goal"][0, :])
                else:
                    change = next_obs_dict["achieved_goal"] - obs_dict["achieved_goal"] - (
                            next_obs_dict["desired_goal"] - obs_dict["desired_goal"])
                achieved_goal_changes.append(change)
                achieved_goals.append(obs_dict["achieved_goal"])
            if self.goal_selection_strategy == GoalSelectionStrategy.FUTURE_STABLE:
                achieved_goals = np.array(achieved_goals)
                achieved_goal_changes = np.array(achieved_goal_changes)
                tot_achieved_goal_changes = np.sum(np.abs(achieved_goal_changes).flatten())
                achieved_goal_changes = np.abs(scipy.signal.fftconvolve(achieved_goal_changes,
                                                                        np.ones((5, achieved_goal_changes.shape[1]),
                                                                                dtype=int), 'valid', axes=0))
                self.stable_indices = np.nonzero(np.all(achieved_goal_changes <= self.stable_max_change, axis=1))[0] + 2
                self.stable_indices = self.stable_indices[
                    np.all((achieved_goals[self.stable_indices] - achieved_goals[0]) >= np.array([0.05, 0.05, 0]))]
                # self.stable_indices = self.stable_indices[np.all(achieved_goals[self.stable_indices] >= self.legal_goal_low, axis=1) & np.all(achieved_goals[self.stable_indices] <= self.legal_goal_high, axis=1)]

        if self.replay_buffer.__name__ == "EpisodicRecurrentReplayBuffer":
            for transition in self.episode_transitions[env_i]:
                self.replay_buffer.add(*transition)
            self.replay_buffer.store_episode(self.episode_transitions[env_i][-1][-1])
            sampled_goals = self._sample_achieved_goals(self.episode_transitions, 0)
            for goal in sampled_goals:
                for transition_idx, transition in enumerate(self.episode_transitions[env_i]):
                    obs, action, reward, next_obs, done, *extra_data = copy.deepcopy(transition)
                    # Convert concatenated obs to dict, so we can update the goals
                    obs_dict, next_obs_dict = map(self.env.convert_obs_to_dict, (obs, next_obs))
                    # Update the desired goal in the transition
                    obs_dict['desired_goal'] = goal
                    next_obs_dict['desired_goal'] = goal

                    # Update the reward according to the new desired goal
                    prev_state = obs_dict["achieved_goal"]
                    achieved_goal = next_obs_dict['achieved_goal']
                    desired_goal = goal
                    if self.env.multi_dimensional_obs:
                        prev_state = prev_state[0]
                        achieved_goal = achieved_goal[0]
                        desired_goal = goal[0]
                    info = {"step": transition_idx, "prev_state": prev_state, "action": action}
                    reward = self.env.compute_reward(achieved_goal, desired_goal, info)
                    # Can we use achieved_goal == desired_goal

                    if done:
                        extra_data = {"my": extra_data[-1]}
                    else:
                        extra_data = {}

                    self.replay_buffer.add(obs, action, reward, next_obs, done, goal, **extra_data)
                self.replay_buffer.store_episode(self.episode_transitions[-1][-1])
        else:
            for transition_idx, transition in enumerate(self.episode_transitions[env_i]):
                obs_t, action, reward, obs_tp1, done, *extra_data = transition
                # Add to the replay buffer
                if self.replay_buffer.__name__ == "StableReplayBuffer":
                    self.replay_buffer.add(obs_t, action, reward, obs_tp1, done, tot_achieved_goal_changes)
                else:
                    self.replay_buffer.add(obs_t, action, reward, obs_tp1, done, *extra_data)

                if self.goal_selection_strategy == GoalSelectionStrategy.FUTURE_STABLE and (
                        self.stable_indices.shape[0] == 0 or transition_idx >= self.stable_indices[-1]):
                    continue

                # We cannot sample a goal from the future in the last step of an episode
                if transition_idx == len(self.episode_transitions[env_i]) - 1 and self.goal_selection_strategy in [GoalSelectionStrategy.FUTURE, GoalSelectionStrategy.HORIZON]:
                    break
                elif transition_idx >= len(
                        self.episode_transitions[env_i]) - 2 and self.goal_selection_strategy == GoalSelectionStrategy.FUTURE_STABLE:
                    continue

                # Sampled n goals per transition, where n is `n_sampled_goal`
                # this is called k in the paper
                sampled_goals = self._sample_achieved_goals(self.episode_transitions[env_i], transition_idx)
                # For each sampled goals, store a new transition
                for goal in sampled_goals:
                    # Copy transition to avoid modifying the original one
                    obs, action, reward, next_obs, done, *extra_data = copy.deepcopy(transition)

                    # Convert concatenated obs to dict, so we can update the goals
                    obs_dict, next_obs_dict = map(self.env.convert_obs_to_dict, (obs, next_obs))
                    # Update the desired goal in the transition
                    obs_dict['desired_goal'] = goal
                    next_obs_dict['desired_goal'] = goal

                    prev_state = obs_dict["achieved_goal"]
                    achieved_goal = next_obs_dict['achieved_goal']
                    desired_goal = goal
                    if self.multi_step_obs:
                        prev_state = prev_state[0]
                        achieved_goal = achieved_goal[0]
                        desired_goal = goal[0]
                    info = {"step": transition_idx + 1, "prev_state": prev_state, "action": action}
                    reward = self.env.compute_reward(achieved_goal, desired_goal, info, indices=env_i)[0]
                    # Can we use achieved_goal == desired_goal?
                    done = False

                    # Transform back to ndarrays
                    obs, next_obs = map(self.env.convert_dict_to_obs, (obs_dict, next_obs_dict))

                    # Add artificial transition to the replay buffer
                    if self.replay_buffer.__name__ == "StableReplayBuffer":
                        self.replay_buffer.add(obs, action, reward, next_obs, done, tot_achieved_goal_changes)
                    elif self.replay_buffer.__name__ == "DRRecurrentReplayBuffer":
                        self.replay_buffer.add_her(goal, reward, transition_idx)
                    else:
                        self.replay_buffer.add(obs, action, reward, next_obs, done, *extra_data)
            if self.replay_buffer.__name__ == "RecurrentReplayBuffer":
                self.replay_buffer.store_episode()

    def update_state(self, *args, **kwargs):
        return self.replay_buffer.update_state(*args, **kwargs)
