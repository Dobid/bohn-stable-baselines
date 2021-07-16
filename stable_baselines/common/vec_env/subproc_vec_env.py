import multiprocessing
from collections import OrderedDict
from typing import Sequence

import gym
import numpy as np

from stable_baselines.common.vec_env.base_vec_env import VecEnv, CloudpickleWrapper


def _worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.var()
    done_not_reset = False
    reset_on_done = True
    reset_data = []
    scenarios_to_run = []
    gather_reset_data = False
    last_step_data = None
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                if not done_not_reset:
                    observation, reward, done, info = env.step(data)
                    if done and reset_on_done:#getattr(env, "training", True):
                        if gather_reset_data:
                            for i in range(5):
                                r_data = {"obs": env.reset()}
                                r_data["initial_state"] = env.get_initial_state()
                                reset_data.append(r_data)
                        info['terminal_observation'] = observation
                        done_not_reset = False
                        if len(scenarios_to_run) > 0:
                            scenario = scenarios_to_run.pop()
                            observation = env.reset(**scenario)
                        else:
                            observation = env.reset()
                    elif done:
                        done_not_reset = True
                        last_step_data = (observation, reward, done, info)
                    remote.send((observation, reward, done, info))
                else:
                    remote.send(last_step_data)
            elif cmd == "add_scenarios":
                if isinstance(data, list):
                    scenarios_to_run.extend(data)
                else:
                    scenarios_to_run.append(data)
            elif cmd == "get_reset_data":
                remote.send(reset_data)
                reset_data = []
            elif cmd == "set_gather_reset_data":
                gather_reset_data = data
            elif cmd == 'reset':
                observation = env.reset(*data[0], **data[1])
                done_not_reset = False
                last_step_data = None
                remote.send(observation)
            elif cmd == 'render':
                remote.send(env.render(*data[0], **data[1]))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            elif cmd == 'env_method':
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == 'get_attr':
                remote.send(getattr(env, data))
            elif cmd == 'set_attr':
                remote.send(setattr(env, data[0], data[1]))
            elif cmd == "set_reset_on_done":
                reset_on_done = data
            else:
                raise NotImplementedError
        except EOFError:
            break


class SubprocVecEnv(VecEnv):
    """
    Creates a multiprocess vectorized wrapper for multiple environments, distributing each environment to its own
    process, allowing significant speed up when the environment is computationally complex.

    For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
    number of logical cores on your CPU.

    .. warning::

        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important when TensorFlow sessions or other non thread-safe
        libraries are used in the parent (see issue #217). However, compared to
        'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods, users must wrap the code in an
        ``if __name__ == "__main__":`` block.
        For more information, see the multiprocessing documentation.

    :param env_fns: ([callable]) A list of functions that will create the environments
        (each callable returns a `Gym.Env` instance when called).
    :param start_method: (str) method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    """

    def __init__(self, env_fns, start_method=None, reset_on_done=True, sampler_manager=None):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = 'forkserver' in multiprocessing.get_all_start_methods()
            start_method = 'forkserver' if forkserver_available else 'spawn'
        ctx = multiprocessing.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe(duplex=True) for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.sampler_manager = sampler_manager
        self.reset_on_done = reset_on_done
        for remote in self.remotes:
            remote.send(("set_reset_on_done", reset_on_done))
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos

    def seed(self, seed=None):
        for idx, remote in enumerate(self.remotes):
            remote.send(('seed', seed + idx))
        return [remote.recv() for remote in self.remotes]

    def reset(self, indices=None, *args, **kwargs):
        if kwargs and (isinstance(kwargs["state"], list) or isinstance(kwargs["state"], np.ndarray)):
            kwargs_list = True
        else:
            kwargs_list = False
        for i, remote in enumerate(self._get_target_remotes(indices)):
            if kwargs_list:
                kwargs_i = {k: v[i] for k, v in kwargs.items()}
            else:
                kwargs_i = kwargs
            remote.send(('reset', (args, {**kwargs_i})))
        obs = [remote.recv() for i, remote in enumerate(self._get_target_remotes(indices))]
        return _flatten_obs(obs, self.observation_space)

    def add_scenarios(self, scenarios, indices=None):
        remotes = self._get_target_remotes(indices)
        remote_data = np.array_split(scenarios, self.num_envs)
        for i, pipe in enumerate(remotes):
            pipe.send(('add_scenarios', list(remote_data[i])))

    def get_reset_data(self, indices=None):
        reset_data = []
        for pipe in self._get_target_remotes(indices):
            pipe.send(('get_reset_data', None))

        for pipe in self._get_target_remotes(indices):
            reset_data.extend(pipe.recv())

        return reset_data

    def set_gather_reset_data(self, status, indices=None):
        for pipe in self._get_target_remotes(indices):
            pipe.send(('set_gather_reset_data', status))

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
        self.closed = True

    def render(self, indices=None, *args, **kwargs):
        for pipe in self._get_target_remotes(indices):
            # gather images from subprocesses
            # `mode` will be taken into account later
            pipe.send(('render', (args, {**kwargs})))

        figs = [pipe.recv() for pipe in self._get_target_remotes(indices)]
        if isinstance(indices, int):
            figs = figs[0]

        return figs
        # Create a big image by tiling images from subprocesses
        bigimg = tile_images(imgs)
        if mode == 'human':
            import cv2
            cv2.imshow('vecenv', bigimg[:, :, ::-1])
            cv2.waitKey(1)
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError

    def get_images(self):
        for pipe in self.remotes:
            pipe.send(('render', {"mode": 'rgb_array'}))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def get_attr(self, attr_name, indices=None):
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('get_attr', attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('set_attr', (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('env_method', (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def _get_target_remotes(self, indices):
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.

        :param indices: (None,int,Iterable) refers to indices of envs.
        :return: ([multiprocessing.Connection]) Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]

    def convert_obs_to_dict(self, obs):
        self.remotes[0].send(("env_method", ("convert_obs_to_dict", [obs], {})))
        return self.remotes[0].recv()

    def convert_dict_to_obs(self, obs):
        self.remotes[0].send(("env_method", ("convert_dict_to_obs", [obs], {})))
        return self.remotes[0].recv()

    def compute_reward(self, a_goal, d_goal, info):
        self.remotes[0].send(("env_method", ("compute_reward", [a_goal, d_goal, info], {})))
        return self.remotes[0].recv()

    def load_running_average(self, path, suffix=None):
        target_remotes = self._get_target_remotes(None)
        for remote in target_remotes:
            remote.send(("env_method", ("load_running_average", [path, suffix], {})))

        return [remote.recv() for remote in target_remotes]

    def save_running_average(self, path, suffix=None):
        target_remotes = self._get_target_remotes(None)
        for remote in target_remotes:
            remote.send(("env_method", ("save_running_average", [path, suffix], {})))

        return [remote.recv() for remote in target_remotes]


def _flatten_obs(obs, space):
    """
    Flatten observations, depending on the observation space.

    :param obs: (list<X> or tuple<X> where X is dict<ndarray>, tuple<ndarray> or ndarray) observations.
                A list or tuple of observations, one per environment.
                Each environment observation may be a NumPy array, or a dict or tuple of NumPy arrays.
    :return (OrderedDict<ndarray>, tuple<ndarray> or ndarray) flattened observations.
            A flattened NumPy array or an OrderedDict or tuple of flattened numpy arrays.
            Each NumPy array has the environment index as its first axis.
    """
    assert isinstance(obs, (list, tuple)), "expected list or tuple of observations per environment"
    assert len(obs) > 0, "need observations from at least one environment"

    if isinstance(space, gym.spaces.Dict):
        assert isinstance(space.spaces, OrderedDict), "Dict space must have ordered subspaces"
        assert isinstance(obs[0], dict), "non-dict observation for environment with Dict observation space"
        return OrderedDict([(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()])
    elif isinstance(space, gym.spaces.Tuple):
        assert isinstance(obs[0], tuple), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple((np.stack([o[i] for o in obs]) for i in range(obs_len)))
    else:
        return np.stack(obs)
