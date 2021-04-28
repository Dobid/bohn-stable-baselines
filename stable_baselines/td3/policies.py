import tensorflow as tf
import numpy as np
from gym.spaces import Box
import copy
from stable_baselines.common.policies import BasePolicy, nature_cnn, register_policy, cnn_1d_extractor
from stable_baselines.sac.policies import mlp
from stable_baselines.a2c.utils import lstm, batch_to_seq, seq_to_batch


class TD3Policy(BasePolicy):
    """
    Policy object that implements a TD3-like actor critic

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param scale: (bool) whether or not to scale the input
    """

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, scale=False,
                 add_action_ph=False):
        super(TD3Policy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=scale,
                                        add_action_ph=add_action_ph)
        assert isinstance(ac_space, Box), "Error: the action space must be of type gym.spaces.Box"

        self.qf1 = None
        self.qf2 = None
        self.q_discrepancy = None
        self.policy = None

    def make_actor(self, obs=None, reuse=False, scope="pi"):
        """
        Creates an actor object

        :param obs: (TensorFlow Tensor) The observation placeholder (can be None for default placeholder)
        :param reuse: (bool) whether or not to reuse parameters
        :param scope: (str) the scope name of the actor
        :return: (TensorFlow Tensor) the output tensor
        """
        raise NotImplementedError

    def make_critics(self, obs=None, action=None, reuse=False,
                     scope="qvalues_fn"):
        """
        Creates the two Q-Values approximator

        :param obs: (TensorFlow Tensor) The observation placeholder (can be None for default placeholder)
        :param action: (TensorFlow Tensor) The action placeholder
        :param reuse: (bool) whether or not to reuse parameters
        :param scope: (str) the scope name
        :return: ([tf.Tensor]) Mean, action and log probability
        """
        raise NotImplementedError

    def step(self, obs, state=None, mask=None):
        """
        Returns the policy for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float]) actions
        """
        raise NotImplementedError

    def proba_step(self, obs, state=None, mask=None):
        """
        Returns the policy for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float]) actions
        """
        return self.step(obs, state, mask)


class FeedForwardPolicy(TD3Policy):
    """
    Policy object that implements a DDPG-like actor critic, using a feed forward neural network.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) The size of the Neural network for the policy (if None, default to [64, 64])
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param layer_norm: (bool) enable layer normalisation
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, layers=None,
                 cnn_extractor=nature_cnn, feature_extraction="cnn",
                 layer_norm=False, act_fun=tf.nn.relu, obs_module_indices=None, **kwargs):
        super(FeedForwardPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                                                reuse=reuse,
                                                scale=(feature_extraction == "cnn" and cnn_extractor == nature_cnn))

        if isinstance(act_fun, str):
            act_fun = getattr(tf.nn, act_fun)

        self._kwargs_check(feature_extraction, kwargs)
        self.layer_norm = layer_norm
        self.feature_extraction = feature_extraction
        self.cnn_kwargs = kwargs
        self.cnn_extractor = cnn_extractor
        self.cnn_vf = self.cnn_kwargs.pop("cnn_vf", True)
        self.reuse = reuse
        if layers is None:
            layers = [64, 64]
        self.layers = layers
        self.obs_module_indices = obs_module_indices
        self.policy_pre_activation = None

        assert len(layers) >= 1, "Error: must have at least one hidden layer for the policy."

        self.activ_fn = act_fun

    def make_actor(self, obs=None, reuse=False, scope="pi"):
        if obs is None:
            obs = self.processed_obs

        if self.obs_module_indices is not None:
            obs = tf.gather(obs, self.obs_module_indices["pi"], axis=-1)
        with tf.variable_scope(scope, reuse=reuse):
            if self.feature_extraction == "cnn":
                pi_h = self.cnn_extractor(obs, name="pi_c1", act_fun=self.activ_fn, **self.cnn_kwargs)
            else:
                pi_h = tf.layers.flatten(obs)

            pi_h = mlp(pi_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)

            self.policy_pre_activation = tf.layers.dense(pi_h, self.ac_space.shape[0])
            self.policy = policy = tf.tanh(self.policy_pre_activation)

        return policy

    def make_critics(self, obs=None, action=None, reuse=False, scope="values_fn", extracted_callback=None):
        if obs is None:
            obs = self.processed_obs

        if self.obs_module_indices is not None:
            obs = tf.gather(obs, self.obs_module_indices["vf"], axis=-1)

        with tf.variable_scope(scope, reuse=reuse):
            if self.feature_extraction == "cnn" and self.cnn_vf:
                critics_h = self.cnn_extractor(obs, name="vf_c1", act_fun=self.activ_fn, **self.cnn_kwargs)
            else:
                critics_h = tf.layers.flatten(obs)

            if extracted_callback is not None:
                critics_h = extracted_callback(critics_h)

            # Concatenate preprocessed state and action
            qf_h = tf.concat([critics_h, action], axis=-1)

            # Double Q values to reduce overestimation
            with tf.variable_scope('qf1', reuse=reuse):
                qf1_h = mlp(qf_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)
                qf1 = tf.layers.dense(qf1_h, 1, name="qf1")

            with tf.variable_scope('qf2', reuse=reuse):
                qf2_h = mlp(qf_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)
                qf2 = tf.layers.dense(qf2_h, 1, name="qf2")

            self.qf1 = qf1
            self.qf2 = qf2
            # TODO: assumes that all qf1 and qf2 can never have opposite signs
            #self.q_discrepancy = tf.square(self.qf1 - self.qf2) / tf.square(tf.maximum(self.qf1, self.qf2))
            #self.q_discrepancy = tf.abs(self.qf1 - self.qf2)

        return self.qf1, self.qf2

    def step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy, {self.obs_ph: obs})

    def get_q_discrepancy(self, obs):
        if isinstance(obs, np.ndarray) and len(obs.shape) == 1: # TODO: check for MLP or CNN policy here
            obs = np.expand_dims(obs, axis=0)
        return self.sess.run(self.q_discrepancy, {self.obs_ph: obs})


class RecurrentPolicy(TD3Policy):
    """
        Policy object that implements a DDPG-like actor critic, using a feed forward neural network.

        :param sess: (TensorFlow session) The current TensorFlow session
        :param ob_space: (Gym Space) The observation space of the environment
        :param ac_space: (Gym Space) The action space of the environment
        :param n_env: (int) The number of environments to run
        :param n_steps: (int) The number of steps to run for each environment
        :param n_batch: (int) The number of batch to run (n_envs * n_steps)
        :param reuse: (bool) If the policy is reusable or not
        :param layers: ([int]) The size of the Neural network for the policy (if None, default to [64, 64])
        :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
        :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
        :param layer_norm: (bool) enable layer normalisation
        :param act_fun: (tf.func) the activation function to use in the neural network.
        :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
        """
    recurrent = True

    def __init__(self, sess, ob_space, ac_space, layers, n_env=1, n_steps=1, n_batch=None, reuse=False,
                 cnn_extractor=nature_cnn, feature_extraction="mlp", n_lstm=128, share_lstm=False, save_state=False,
                 save_target_state=False, layer_norm=False, act_fun=tf.nn.relu, obs_module_indices=None, **kwargs):
        super(RecurrentPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                                                reuse=reuse, add_action_ph=True,
                                                scale=(feature_extraction == "cnn" and cnn_extractor == nature_cnn))

        self._kwargs_check(feature_extraction, kwargs)
        self.layer_norm = layer_norm
        self.feature_extraction = feature_extraction
        self.cnn_kwargs = kwargs
        self.cnn_extractor = cnn_extractor
        self.cnn_vf = self.cnn_kwargs.pop("cnn_vf", True)
        self.reuse = reuse
        self.layers = layers
        self.obs_module_indices = obs_module_indices

        self.activ_fn = act_fun
        self.n_lstm = n_lstm
        self.share_lstm = share_lstm
        self._obs_ph = self.processed_obs  # Base class has self.obs_ph as property getting self._obs_ph
        self.obs_tp1_ph = self.processed_obs

        assert self.n_batch % self.n_steps == 0, "The batch size must be a multiple of sequence length (n_steps)"
        self._lstm_n_batch = self.n_batch // self.n_steps

        self.action_prev = np.zeros((1, *self.ac_space.shape))

        self._initial_state = np.zeros((self._lstm_n_batch, self.n_lstm * 2), dtype=np.float32)
        if self.share_lstm:
            self.state = None
        else:
            self.pi_state = None
            self.qf1_state = None
            self.qf2_state = None

        with tf.variable_scope("input", reuse=False):
            self.dones_ph = tf.placeholder_with_default(np.zeros((self.n_batch,), dtype=np.float32), (self.n_batch,), name="dones_ph")  # (done t-1)
            if self.share_lstm:
                self.state_ph = tf.placeholder_with_default(self.initial_state, (self._lstm_n_batch, self.n_lstm * 2), name="state_ph")
            else:
                self.pi_state_ph = tf.placeholder_with_default(self.initial_state, (self._lstm_n_batch, self.n_lstm * 2), name="pi_state_ph")
                self.qf1_state_ph = tf.placeholder_with_default(self.initial_state, (self._lstm_n_batch, self.n_lstm * 2), name="qf1_state_ph")
                self.qf2_state_ph = tf.placeholder_with_default(self.initial_state, (self._lstm_n_batch, self.n_lstm * 2), name="qf2_state_ph")

            self.action_prev_ph = tf.placeholder(np.float32, (self.n_batch, *self.ac_space.shape), name="action_prev_ph")

        self.save_state = save_state
        self.save_target_state = save_target_state

        self.extra_phs = ["action_prev"]
        self.rnn_inputs = ["obs", "action_prev"]
        self.extra_data_names = ["action_prev"]

        if self.save_target_state:
            self.extra_data_names = sorted(self.extra_data_names + ["target_action_prev"])
            self.rnn_inputs = sorted(self.rnn_inputs + ["obs_tp1"])
            self.extra_phs = sorted(self.extra_phs + ["target_action_prev"])

        if self.save_state:
            state_names = ["state"] if self.share_lstm else ["pi_state", "qf1_state", "qf2_state"]
            if self.save_target_state:
                state_names.extend(["target_" + state_name for state_name in state_names])
            if self.share_lstm:
                self.extra_data_names = sorted(self.extra_data_names + state_names)
                self.extra_phs = sorted(self.extra_phs + state_names)
            else:
                self.extra_data_names = sorted(self.extra_data_names + state_names)
                self.extra_phs = sorted(self.extra_phs + state_names)

    def _process_phs(self, **phs):
        for ph_name, ph_val in phs.items():
            if ph_val is None:
                phs[ph_name] = getattr(self, ph_name + "_ph")
            else:
                try:
                    setattr(self, ph_name + "_ph", ph_val)
                except AttributeError:
                    setattr(self, "_" + ph_name + "_ph", ph_val)

        return phs.values()

    def _make_branch(self, branch_name, input_tensor, dones=None, state_ph=None):
        if branch_name == "lstm":
            for i, fc_layer_units in enumerate(self.layers["lstm"]):
                input_tensor = self.activ_fn(tf.layers.dense(input_tensor, fc_layer_units, name="lstm_fc{}".format(i)))

            input_tensor = batch_to_seq(input_tensor, self._lstm_n_batch, self.n_steps)
            masks = batch_to_seq(dones, self._lstm_n_batch, self.n_steps)
            input_tensor, state = lstm(input_tensor, masks, state_ph, "lstm", n_hidden=self.n_lstm,
                                       layer_norm=self.layer_norm)
            input_tensor = seq_to_batch(input_tensor)

            return input_tensor, state
        else:
            for i, fc_layer_units in enumerate(self.layers[branch_name]):
                input_tensor = self.activ_fn(tf.layers.dense(input_tensor, fc_layer_units, name="{}_fc{}".format(branch_name, i)))

            return input_tensor

    def make_actor(self, ff_phs=None, rnn_phs=None, dones=None, reuse=False, scope="pi"):
        lstm_branch = tf.concat([tf.layers.flatten(ph) for ph in rnn_phs], axis=-1)
        if ff_phs is not None:
            ff_branch = tf.concat([tf.layers.flatten(ph) for ph in ff_phs], axis=-1)

        if dones is None:
            dones = self.dones_ph

        if self.share_lstm:
            with tf.variable_scope("shared", reuse=tf.AUTO_REUSE):
                lstm_branch, self.state = self._make_branch("lstm", lstm_branch, dones, self.state_ph)

        with tf.variable_scope(scope, reuse=reuse):
            if self.layers["ff"] is not None:
               ff_branch = self._make_branch("ff", ff_branch)

            if not self.share_lstm:
                lstm_branch, self.pi_state = self._make_branch("lstm", lstm_branch, dones, self.pi_state_ph)

            if ff_phs is not None:
                head = tf.concat([ff_branch, lstm_branch], axis=-1)
            else:
                head = lstm_branch

            head = self._make_branch("head", head)

            self.policy_pre_activation = tf.layers.dense(head, self.ac_space.shape[0])
            self.policy = policy = tf.tanh(self.policy_pre_activation)

        return policy

    def make_critics(self, ff_phs=None, rnn_phs=None, dones=None, reuse=False, scope="values_fn"):
        lstm_branch_in = tf.concat([tf.layers.flatten(ph) for ph in rnn_phs], axis=-1)
        if ff_phs is not None:
            ff_branch_in = tf.concat([tf.layers.flatten(ph) for ph in ff_phs], axis=-1)

        if dones is None:
            dones = self.dones_ph

        self.qf1, self.qf2 = None, None
        self.qf1_state, self.qf2_state = None, None

        if self.share_lstm:
            with tf.variable_scope("shared", reuse=tf.AUTO_REUSE):
                lstm_branch_s, self.state = self._make_branch("lstm", lstm_branch_in, dones, self.state_ph)

        with tf.variable_scope(scope, reuse=reuse):
            # Double Q values to reduce overestimation
            for qf_i in range(1, 3):
                with tf.variable_scope('qf{}'.format(qf_i), reuse=reuse):
                    lstm_branch = lstm_branch_in
                    if self.layers["ff"] is not None:
                        ff_branch = self._make_branch("ff", ff_branch_in)
                    elif ff_phs is not None:
                        ff_branch = ff_branch_in

                    if not self.share_lstm:
                        lstm_branch, state = self._make_branch("lstm", lstm_branch, dones,
                                                               getattr(self, "qf{}_state_ph".format(qf_i)))
                        setattr(self, "qf{}_state".format(qf_i), state)
                    else:
                        lstm_branch = lstm_branch_s

                    if ff_phs is not None:
                        head = tf.concat([ff_branch, lstm_branch], axis=-1)
                    else:
                        head = lstm_branch

                    head = self._make_branch("head", head)

                    setattr(self, "qf{}".format(qf_i), tf.layers.dense(head, 1, name="qf{}".format(qf_i)))

        return self.qf1, self.qf2

    def step(self, obs, action_prev=None, state=None, mask=None, feed_dict=None, **kwargs):
        if feed_dict is None:
            feed_dict = {}
        if state is None:
            state = self.initial_state
        if mask is None:
            mask = np.array([False])
        if action_prev is None:
            assert obs.shape[0] == 1
            if mask[0]:
                self.action_prev = np.zeros((1, *self.ac_space.shape))
            action_prev = self.action_prev

        rnn_node = self.state if self.share_lstm else self.pi_state
        state_ph = self.state_ph if self.share_lstm else self.pi_state_ph

        feed_dict.update({self.obs_ph: obs, state_ph: state, self.dones_ph: mask,
                                      self.action_prev_ph: action_prev})

        action, out_state = self.sess.run([self.policy, rnn_node], feed_dict)
        self.action_prev = action

        return action, out_state

    @property
    def initial_state(self):
        return self._initial_state

    def collect_data(self, _locals, _globals):
        data = {}
        if self.save_state:
            if self.share_lstm:
                data["state"] = _locals["prev_policy_state"][0, :]
            else:
                data["pi_state"] = _locals["prev_policy_state"][0, :]
                if len(_locals["episode_data"]) == 0:
                    qf1_state, qf2_state = self.initial_state, self.initial_state
                else:
                    qf_feed_dict = {
                        self.qf1_state_ph: _locals["episode_data"][-1]["qf1_state"][None],
                        self.qf2_state_ph: _locals["episode_data"][-1]["qf2_state"][None],
                    }
                    qf_feed_dict.update({getattr(self, data_name + "_ph"): _locals["episode_data"][-1][data_name][None]
                                         for data_name in self.rnn_inputs})
                    qf1_state, qf2_state = self.sess.run([self.qf1_state, self.qf2_state], feed_dict=qf_feed_dict)
                data["qf1_state"] = qf1_state[0, :]
                data["qf2_state"] = qf2_state[0, :]

        if len(_locals["episode_data"]) == 0:
            data["action_prev"] = np.zeros(*self.ac_space.shape, dtype=np.float32)
        else:
            data["action_prev"] = _locals["episode_data"][-1]["action"]

        if self.save_target_state:
            data["target_action_prev_rnn"] = _locals["action"]

        return data


class DRPolicy(RecurrentPolicy):
    """
        Policy object that implements a DDPG-like actor critic, using a feed forward neural network.

        :param sess: (TensorFlow session) The current TensorFlow session
        :param ob_space: (Gym Space) The observation space of the environment
        :param ac_space: (Gym Space) The action space of the environment
        :param n_env: (int) The number of environments to run
        :param n_steps: (int) The number of steps to run for each environment
        :param n_batch: (int) The number of batch to run (n_envs * n_steps)
        :param reuse: (bool) If the policy is reusable or not
        :param layers: ([int]) The size of the Neural network for the policy (if None, default to [64, 64])
        :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
        :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
        :param layer_norm: (bool) enable layer normalisation
        :param act_fun: (tf.func) the activation function to use in the neural network.
        :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
        """
    recurrent = True

    def __init__(self, sess, ob_space, ac_space, goal_size, my_size, n_env=1, n_steps=1, n_batch=None, reuse=False, layers=None,
                 cnn_extractor=nature_cnn, feature_extraction="mlp", n_lstm=128, share_lstm=False,
                 layer_norm=False, act_fun=tf.nn.relu, obs_module_indices=None, **kwargs):
        if layers is None:
            layers = {"ff": [128], "lstm": [128], "head": [128, 128]}
        super().__init__(sess, ob_space, ac_space, layers, n_env, n_steps, n_batch,
                                                reuse=reuse, cnn_extractor=cnn_extractor,
                                                feature_extraction=feature_extraction, n_lstm=n_lstm,
                                                share_lstm=share_lstm, layer_norm=layer_norm, act_fun=act_fun,
                                                obs_module_indices=obs_module_indices, **kwargs)

        with tf.variable_scope("input", reuse=False):
            self.my_ph = tf.placeholder(tf.float32, (None, my_size), name="my_ph")  # the dynamics of the environment

        self.goal_size = goal_size
        self.extra_phs = sorted(self.extra_phs + ["my"])
        self.extra_data_names = sorted(self.extra_data_names + ["my"])

    def make_actor(self, obs_ff=None, obs_rnn=None, action_prev=None, dones=None, reuse=False, scope="pi"):
        if obs_ff is None:
            obs_ff = self.processed_obs
        if obs_rnn is None:
            obs_rnn = self.processed_obs
        if action_prev is None:
            action_prev = self.action_prev_ph

        obs_ff, goal = obs_ff[:, :-self.goal_size], obs_ff[:, -self.goal_size:]
        goal = tf.subtract(goal, obs_ff[:, -self.goal_size:], name="goal_relative")
        obs_rnn = obs_rnn[:, :-self.goal_size]

        ff_phs = [obs_ff, goal]
        rnn_phs = [obs_rnn, action_prev]
        return super().make_actor(ff_phs=ff_phs, rnn_phs=rnn_phs, dones=dones, reuse=reuse, scope=scope)

    def make_critics(self, obs_ff=None, action_ff=None, my=None, obs_rnn=None, action_prev=None, dones=None, reuse=False, scope="values_fn"):
        if obs_ff is None:
            obs_ff = self.processed_obs
        if action_ff is None:
            action_ff = self.action_ph
        if my is None:
            my = self.my_ph
        if obs_rnn is None:
            obs_rnn = self.processed_obs
        if action_prev is None:
            action_prev = self.action_prev_ph

        obs_ff, goal = obs_ff[:, :-self.goal_size], obs_ff[:, -self.goal_size:]
        goal = tf.subtract(goal, obs_ff[:, -self.goal_size:], name="goal_relative")
        obs_rnn = obs_rnn[:, :-self.goal_size]

        ff_phs = [obs_ff, goal, my, action_ff]
        rnn_phs = [obs_rnn, action_prev]
        return super().make_critics(ff_phs=ff_phs, rnn_phs=rnn_phs, dones=dones, reuse=reuse, scope=scope)

    def collect_data(self, _locals, _globals, **kwargs):  # TODO: update for multiprocessing
        data = []
        data = super().collect_data(_locals, _globals)
        for env_i in range(_locals["self"].n_envs):
            d = {}
            if len(_locals["episode_data"][env_i]) == 0 or "my" not in _locals["episode_data"][env_i]:
                if _locals["self"].n_envs == 1:
                    d["my"] = _locals["self"].env.get_env_parameters()
                else:
                    d["my"] = _locals["self"].env.env_method("get_env_parameters", indices=env_i)[0]
            else:
                d["my"] = _locals["episode_data"][env_i][-1]["my"]

            data.append(d)

        return data


class LstmMlpPolicy(RecurrentPolicy):
    recurrent = True

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False,
                 layers=None,
                 cnn_extractor=nature_cnn, feature_extraction="mlp", n_lstm=128, share_lstm=False,
                 layer_norm=False, act_fun=tf.nn.relu, obs_module_indices=None, **kwargs):
        if layers is None:
            layers = {"ff": None, "lstm": [64, 64], "head": []}
        else:
            assert layers["ff"] is None
        super().__init__(sess, ob_space, ac_space, layers, n_env, n_steps, n_batch,
                         reuse=reuse, cnn_extractor=cnn_extractor,
                         feature_extraction=feature_extraction, n_lstm=n_lstm,
                         share_lstm=share_lstm, layer_norm=layer_norm, act_fun=act_fun,
                         obs_module_indices=obs_module_indices, **kwargs)

    def make_actor(self, obs=None, action_prev=None, dones=None, reuse=False, scope="pi"):
        obs, action_prev, dones = self._process_phs(obs=obs, action_prev=action_prev, dones=dones)

        ff_phs = None
        rnn_phs = [obs, action_prev]
        return super().make_actor(ff_phs=ff_phs, rnn_phs=rnn_phs, dones=dones, reuse=reuse, scope=scope)

    def make_critics(self, obs=None, action=None, action_prev=None, dones=None, reuse=False, scope="values_fn"):
        obs, action, action_prev, dones = self._process_phs(obs=obs, action=action, action_prev=action_prev, dones=dones)

        ff_phs = [action]
        rnn_phs = [obs, action_prev]
        return super().make_critics(ff_phs=ff_phs, rnn_phs=rnn_phs, dones=dones, reuse=reuse, scope=scope)


class LstmFFMlpPolicy(RecurrentPolicy):
    recurrent = True

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False,
                 layers=None,
                 cnn_extractor=nature_cnn, feature_extraction="mlp", n_lstm=128, share_lstm=False,
                 layer_norm=False, act_fun=tf.nn.relu, obs_module_indices=None, **kwargs):
        if layers is None:
            layers = {"ff": [64], "lstm": [64, 64], "head": []}

        super().__init__(sess, ob_space, ac_space, layers, n_env, n_steps, n_batch,
                         reuse=reuse, cnn_extractor=cnn_extractor,
                         feature_extraction=feature_extraction, n_lstm=n_lstm,
                         share_lstm=share_lstm, layer_norm=layer_norm, act_fun=act_fun,
                         obs_module_indices=obs_module_indices, **kwargs)

    def make_actor(self, obs=None, action_prev=None, dones=None, reuse=False, scope="pi"):
        obs, action_prev, dones = self._process_phs(obs=obs, action_prev=action_prev, dones=dones)

        ff_phs = [obs]
        rnn_phs = [obs, action_prev]
        return super().make_actor(ff_phs=ff_phs, rnn_phs=rnn_phs, dones=dones, reuse=reuse, scope=scope)

    def make_critics(self, obs=None, action=None, action_prev=None, dones=None, reuse=False, scope="values_fn"):
        obs, action, action_prev, dones = self._process_phs(obs=obs, action=action, action_prev=action_prev, dones=dones)

        ff_phs = [obs, action]
        rnn_phs = [obs, action_prev]
        return super().make_critics(ff_phs=ff_phs, rnn_phs=rnn_phs, dones=dones, reuse=reuse, scope=scope)


class CnnPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a CNN (the nature CNN)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, **_kwargs):
        super(CnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="cnn", **_kwargs)


class CnnMlpPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a CNN (the nature CNN)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, **_kwargs):
        super(CnnMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                           cnn_extractor=cnn_1d_extractor, feature_extraction="cnn", **_kwargs)


class DRCnnMlpPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a CNN (the nature CNN)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, my_size, n_env=1, n_steps=1, n_batch=None, reuse=False, **_kwargs):
        super(DRCnnMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                           cnn_extractor=cnn_1d_extractor, feature_extraction="cnn", **_kwargs)

        with tf.variable_scope("input", reuse=False):
            self.my_ph = tf.placeholder(tf.float32, (self.n_batch, *my_size), name="my_ph")  # (done t-1)
        self.extra_phs = ["my", "target_my"]
        self.extra_data_names = ["my", "target_my"]

    def make_critics(self, obs=None, action=None, my=None, reuse=False, scope="values_fn"):
        if my is None:
            my = self.my_ph

        return super().make_critics(obs, action, reuse, scope, extracted_callback=lambda x: tf.concat([x, my], axis=-1))

    def collect_data(self, _locals, _globals):
        data = []
        for env_i in range(_locals["self"].n_envs):
            d = {}
            if len(_locals["episode_data"][env_i]) == 0 or "my" not in _locals["episode_data"][env_i]:
                if _locals["self"].n_envs == 1:
                    d["my"] = _locals["self"].env.get_env_parameters()
                else:
                    d["my"] = _locals["self"].env.env_method("get_env_parameters", indices=env_i)[0]
            else:
                d["my"] = _locals["episode_data"][env_i][-1]["my"]

            d["target_my"] = d["my"]
            data.append(d)

        return data


class DRMyEstPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a CNN (the nature CNN)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, my_size, n_env=1, n_steps=1, n_batch=None, reuse=False, loss_weight=1e-3, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                           cnn_extractor=cnn_1d_extractor, feature_extraction="mlp", **_kwargs)

        self._obs_ph = self.processed_obs  # Base class has self.obs_ph as property getting self._obs_ph
        with tf.variable_scope("input", reuse=False):
            self.my_ph = tf.placeholder(tf.float32, (self.n_batch, *my_size), name="my_ph")  # (done t-1)
            self.action_prev_ph = tf.placeholder(tf.float32, (self.n_batch, *self.ac_space.shape), name="action_prev_ph")
            self.obs_prev_ph = tf.placeholder(tf.float32, (self.n_batch, *self.ob_space.shape), name="obs_prev_ph")

        self.loss_weight = loss_weight
        self.obs_prev = np.zeros((1, *self.ob_space.shape))
        self.action_prev = np.zeros((1, *self.ac_space.shape))
        self.my_est_loss_op = None
        self.my_est_op = None
        self.policy_loss = None
        self.my_est = None
        self.extra_phs = ["my", "action_prev", "obs_prev", "target_my", "target_action_prev", "target_obs_prev"]
        self.extra_data_names = ["my", "action_prev", "obs_prev", "target_my", "target_action_prev", "target_obs_prev"]

    def _process_phs(self, **phs):
        for ph_name, ph_val in phs.items():
            if ph_val is None:
                phs[ph_name] = getattr(self, ph_name + "_ph")
            else:
                try:
                    setattr(self, ph_name + "_ph", ph_val)
                except AttributeError:
                    setattr(self, "_" + ph_name + "_ph", ph_val)

        return phs.values()

    def make_actor(self, obs=None, obs_prev=None, action_prev=None, my_gt=None, reuse=False, scope="pi"):
        obs, obs_prev, action_prev, my_gt = self._process_phs(obs=obs, obs_prev=obs_prev, action_prev=action_prev, my=my_gt)

        if self.obs_module_indices is not None:
            obs = tf.gather(obs, self.obs_module_indices["pi"], axis=-1)
            obs_prev = tf.gather(obs_prev, self.obs_module_indices["pi"], axis=-1)

        with tf.variable_scope(scope + "/my", reuse=reuse):
            my_h = tf.concat([obs, obs_prev, action_prev], axis=-1)
            my_h = mlp(my_h, [64, 64], self.activ_fn, layer_norm=self.layer_norm)
            self.my_est_op = tf.layers.dense(my_h, self.my_ph.shape[-1])
            self.my_est_loss_op = tf.reduce_mean((self.my_est_op - my_gt) ** 2)
            self.policy_loss = self.loss_weight * self.my_est_loss_op

        obs = tf.concat([obs, self.my_est_op], axis=-1)

        with tf.variable_scope(scope, reuse=reuse):
            if self.feature_extraction == "cnn":
                pi_h = self.cnn_extractor(obs, name="pi_c1", act_fun=self.activ_fn, **self.cnn_kwargs)
            else:
                pi_h = tf.layers.flatten(obs)

            pi_h = mlp(pi_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)

            self.policy_pre_activation = tf.layers.dense(pi_h, self.ac_space.shape[0])
            self.policy = policy = tf.tanh(self.policy_pre_activation)

        return policy

    def make_critics(self, obs=None, action=None, my=None, reuse=False, scope="values_fn"):
        obs, action, my = self._process_phs(obs=obs, action=action, my=my)

        return super().make_critics(obs, action, reuse, scope, extracted_callback=lambda x: tf.concat([x, my], axis=-1))

    def collect_data(self, _locals, _globals):
        data = {}
        if "my" not in _locals or _locals["episode_data"]:
            data["my"] = _locals["self"].env.get_env_parameters()
            data["target_my"] = data["my"]
        if len(_locals["episode_data"]) == 0:
            data["obs_prev"] = _locals["obs"]
            data["action_prev"] = _locals["action"]
        else:
            data["obs_prev"] = _locals["episode_data"][-1]["obs"]
            data["action_prev"] = _locals["episode_data"][-1]["action"]
        data["target_obs_prev"] = data["obs_prev"]
        data["target_action_prev"] = data["action_prev"]

        return data
    
    def step(self, obs, obs_prev=None, action_prev=None, mask=None):
        if action_prev is None:
            assert obs.shape[0] == 1
            if mask is not None and mask[0]:
                self.action_prev = np.zeros((1, *self.ac_space.shape))
            action_prev = self.action_prev
        if obs_prev is None:
            if mask is not None and mask[0]:
                self.obs_prev = np.zeros((1, *self.ob_space.shape))
            obs_prev = self.obs_prev

        action, my_est = self.sess.run([self.policy, self.my_est_op], {self.obs_ph: obs,
                                             self.action_prev_ph: action_prev,
                                             self.obs_prev_ph: obs_prev})
        self.action_prev = action
        self.obs_prev = obs
        self.my_est = my_est

        #return action, my_est
        return action


class LnCnnPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a CNN (the nature CNN), with layer normalisation

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, **_kwargs):
        super(LnCnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                          feature_extraction="cnn", layer_norm=True, **_kwargs)


class MlpPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a MLP (2 layers of 64)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, **_kwargs):
        super(MlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="mlp", **_kwargs)


class LnMlpPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a MLP (2 layers of 64), with layer normalisation

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, **_kwargs):
        super(LnMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                          feature_extraction="mlp", layer_norm=True, **_kwargs)


register_policy("LstmFFMlpPolicy", LstmFFMlpPolicy)
register_policy("LstmMlpPolicy", LstmMlpPolicy)
register_policy("DRPolicy", DRPolicy)
register_policy("CnnPolicy", CnnPolicy)
register_policy("LnCnnPolicy", LnCnnPolicy)
register_policy("MlpPolicy", MlpPolicy)
register_policy("LnMlpPolicy", LnMlpPolicy)
register_policy("CnnMlpPolicy", CnnMlpPolicy)
register_policy("DRCnnMlpPolicy", DRCnnMlpPolicy)
register_policy("DRMyEstPolicy", DRMyEstPolicy)
