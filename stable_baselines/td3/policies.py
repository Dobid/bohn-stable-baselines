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

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, scale=False):
        super(TD3Policy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=scale)
        assert isinstance(ac_space, Box), "Error: the action space must be of type gym.spaces.Box"
        assert (np.abs(ac_space.low) == ac_space.high).all(), "Error: the action space low and high must be symmetric"

        self.qf1 = None
        self.qf2 = None
        self.q_discrepancy = None
        self.policy = None

    def make_actor(self, obs=None, reuse=False, scope="pi"):
        """
        Creates an actor object

        :param obs: (TensorFlow Tensor) The observation placeholder (can be None for default placeholder)
        :param reuse: (bool) whether or not to resue parameters
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
        :param reuse: (bool) whether or not to resue parameters
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

    def make_critics(self, obs=None, action=None, reuse=False, scope="values_fn"):
        if obs is None:
            obs = self.processed_obs

        if self.obs_module_indices is not None:
            obs = tf.gather(obs, self.obs_module_indices["vf"], axis=-1)

        with tf.variable_scope(scope, reuse=reuse):
            if self.feature_extraction == "cnn" and self.cnn_vf:
                critics_h = self.cnn_extractor(obs, name="vf_c1", act_fun=self.activ_fn, **self.cnn_kwargs)
            else:
                critics_h = tf.layers.flatten(obs)

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
            self.q_discrepancy = tf.abs(self.qf1 - self.qf2)

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
                 layer_norm=False, act_fun=tf.nn.relu, obs_module_indices=None, **kwargs):
        super(RecurrentPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                                                reuse=reuse,
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

        self._initial_state = np.zeros((1, self.n_lstm * 2), dtype=np.float32)
        if self.share_lstm:
            self.state = None
        else:
            self.pi_state = None
            self.qf1_state = None
            self.qf2_state = None

        with tf.variable_scope("input", reuse=False):
            self.mask_ph = tf.placeholder(tf.float32, (None, None,), name="dones_ph")  # (batch, sequence_length) ignore computation when 0
            if self.share_lstm:
                self.state_ph = tf.placeholder(tf.float32, (None, self.n_lstm * 2), name="state_ph")
            else:
                self.pi_state_ph = tf.placeholder(tf.float32, (None, self.n_lstm * 2), name="pi_state_ph")
                self.qf1_state_ph = tf.placeholder(tf.float32, (None, self.n_lstm * 2), name="qf1_state_ph")
                self.qf2_state_ph = tf.placeholder(tf.float32, (None, self.n_lstm * 2), name="qf2_state_ph")

        self.extra_phs = []
        self.rnn_inputs = []
        self.extra_data_names = []

        self.save_state = save_state
        if self.save_state:
            if self.share_lstm:
                self.extra_data_names = ["state"]
                self.extra_phs = ["state"]
            else:
                self.extra_data_names = ["pi_state", "qf1_state", "qf2_state"]
                self.extra_phs = ["pi_state", "qf1_state", "qf2_state"]

    def _make_branch(self, branch_name, input_tensor, mask=None):
        if branch_name == "lstm":
            # reshape (b, s, F) or (b/s, F) into (b * s, F)
            if len(input_tensor.shape) == 2:
                n_batch = input_tensor.shape[0]
                n_seq = 1
            else:
                n_batch = input_tensor.shape[0]
                n_seq = input_tensor.shape[1]
            input_tensor = tf.reshape(input_tensor, (n_batch * n_seq, -1))
            for i, fc_layer_units in enumerate(self.layers["lstm"]):
                input_tensor = self.activ_fn(tf.layers.dense(input_tensor, fc_layer_units, name="lstm_fc{}".format(i)))

            # reshape (b * s, input_tensor_nodes) into (b, s, input_tensor_nodes)
            input_tensor = tf.reshape(input_tensor, (n_batch, n_seq, -1))
            input_tensor, state = tf.keras.layers.LSTM(input_tensor, mask, self.state_ph, 'lstm', n_hidden=self.n_lstm)
            input_tensor = tf.reshape(input_tensor, (n_batch, n_seq, -1))
            return input_tensor, state
            # reshape (b, s, n_lstm_hidden) into (b*s, n_lstm_hidden)
        else:
            for i, fc_layer_units in enumerate(self.layers[branch_name]):
                input_tensor = self.activ_fn(tf.layers.dense(input_tensor, fc_layer_units, name="{}_fc{}".format(branch_name, i)))

            return input_tensor

    def make_actor(self, ff_phs=None, rnn_phs=None, mask=None, reuse=False, scope="pi"):
        lstm_branch = tf.concat([tf.layers.flatten(ph) for ph in rnn_phs], axis=-1)
        if ff_phs is not None:
            ff_branch = tf.concat([tf.layers.flatten(ph) for ph in ff_phs], axis=-1)

        if mask is None:
            mask = self.mask_ph

        if self.share_lstm:
            with tf.variable_scope("shared", reuse=tf.AUTO_REUSE):
                lstm_branch, self.state = self._make_branch("lstm", lstm_branch, mask)

        with tf.variable_scope(scope, reuse=reuse):
            if self.layers["ff"] is not None:
               ff_branch = self._make_branch("ff", ff_branch)

            if not self.share_lstm:
                lstm_branch, self.pi_state = self._make_branch("lstm", lstm_branch, mask)

            if ff_phs is not None:
                head = tf.concat([ff_branch, lstm_branch], axis=-1)
            else:
                head = lstm_branch

            head = self._make_branch("head", head)

            self.policy_pre_activation = tf.layers.dense(head, self.ac_space.shape[0])
            self.policy = policy = tf.tanh(self.policy_pre_activation)

        return policy

    def make_critics(self, ff_phs=None, rnn_phs=None, mask=None, reuse=False, scope="values_fn"):
        lstm_branch_in = tf.concat([tf.layers.flatten(ph) for ph in rnn_phs], axis=-1)
        if ff_phs is not None:
            ff_branch_in = tf.concat([tf.layers.flatten(ph) for ph in ff_phs], axis=-1)

        if mask is None:
            mask = self.mask_ph

        self.qf1, self.qf2 = None, None
        self.qf1_state, self.qf2_state = None, None

        if self.share_lstm:
            with tf.variable_scope("shared", reuse=tf.AUTO_REUSE):
                lstm_branch_s, self.state = self._make_branch("lstm", lstm_branch_in, mask)

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
                        lstm_branch, state = self._make_branch("lstm", lstm_branch, mask)
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

    def step(self, obs, state=None, mask=None):
        raise NotImplementedError

    @property
    def initial_state(self):
        return self._initial_state

    def collect_data(self, _locals, _globals):
        raise NotImplementedError


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
        super().__init__(sess, ob_space, ac_space, layers, n_env, n_steps, n_batch * n_steps,
                                                reuse=reuse, cnn_extractor=cnn_extractor,
                                                feature_extraction=feature_extraction, n_lstm=n_lstm,
                                                share_lstm=share_lstm, layer_norm=layer_norm, act_fun=act_fun,
                                                obs_module_indices=obs_module_indices, **kwargs)

        self.n_batch = n_batch  # TODO: fix this in a less hacky way?

        with tf.variable_scope("input", reuse=False):
            self.action_prev_rnn_ph = tf.placeholder(tf.float32, (None,) + ac_space.shape, name="action_prev_ph")
            self.my_ph = tf.placeholder(tf.float32, (None, my_size), name="my_ph")  # the dynamics of the environment

        self.action_prev = np.zeros((1, *self.ac_space.shape))
        self.goal_size = goal_size
        self.extra_phs = sorted(self.extra_phs + ["action_prev_rnn", "my", "target_action_prev_rnn"])
        self.rnn_inputs = sorted(self.rnn_inputs + ["obs", "action_prev_rnn", "target_action_prev_rnn", "obs_tp1"])
        self.extra_data_names = sorted(self.extra_data_names + ["action_prev_rnn", "my", "target_action_prev_rnn"])

    def make_actor(self, obs_ff=None, obs_rnn=None, action_prev=None, dones=None, reuse=False, scope="pi"):
        if obs_ff is None:
            obs_ff = self.processed_obs
        if obs_rnn is None:
            obs_rnn = self.processed_obs
        if action_prev is None:
            action_prev = self.action_prev_rnn_ph

        if self.n_steps > 1:
            obs_ff = obs_ff[::self.n_steps, :]
        obs_ff, goal = obs_ff[:, :-self.goal_size], obs_ff[:, -self.goal_size:]
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
            action_prev = self.action_prev_rnn_ph

        if self.n_steps > 1:
            obs_ff = obs_ff[::self.n_steps, :]
        obs_ff, goal = obs_ff[:, :-self.goal_size], obs_ff[:, -self.goal_size:]
        obs_rnn = obs_rnn[:, :-self.goal_size]

        ff_phs = [obs_ff, goal, my, action_ff]
        rnn_phs = [obs_rnn, action_prev]
        return super().make_critics(ff_phs=ff_phs, rnn_phs=rnn_phs, dones=dones, reuse=reuse, scope=scope)

    def step(self, obs, state=None, action_prev=None, mask=None):
        if state is None:
            state = self.initial_state
        if action_prev is None:
            assert obs.shape[0] == 1
            if mask[0]:
                self.action_prev = np.zeros((1, *self.ac_space.shape))
            action_prev = self.action_prev

        lstm_node = self.state if self.share_lstm else self.pi_state
        state_ph = self.state_ph if self.share_lstm else self.pi_state_ph

        action, out_state = self.sess.run([self.policy, lstm_node],
                            {self.obs_ph: obs, state_ph: state, self.dones_ph: mask,
                            self.action_prev_rnn_ph: action_prev})
        self.action_prev = action

        return action, out_state

    def collect_data(self, _locals, _globals):
        data = {}

        if self.save_state:
            if self.share_lstm:
                data["state"] = _locals["policy_state"][0, :]
            else:
                data["pi_state"] = _locals["policy_state"][0, :]
                if len(_locals["episode_data"]) == 0:
                    qf1_state_prev, qf2_state_prev = self.initial_state, self.initial_state
                    action_prev = np.zeros(self.ac_space.shape)
                else:
                    qf1_state_prev = _locals["episode_data"][-1].get("qf1_state", self.initial_state)[None]
                    qf2_state_prev = _locals["episode_data"][-1].get("qf2_state", self.initial_state)[None]
                    action_prev = _locals["episode_data"][-1]["action"]

                qf1_state, qf2_state = self.sess.run([self.qf1_state, self.qf2_state], feed_dict={
                    self.processed_obs: _locals["obs"][None],
                    self.action_prev_rnn_ph: action_prev[None],
                    self.qf1_state_ph: qf1_state_prev,
                    self.qf2_state_ph: qf2_state_prev,
                    self.dones_ph: np.array(_locals["done"])[None]
                })
                data["qf1_state"] = qf1_state[0, :]
                data["qf2_state"] = qf2_state[0, :]

        if len(_locals["episode_data"]) == 0:
            data["action_prev_rnn"] = np.zeros(self.ac_space.shape)
        else:
            data["action_prev_rnn"] = _locals["episode_data"][-1]["action"]
        if "my" not in _locals or _locals["ep_data"]:
            data["my"] = _locals["self"].env.get_env_parameters()
        data["target_action_prev_rnn"] = _locals["action"]

        return data


class LstmMlpPolicy(RecurrentPolicy):
    recurrent = True

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False,
                 layers=None,
                 cnn_extractor=nature_cnn, feature_extraction="mlp", n_lstm=128, share_lstm=False,
                 layer_norm=False, act_fun=tf.nn.relu, obs_module_indices=None, **kwargs):
        if layers is None:
            layers = {"ff": None, "lstm": [], "head": [256, 128]}
        else:
            assert layers["ff"] is None
        super().__init__(sess, ob_space, ac_space, layers, n_env, n_steps, n_batch,
                         reuse=reuse, cnn_extractor=cnn_extractor,
                         feature_extraction=feature_extraction, n_lstm=n_lstm,
                         share_lstm=share_lstm, layer_norm=layer_norm, act_fun=act_fun,
                         obs_module_indices=obs_module_indices, **kwargs)

    def make_actor(self, obs=None, dones=None, reuse=False, scope="pi"):
        if obs is None:
            obs = self.processed_obs

        ff_phs = None
        rnn_phs = [obs]
        return super().make_actor(ff_phs=ff_phs, rnn_phs=rnn_phs, dones=dones, reuse=reuse, scope=scope)

    def make_critics(self, obs=None, action=None, dones=None, reuse=False, scope="values_fn"):
        if obs is None:
            obs = self.processed_obs
        if action is None:
            action = self.action_ph

        ff_phs = None
        rnn_phs = [obs, action]
        return super().make_critics(ff_phs=ff_phs, rnn_phs=rnn_phs, dones=dones, reuse=reuse, scope=scope)

    def step(self, obs, state=None, mask=None):
        if state is None:
            state = self.initial_state
        if mask is None:
            mask = np.array([False])

        return self.sess.run([self.policy, self.pi_state],
                             {self.obs_ph: obs, self.pi_state_ph: state, self.dones_ph: mask})


class LstmFFMlpPolicy(RecurrentPolicy):
    recurrent = True

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False,
                 layers=None,
                 cnn_extractor=nature_cnn, feature_extraction="mlp", n_lstm=128, share_lstm=False,
                 layer_norm=False, act_fun=tf.nn.relu, obs_module_indices=None, **kwargs):
        if layers is None:
            layers = {"ff": [], "lstm": [], "head": [64, 64]}

        super().__init__(sess, ob_space, ac_space, layers, n_env, n_steps, n_batch,
                         reuse=reuse, cnn_extractor=cnn_extractor,
                         feature_extraction=feature_extraction, n_lstm=n_lstm,
                         share_lstm=share_lstm, layer_norm=layer_norm, act_fun=act_fun,
                         obs_module_indices=obs_module_indices, **kwargs)

    def make_actor(self, obs=None, dones=None, reuse=False, scope="pi"):
        if obs is None:
            obs = self.processed_obs

        ff_phs = [obs]
        rnn_phs = [obs]
        return super().make_actor(ff_phs=ff_phs, rnn_phs=rnn_phs, dones=dones, reuse=reuse, scope=scope)

    def make_critics(self, obs=None, action=None, dones=None, reuse=False, scope="values_fn"):
        if obs is None:
            obs = self.processed_obs
        if action is None:
            action = self.action_ph

        ff_phs = [obs, action]
        rnn_phs = [obs, action]
        return super().make_critics(ff_phs=ff_phs, rnn_phs=rnn_phs, dones=dones, reuse=reuse, scope=scope)

    def step(self, obs, state=None, mask=None):
        if state is None:
            state = self.initial_state
        if mask is None:
            mask = np.array([False])

        return self.sess.run([self.policy, self.pi_state],
                             {self.obs_ph: obs, self.pi_state_ph: state, self.dones_ph: mask})


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
