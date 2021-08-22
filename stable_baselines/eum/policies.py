import tensorflow as tf
import numpy as np
from gym.spaces import Box
import copy

from stable_baselines.common.policies import BasePolicy, nature_cnn, register_policy, cnn_1d_extractor
from stable_baselines.sac.policies import mlp
from stable_baselines.a2c.utils import lstm, batch_to_seq, seq_to_batch


class EUMPolicy(BasePolicy):
    """
    Policy object that implements a EUM-like actor critic

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
        super(EUMPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=scale)
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


class FeedForwardPolicy(EUMPolicy):
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
                 cnn_extractor=nature_cnn, feature_extraction="cnn", n_critics=4,
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
        self.n_critics = n_critics

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
            
            qfs = []

            for qf_i in range(self.n_critics):
                # Double Q values to reduce overestimation
                with tf.variable_scope('qf{}'.format(qf_i), reuse=reuse):
                    qf = mlp(qf_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)
                    qf = tf.layers.dense(qf, 1, name="qf{}".format(qf_i))

                setattr(self, "qf{}".format(qf_i), qf)
                qfs.append(qf)

            # TODO: assumes that all qf1 and qf2 can never have opposite signs
            #self.q_discrepancy = tf.square(self.qf1 - self.qf2) / tf.square(tf.maximum(self.qf1, self.qf2))
            self.q_discrepancy = tf.abs(self.qf1 - self.qf2)

        return qfs

    def step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy, {self.obs_ph: obs})

    def get_q_discrepancy(self, obs):
        if isinstance(obs, np.ndarray) and len(obs.shape) == 1: # TODO: check for MLP or CNN policy here
            obs = np.expand_dims(obs, axis=0)
        return self.sess.run(self.q_discrepancy, {self.obs_ph: obs})


class RecurrentPolicy(EUMPolicy):
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
                 cnn_extractor=nature_cnn, feature_extraction="mlp", n_lstm=128,
                 layer_norm=False, act_fun=tf.nn.relu, obs_module_indices=None, **kwargs):
        #ob_space  # Hack away goal from obs placeholder
        policy_ob_space = copy.deepcopy(ob_space)
        policy_ob_space.low = policy_ob_space.low[:-goal_size]
        policy_ob_space.high = policy_ob_space.high[:-goal_size]
        policy_ob_space.shape = policy_ob_space.high.shape
        super(RecurrentPolicy, self).__init__(sess, policy_ob_space, ac_space, n_env, n_steps, n_batch,
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
            layers = {"ff": [128], "lstm": [128], "head": [128, 128]}
        self.layers = layers
        self.obs_module_indices = obs_module_indices

        assert len(layers) >= 1, "Error: must have at least one hidden layer for the policy."

        self.activ_fn = act_fun
        self.n_lstm = n_lstm

        with tf.variable_scope("input", reuse=False):
            self.dones_ph = tf.placeholder(tf.float32, (None,), name="dones_ph")  # (done t-1)
            self.goal_ph = tf.placeholder(tf.float32, (None, goal_size), name="goal_ph")
            self.action_prev_ph = tf.placeholder(tf.float32, (None,) + ac_space.shape, name="action_prev_ph")
            self.pi_state_ph = tf.placeholder(tf.float32, (1, self.n_lstm * 2), name="pi_state_ph")
            self.qf1_state_ph = tf.placeholder(tf.float32, (1, self.n_lstm * 2), name="qf1_state_ph")
            self.qf2_state_ph = tf.placeholder(tf.float32, (1, self.n_lstm * 2), name="qf2_state_ph")
            self.my_ph = tf.placeholder(tf.float32, (None, my_size), name="my_ph")  # the dynamics of the environment
            obs_rnn_shape = [dim for dim in self.processed_obs.shape]
            obs_rnn_shape[0] *= self.n_steps
            self.obs_rnn_ph = tf.placeholder(tf.float32, shape=obs_rnn_shape, name="obs_rnn_ph")

        self.pi_initial_state = np.zeros((1, self.n_lstm * 2), dtype=np.float32)
        self.qf1_initial_state = np.zeros((1, self.n_lstm * 2), dtype=np.float32)
        self.qf2_initial_state = np.zeros((1, self.n_lstm * 2), dtype=np.float32)

        self.pi_state = None
        self.qf1_state = None
        self.qf2_state = None

    def make_actor(self, obs_ff=None, goal=None, obs_rnn=None, action_rnn=None, dones=None, reuse=False, scope="pi"):
        if obs_ff is None:
            obs_ff = self.processed_obs
        if obs_rnn is None:
            obs_rnn = self.processed_obs
        if goal is None:
            goal = self.goal_ph
        if action_rnn is None:
            action_rnn = self.action_prev_ph
        if dones is None:
            dones = self.dones_ph

        #if self.obs_module_indices is not None:
        #    obs = tf.gather(obs, self.obs_module_indices["pi"], axis=-1)
        with tf.variable_scope(scope, reuse=reuse):
            #if self.feature_extraction == "cnn":
            #    pi_h = self.cnn_extractor(obs, name="pi_c1", act_fun=self.activ_fn, **self.cnn_kwargs)
            #else:
            pi_h_ff = tf.layers.flatten(obs_ff)
            pi_h_lstm = tf.layers.flatten(obs_rnn)

            ff_branch = tf.concat([pi_h_ff, goal], axis=-1)
            lstm_branch = tf.concat([pi_h_lstm, action_rnn], axis=-1)

            for i, fc_layer_units in enumerate(self.layers["ff"]):
                ff_branch = self.activ_fn(tf.layers.dense(ff_branch, fc_layer_units, name="ff_fc{}".format(i)))

            for i, fc_layer_units in enumerate(self.layers["lstm"]):
                lstm_branch = self.activ_fn(tf.layers.dense(lstm_branch, fc_layer_units, name="rnn_fc{}".format(i)))

            lstm_branch = batch_to_seq(lstm_branch, n_batch=self.n_batch, n_steps=self.n_steps)
            masks = batch_to_seq(dones, self.n_batch, self.n_steps)
            lstm_branch, self.pi_state = lstm(lstm_branch, masks, self.pi_state_ph, 'lstm_pi', n_hidden=self.n_lstm,
                                         layer_norm=self.layer_norm)

            if self.n_steps > 1:
                lstm_branch = [lstm_branch[-1]]
            lstm_branch = seq_to_batch(lstm_branch)
            head = tf.concat([ff_branch, lstm_branch], axis=-1)

            for i, fc_layer_units in enumerate(self.layers["head"]):
                head = self.activ_fn(tf.layers.dense(head, fc_layer_units, name="head_fc{}".format(i)))

            self.policy_pre_activation = tf.layers.dense(head, self.ac_space.shape[0])
            self.policy = policy = tf.tanh(self.policy_pre_activation)

        return policy

    def make_critics(self, obs_ff=None, action_ff=None, goal=None, my=None, obs_rnn=None, action_rnn=None, dones=None, reuse=False, scope="values_fn"):
        if obs_ff is None:
            obs_ff = self.processed_obs
        if obs_rnn is None:
            obs_rnn = self.processed_obs
        if goal is None:
            goal = self.goal_ph
        if action_rnn is None:
            action_rnn = self.action_prev_ph
        if my is None:
            my = self.my_ph
        if dones is None:
            dones = self.dones_ph

        with tf.variable_scope(scope, reuse=reuse):
            #if self.feature_extraction == "cnn" and self.cnn_vf:
            #    critics_h = self.cnn_extractor(obs, name="vf_c1", act_fun=self.activ_fn, **self.cnn_kwargs)
            #else:
            critics_h_ff = tf.layers.flatten(obs_ff)
            critics_h_lstm = tf.layers.flatten(obs_rnn)

            # Concatenate preprocessed state and action
            ff_branch_in = tf.concat([critics_h_ff, action_ff, goal, my], axis=-1)
            lstm_branch_in = tf.concat([critics_h_lstm, action_rnn], axis=-1)

            self.qf1, self.qf2 = None, None
            self.qf1_state, self.qf2_state = None, None

            masks = batch_to_seq(dones, self.n_batch, self.n_steps)

            # Double Q values to reduce overestimation
            for qf_i in range(1, 3):
                with tf.variable_scope('qf{}'.format(qf_i), reuse=reuse):
                    ff_branch, lstm_branch = ff_branch_in, lstm_branch_in
                    for i, fc_layer_units in enumerate(self.layers["ff"]):
                        ff_branch = self.activ_fn(tf.layers.dense(ff_branch, fc_layer_units, name="ff_fc{}".format(i)))

                    for i, fc_layer_units in enumerate(self.layers["lstm"]):
                        lstm_branch = self.activ_fn(tf.layers.dense(lstm_branch, fc_layer_units, name="rnn_fc{}".format(i)))

                    lstm_branch = batch_to_seq(lstm_branch, self.n_batch, self.n_steps)
                    lstm_branch, q_state = lstm(lstm_branch, masks, getattr(self, "qf{}_state_ph".format(qf_i)),
                                                "lstm_qf{}".format(qf_i), n_hidden=self.n_lstm, layer_norm=self.layer_norm)
                    setattr(self, "qf{}_state".format(qf_i), q_state)
                    if self.n_steps > 1:
                        lstm_branch = [lstm_branch[-1]]
                    lstm_branch = seq_to_batch(lstm_branch)
                    head = tf.concat([ff_branch, lstm_branch], axis=-1)

                    for i, fc_layer_units in enumerate(self.layers["head"]):
                        head = self.activ_fn(tf.layers.dense(head, fc_layer_units, name="head_fc{}".format(i)))

                    setattr(self, "qf{}".format(qf_i), tf.layers.dense(head, 1, name="qf{}".format(i)))

        return self.qf1, self.qf2

    def step(self, obs, state=None, goal=None, action_prev=None, mask=None, obs_rnn=None):
        if state is None:
            state = self.initial_state
        if obs_rnn is None:
            obs_rnn = obs
        return self.sess.run([self.policy, self.pi_state],
                            {self.obs_ph: obs, self.goal_ph: goal, self.pi_state_ph: state, self.dones_ph: mask,
                         self.action_prev_ph: action_prev, self.obs_rnn_ph: obs_rnn})

    @property
    def initial_state(self):
        return self.pi_initial_state


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

class DRPolicy(RecurrentPolicy):
    """
    Policy object that implements actor critic, using a recurrent Policy

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
        super(DRPolicy, self).__init__(sess, ob_space, ac_space, n_env=n_env, n_steps=n_steps, n_batch=n_batch,
                                       reuse=reuse, feature_extraction="mlp", layer_norm=False, **_kwargs)


register_policy("DRPolicy", DRPolicy)
register_policy("CnnPolicy", CnnPolicy)
register_policy("LnCnnPolicy", LnCnnPolicy)
register_policy("MlpPolicy", MlpPolicy)
register_policy("LnMlpPolicy", LnMlpPolicy)
register_policy("CnnMlpPolicy", CnnMlpPolicy)
