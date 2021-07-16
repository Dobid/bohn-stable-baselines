import warnings
from itertools import zip_longest
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from gym.spaces import Discrete

from stable_baselines.common.tf_util import batch_to_seq, seq_to_batch
from stable_baselines.common.tf_layers import conv, linear, conv_to_fc, lstm
from stable_baselines.common.distributions import make_proba_dist_type, CategoricalProbabilityDistribution, \
    MultiCategoricalProbabilityDistribution, DiagGaussianProbabilityDistribution, BernoulliProbabilityDistribution, \
    BetaProbabilityDistribution, MixProbabilityDistribution, GeneralizedPoissonProbabilityDistribution, GeneralizedPoissonProbabilityDistributionType, BoundedDiagGaussianProbabilityDistributionType, RLMPCProbabilityDistributionType
try:
    from stable_baselines.common.distributions import NegativeBinomialProbabilityDistribution, NegativeBinomialProbabilityDistributionType, PoissonProbabilityDistribution, PoissonProbabilityDistributionType
    tfp_import = True
except ImportError:
    tfp_import = False

from stable_baselines.common.distributions import MixProbabilityDistributionType
from stable_baselines.common.input import observation_input
import time


def nature_cnn(scaled_images, act_fun=tf.nn.relu, **kwargs):
    """
    CNN from Nature paper.

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = act_fun
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_3)
    return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))


def cnn_1d_extractor(obs, act_fun, name, **kwargs):
    kwargs["n_filters"] = kwargs.get("n_filters", 3)
    obs = tf.expand_dims(obs, -1)
    return conv_to_fc(act_fun(conv(obs, name, filter_size=(obs.shape[1], 1), stride=1, init_scale=np.sqrt(2), **kwargs)))


def cnn_mlp_extractor(obs, net_arch, act_fun, obs_module_indices=None, dual_critic=False, **kwargs):
    conv_vf = kwargs.pop("cnn_vf", True)
    shared = kwargs.pop("cnn_shared", None)
    if shared or shared is None:
        name = "shared_c1" if shared is not None else "c1"  # Backwards Compatability
        conv_layer = cnn_1d_extractor(obs, act_fun, name, **kwargs)
        mlps = mlp_extractor(conv_layer, net_arch, act_fun, obs_module_indices, dual_critic=dual_critic)
        if dual_critic:
            pi_mlp, vf_mlp, vf_mlp_d = mlps
        else:
            pi_mlp, vf_mlp = mlps
    else:
        if obs_module_indices is not None:
            pi_obs = tf.gather(obs, indices=obs_module_indices["pi"], axis=-1)
            vf_obs = tf.gather(obs, indices=obs_module_indices["vf"], axis=-1)
        else:
            pi_obs = obs
            vf_obs = obs
        if not conv_vf:
            if len(obs.shape) == 3:
                vf_obs = vf_obs[:, 0, :]
            vf_mlps = mlp_extractor(vf_obs, [dict(vf=net_arch[-1]["vf"])], act_fun, dual_critic=dual_critic)
            if dual_critic:
                _, vf_mlp, vf_mlp_d = vf_mlps
            else:
                _, vf_mlp = vf_mlps
            pi_conv = cnn_1d_extractor(pi_obs, act_fun, "pi_c1", **kwargs)
            # TODO: does support for other shared layers here make sense.
            pi_mlp = mlp_extractor(pi_conv, [dict(pi=net_arch[-1]["pi"])], act_fun, dual_critic=False)[0]
        else:
            pi_conv = cnn_1d_extractor(pi_obs, act_fun, "pi_c1", **kwargs)
            vf_conv = cnn_1d_extractor(vf_obs, act_fun, "vf_c1", **kwargs)
            # TODO: does support for other shared layers here make sense.
            pi_mlp = mlp_extractor(pi_conv, [dict(pi=net_arch[-1]["pi"])], act_fun, dual_critic=False)[0]
            vf_mlps = mlp_extractor(vf_conv, [dict(vf=net_arch[-1]["vf"])], act_fun, dual_critic=dual_critic)
            if dual_critic:
                _, vf_mlp, vf_mlp_d = vf_mlps
            else:
                _, vf_mlp = vf_mlps

    if dual_critic:
        return pi_mlp, vf_mlp, vf_mlp_d
    return pi_mlp, vf_mlp


def mlp_extractor(flat_observations, net_arch, act_fun, obs_module_indices=None, dual_critic=False):
    """
    Constructs an MLP that receives observations as an input and outputs a latent representation for the policy and
    a value network. The ``net_arch`` parameter allows to specify the amount and size of the hidden layers and how many
    of them are shared between the policy network and the value network. It is assumed to be a list with the following
    structure:

    1. An arbitrary length (zero allowed) number of integers each specifying the number of units in a shared layer.
       If the number of ints is zero, there will be no shared layers.
    2. An optional dict, to specify the following non-shared layers for the value network and the policy network.
       It is formatted like ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``.
       If it is missing any of the keys (pi or vf), no non-shared layers (empty list) is assumed.

    For example to construct a network with one shared layer of size 55 followed by two non-shared layers for the value
    network of size 255 and a single non-shared layer of size 128 for the policy network, the following layers_spec
    would be used: ``[55, dict(vf=[255, 255], pi=[128])]``. A simple shared network topology with two layers of size 128
    would be specified as [128, 128].

    :param flat_observations: (tf.Tensor) The observations to base policy and value function on.
    :param net_arch: ([int or dict]) The specification of the policy and value networks.
        See above for details on its formatting.
    :param act_fun: (tf function) The activation function to use for the networks.
    :return: (tf.Tensor, tf.Tensor) latent_policy, latent_value of the specified network.
        If all layers are shared, then ``latent_policy == latent_value``
    """
    latent = flat_observations
    policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
    value_only_layers = []  # Layer sizes of the network that only belongs to the value network

    # Iterate through the shared layers and build the shared parts of the network
    for idx, layer in enumerate(net_arch):
        if isinstance(layer, int):  # Check that this is a shared layer
            layer_size = layer
            latent = act_fun(linear(latent, "shared_fc{}".format(idx), layer_size, init_scale=np.sqrt(2)))
        else:
            assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
            if 'pi' in layer:
                assert isinstance(layer['pi'], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                policy_only_layers = layer['pi']

            if 'vf' in layer:
                assert isinstance(layer['vf'], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                value_only_layers = layer['vf']
            break  # From here on the network splits up in policy and value network

    # Build the non-shared part of the network
    if obs_module_indices is not None:
        latent_policy = tf.gather(latent, obs_module_indices["pi"], axis=1)
        latent_value = tf.gather(latent, obs_module_indices["vf"], axis=1)
    else:
        latent_policy = latent
        latent_value = latent

    if dual_critic:
        latent_value_d = latent

    for idx, (pi_layer_size, vf_layer_size) in enumerate(zip_longest(policy_only_layers, value_only_layers)):
        if pi_layer_size is not None:
            assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
            latent_policy = act_fun(linear(latent_policy, "pi_fc{}".format(idx), pi_layer_size, init_scale=np.sqrt(2)), name="pi_fc{}_tanh".format(idx))

        if vf_layer_size is not None:
            assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
            latent_value = act_fun(linear(latent_value, "vf_fc{}".format(idx), vf_layer_size, init_scale=np.sqrt(2)), name="vf_fc{}_tanh".format(idx))
            if dual_critic:
                latent_value_d = act_fun(linear(latent_value_d, "vf_d_fc{}".format(idx), vf_layer_size, init_scale=np.sqrt(2)), name="vf_d_fc{}_tanh".format(idx))

    if dual_critic:
        return latent_policy, latent_value, latent_value_d
    return latent_policy, latent_value


class BasePolicy(ABC):
    """
    The base policy object

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batches to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param scale: (bool) whether or not to scale the input
    :param obs_phs: (TensorFlow Tensor, TensorFlow Tensor) a tuple containing an override for observation placeholder
        and the processed observation placeholder respectively
    :param add_action_ph: (bool) whether or not to create an action placeholder
    """

    recurrent = False

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, scale=False,
                 obs_phs=None, add_action_ph=False):
        self.n_env = n_env
        self.n_steps = n_steps
        self.n_batch = n_batch
        with tf.variable_scope("input", reuse=False):
            if obs_phs is None:
                self._obs_ph, self._processed_obs = observation_input(ob_space, n_batch, scale=scale)
            else:
                self._obs_ph, self._processed_obs = obs_phs

            self._action_ph = None
            if add_action_ph:
                self._action_ph = tf.placeholder(dtype=ac_space.dtype, shape=(n_batch,) + ac_space.shape,
                                                 name="action_ph")
        self.sess = sess
        self.reuse = reuse
        self.ob_space = ob_space
        self.ac_space = ac_space

    @property
    def is_discrete(self):
        """bool: is action space discrete."""
        return isinstance(self.ac_space, Discrete)

    @property
    def initial_state(self):
        """
        The initial state of the policy. For feedforward policies, None. For a recurrent policy,
        a NumPy array of shape (self.n_env, ) + state_shape.
        """
        assert not self.recurrent, "When using recurrent policies, you must overwrite `initial_state()` method"
        return None

    @property
    def obs_ph(self):
        """tf.Tensor: placeholder for observations, shape (self.n_batch, ) + self.ob_space.shape."""
        return self._obs_ph

    @property
    def processed_obs(self):
        """tf.Tensor: processed observations, shape (self.n_batch, ) + self.ob_space.shape.

        The form of processing depends on the type of the observation space, and the parameters
        whether scale is passed to the constructor; see observation_input for more information."""
        return self._processed_obs

    @property
    def action_ph(self):
        """tf.Tensor: placeholder for actions, shape (self.n_batch, ) + self.ac_space.shape."""
        return self._action_ph

    @staticmethod
    def _kwargs_check(feature_extraction, kwargs):
        """
        Ensure that the user is not passing wrong keywords
        when using policy_kwargs.

        :param feature_extraction: (str)
        :param kwargs: (dict)
        """
        # When using policy_kwargs parameter on model creation,
        # all keywords arguments must be consumed by the policy constructor except
        # the ones for the cnn_extractor network (cf nature_cnn()), where the keywords arguments
        # are not passed explicitly (using **kwargs to forward the arguments)
        # that's why there should be not kwargs left when using the mlp_extractor
        # (in that case the keywords arguments are passed explicitly)
        if feature_extraction == 'mlp' and len(kwargs) > 0:
            raise ValueError("Unknown keywords for policy: {}".format(kwargs))

    @abstractmethod
    def step(self, obs, state=None, mask=None):
        """
        Returns the policy for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float], [float], [float], [float]) actions, values, states, neglogp
        """
        raise NotImplementedError

    @abstractmethod
    def proba_step(self, obs, state=None, mask=None):
        """
        Returns the action probability for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float]) the action probability
        """
        raise NotImplementedError


class ActorCriticPolicy(BasePolicy):
    """
    Policy object that implements actor critic

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param scale: (bool) whether or not to scale the input
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, scale=False, dist_type=None):
        super(ActorCriticPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                                scale=scale)
        self._pdtype = make_proba_dist_type(ac_space, dist_type=dist_type)
        self._is_discrete = isinstance(ac_space, Discrete)
        self._policy = None
        self._proba_distribution = None
        self._value_fn = None
        self._value_fn_d = None
        self._action = None
        self._deterministic_action = None
        self.dual_critic = False

    def _setup_init(self):
        """Sets up the distributions, actions, and value."""
        with tf.variable_scope("output", reuse=True):
            assert self.policy is not None and self.proba_distribution is not None and self.value_fn is not None
            self._action = self.proba_distribution.sample()
            self._deterministic_action = self.proba_distribution.mode()
            self._neglogp = self.proba_distribution.neglogp(self.action)
            if isinstance(self.proba_distribution, CategoricalProbabilityDistribution):
                self._policy_proba = tf.nn.softmax(self.policy)
            elif isinstance(self.proba_distribution, DiagGaussianProbabilityDistribution):
                self._policy_proba = [self.proba_distribution.mean, self.proba_distribution.std]
            elif isinstance(self.proba_distribution, BetaProbabilityDistribution):
                self._policy_proba = [self.proba_distribution.alpha, self.proba_distribution.beta]
            elif isinstance(self.proba_distribution, BernoulliProbabilityDistribution):
                self._policy_proba = tf.nn.sigmoid(self.policy)
            elif isinstance(self.proba_distribution, MultiCategoricalProbabilityDistribution):
                self._policy_proba = [tf.nn.softmax(categorical.flatparam())
                                     for categorical in self.proba_distribution.categoricals]
            elif isinstance(self.proba_distribution, MixProbabilityDistribution):
                self._policy_proba = [tf.nn.sigmoid(self.policy[:, 0]), self.proba_distribution.g_mean, self.proba_distribution.g_std]#self.proba_distribution.gaussian.mean, self.proba_distribution.gaussian.std]
            else:
                self._policy_proba = []  # it will return nothing, as it is not implemented
            if self.dual_critic:
                self._value_flat = tf.minimum(self.value_fn[:, 0], self.value_fn_d[:, 0])
                self._critic_discrepancy = tf.square(self.value_fn[:, 0] - self.value_fn_d[:, 0])
            else:
                self._value_flat = self.value_fn[:, 0]

    @property
    def pdtype(self):
        """ProbabilityDistributionType: type of the distribution for stochastic actions."""
        return self._pdtype

    @property
    def policy(self):
        """tf.Tensor: policy output, e.g. logits."""
        return self._policy

    @property
    def proba_distribution(self):
        """ProbabilityDistribution: distribution of stochastic actions."""
        return self._proba_distribution

    @property
    def value_fn(self):
        """tf.Tensor: value estimate, of shape (self.n_batch, 1)"""
        return self._value_fn

    @property
    def value_fn_d(self):
        """tf.Tensor: value estimate, of shape (self.n_batch, 1)"""
        return self._value_fn_d

    @property
    def value_flat(self):
        """tf.Tensor: value estimate, of shape (self.n_batch, )"""
        return self._value_flat

    @property
    def action(self):
        """tf.Tensor: stochastic action, of shape (self.n_batch, ) + self.ac_space.shape."""
        return self._action

    @property
    def deterministic_action(self):
        """tf.Tensor: deterministic action, of shape (self.n_batch, ) + self.ac_space.shape."""
        return self._deterministic_action

    @property
    def neglogp(self):
        """tf.Tensor: negative log likelihood of the action sampled by self.action."""
        return self._neglogp

    @property
    def policy_proba(self):
        """tf.Tensor: parameters of the probability distribution. Depends on pdtype."""
        return self._policy_proba

    @abstractmethod
    def step(self, obs, state=None, mask=None, deterministic=False):
        """
        Returns the policy for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: ([float], [float], [float], [float]) actions, values, states, neglogp
        """
        raise NotImplementedError

    @abstractmethod
    def value(self, obs, state=None, mask=None):
        """
        Returns the value for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float]) The associated value of the action
        """
        raise NotImplementedError


class RecurrentActorCriticPolicy(ActorCriticPolicy):
    """
    Actor critic policy object uses a previous state in the computation for the current step.
    NOTE: this class is not limited to recurrent neural network policies,
    see https://github.com/hill-a/stable-baselines/issues/241

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param state_shape: (tuple<int>) shape of the per-environment state space.
    :param reuse: (bool) If the policy is reusable or not
    :param scale: (bool) whether or not to scale the input
    """

    recurrent = True

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                 state_shape, reuse=False, scale=False):
        super(RecurrentActorCriticPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps,
                                                         n_batch, reuse=reuse, scale=scale)

        with tf.variable_scope("input", reuse=False):
            self._dones_ph = tf.placeholder(tf.float32, (n_batch, ), name="dones_ph")  # (done t-1)
            state_ph_shape = (self.n_env, ) + tuple(state_shape)
            self._states_ph = tf.placeholder(tf.float32, state_ph_shape, name="states_ph")

        initial_state_shape = (self.n_env, ) + tuple(state_shape)
        self._initial_state = np.zeros(initial_state_shape, dtype=np.float32)

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def dones_ph(self):
        """tf.Tensor: placeholder for whether episode has terminated (done), shape (self.n_batch, ).
        Internally used to reset the state before the next episode starts."""
        return self._dones_ph

    @property
    def states_ph(self):
        """tf.Tensor: placeholder for states, shape (self.n_env, ) + state_shape."""
        return self._states_ph

    @abstractmethod
    def value(self, obs, state=None, mask=None):
        """
        Cf base class doc.
        """
        raise NotImplementedError


class LstmPolicy(RecurrentActorCriticPolicy):
    """
    Policy object that implements actor critic, using LSTMs.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) The size of the Neural network before the LSTM layer  (if None, default to [64, 64])
    :param net_arch: (list) Specification of the actor-critic policy network architecture. Notation similar to the
        format described in mlp_extractor but with additional support for a 'lstm' entry in the shared network part.
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param layer_norm: (bool) Whether or not to use layer normalizing LSTMs
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    recurrent = True

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, layers=None,
                 net_arch=None, act_fun=tf.tanh, cnn_extractor=nature_cnn, layer_norm=False, feature_extraction="cnn",
                 **kwargs):
        # state_shape = [n_lstm * 2] dim because of the cell and hidden states of the LSTM
        super(LstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                                         state_shape=(2 * n_lstm, ), reuse=reuse,
                                         scale=(feature_extraction == "cnn"))

        self._kwargs_check(feature_extraction, kwargs)

        if net_arch is None:  # Legacy mode
            if layers is None:
                layers = [64, 64]
            else:
                warnings.warn("The layers parameter is deprecated. Use the net_arch parameter instead.")

            with tf.variable_scope("model", reuse=reuse):
                if feature_extraction == "cnn":
                    extracted_features = cnn_extractor(self.processed_obs, **kwargs)
                else:
                    extracted_features = tf.layers.flatten(self.processed_obs)
                    for i, layer_size in enumerate(layers):
                        extracted_features = act_fun(linear(extracted_features, 'pi_fc' + str(i), n_hidden=layer_size,
                                                            init_scale=np.sqrt(2)))
                input_sequence = batch_to_seq(extracted_features, self.n_env, n_steps)
                masks = batch_to_seq(self.dones_ph, self.n_env, n_steps)
                rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm,
                                             layer_norm=layer_norm)
                rnn_output = seq_to_batch(rnn_output)
                value_fn = linear(rnn_output, 'vf', 1)

                self._proba_distribution, self._policy, self.q_value = \
                    self.pdtype.proba_distribution_from_latent(rnn_output, rnn_output)

            self._value_fn = value_fn
        else:  # Use the new net_arch parameter
            if layers is not None:
                warnings.warn("The new net_arch parameter overrides the deprecated layers parameter.")
            if feature_extraction == "cnn":
                raise NotImplementedError()

            with tf.variable_scope("model", reuse=reuse):
                latent = tf.layers.flatten(self.processed_obs)
                policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
                value_only_layers = []  # Layer sizes of the network that only belongs to the value network

                # Iterate through the shared layers and build the shared parts of the network
                lstm_layer_constructed = False
                for idx, layer in enumerate(net_arch):
                    if isinstance(layer, int):  # Check that this is a shared layer
                        layer_size = layer
                        latent = act_fun(linear(latent, "shared_fc{}".format(idx), layer_size, init_scale=np.sqrt(2)))
                    elif layer == "lstm":
                        if lstm_layer_constructed:
                            raise ValueError("The net_arch parameter must only contain one occurrence of 'lstm'!")
                        input_sequence = batch_to_seq(latent, self.n_env, n_steps)
                        masks = batch_to_seq(self.dones_ph, self.n_env, n_steps)
                        rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm,
                                                     layer_norm=layer_norm)
                        latent = seq_to_batch(rnn_output)
                        lstm_layer_constructed = True
                    else:
                        assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
                        if 'pi' in layer:
                            assert isinstance(layer['pi'],
                                              list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                            policy_only_layers = layer['pi']

                        if 'vf' in layer:
                            assert isinstance(layer['vf'],
                                              list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                            value_only_layers = layer['vf']
                        break  # From here on the network splits up in policy and value network

                # Build the non-shared part of the policy-network
                latent_policy = latent
                for idx, pi_layer_size in enumerate(policy_only_layers):
                    if pi_layer_size == "lstm":
                        raise NotImplementedError("LSTMs are only supported in the shared part of the policy network.")
                    assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
                    latent_policy = act_fun(
                        linear(latent_policy, "pi_fc{}".format(idx), pi_layer_size, init_scale=np.sqrt(2)))

                # Build the non-shared part of the value-network
                latent_value = latent
                for idx, vf_layer_size in enumerate(value_only_layers):
                    if vf_layer_size == "lstm":
                        raise NotImplementedError("LSTMs are only supported in the shared part of the value function "
                                                  "network.")
                    assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
                    latent_value = act_fun(
                        linear(latent_value, "vf_fc{}".format(idx), vf_layer_size, init_scale=np.sqrt(2)))

                if not lstm_layer_constructed:
                    raise ValueError("The net_arch parameter must contain at least one occurrence of 'lstm'!")

                self._value_fn = linear(latent_value, 'vf', 1)
                # TODO: why not init_scale = 0.001 here like in the feedforward
                self._proba_distribution, self._policy, self.q_value = \
                    self.pdtype.proba_distribution_from_latent(latent_policy, latent_value)
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            return self.sess.run([self.deterministic_action, self.value_flat, self.snew, self.neglogp],
                                 {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})
        else:
            return self.sess.run([self.action, self.value_flat, self.snew, self.neglogp],
                                 {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})


class FeedForwardPolicy(ActorCriticPolicy):
    """
    Policy object that implements actor critic, using a feed forward neural network.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) (deprecated, use net_arch instead) The size of the Neural network for the policy
        (if None, default to [64, 64])
    :param net_arch: (list) Specification of the actor-critic policy network architecture (see mlp_extractor
        documentation for details).
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None, net_arch=None,
                 act_fun=tf.tanh, cnn_extractor=nature_cnn, feature_extraction="cnn", dual_critic=False, measure_execution_time=False, **kwargs):
        dist_type = kwargs.pop("dist_type", None)
        super(FeedForwardPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                                scale=(feature_extraction == "cnn"), dist_type=dist_type)

        #self._kwargs_check(feature_extraction, kwargs)

        if layers is not None:
            warnings.warn("Usage of the `layers` parameter is deprecated! Use net_arch instead "
                          "(it has a different semantics though).", DeprecationWarning)
            if net_arch is not None:
                warnings.warn("The new `net_arch` parameter overrides the deprecated `layers` parameter!",
                              DeprecationWarning)

        if net_arch is None:
            if layers is None:
                layers = {"pi": [64, 64], "vf": [128, 128]}
            net_arch = [layers]

        self.dual_critic = dual_critic
        self.measure_execution_time = measure_execution_time
        self.last_execution_time = None

        proba_kw = {"init_bias": kwargs.pop("init_bias", 0.0), "init_scale": kwargs.pop("init_scale", 0.01)}
        if "init_bias_vf" in kwargs:
            proba_kw["init_bias_vf"] = kwargs.pop("init_bias_vf")

        with tf.variable_scope("model", reuse=reuse):
            if feature_extraction == "cnn":
                pi_latent = vf_latent = cnn_extractor(self.processed_obs, **kwargs)
            else:
                if feature_extraction == "cnn_mlp":
                    latents = cnn_mlp_extractor(self.processed_obs, net_arch, act_fun, dual_critic=dual_critic, **kwargs)
                else:
                    latents = mlp_extractor(tf.layers.flatten(self.processed_obs), net_arch, act_fun, dual_critic=dual_critic, **kwargs)

                if self.dual_critic:
                    pi_latent, vf_latent, vf_latent_d = latents
                else:
                    pi_latent, vf_latent = latents

            self._value_fn = linear(vf_latent, 'vf', 1)
            if self.dual_critic:
                self._value_fn_d = linear(vf_latent_d, "vf_d", 1)

            # TODO: maybe take minimum here
            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, **proba_kw)

        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if self.measure_execution_time:
            start_time = time.process_time()
            action = self.sess.run(self.deterministic_action if deterministic else self.action, {self.obs_ph: obs})
            self.last_execution_time = time.process_time() - start_time
            value, neglogp = self.sess.run([self.value_flat, self.neglogp],{self.obs_ph: obs})
        else:
            if deterministic:
                action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                       {self.obs_ph: obs})
            else:
                action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp], {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})

    def get_critic_discrepancy(self, obs):
        assert self.dual_critic

        return self.sess.run(self._critic_discrepancy, {self.obs_ph: obs})


class LQRPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, A, B, Q, R, obs_module_indices=None, std=1, reuse=False, layers=None, net_arch=None,
                 act_fun=tf.tanh, cnn_extractor=nature_cnn, feature_extraction="mlp", measure_execution_time=False, **kwargs):
        from stable_baselines.lqr.policy import LQR
        #dist_type = kwargs.pop("dist_type", "guassian")
        super(LQRPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                                scale=(feature_extraction == "cnn"))#, dist_type=dist_type)

        #self._kwargs_check(feature_extraction, kwargs)
        self.lqr = LQR(A, B, Q, R)

        if layers is not None:
            warnings.warn("Usage of the `layers` parameter is deprecated! Use net_arch instead "
                          "(it has a different semantics though).", DeprecationWarning)
            if net_arch is not None:
                warnings.warn("The new `net_arch` parameter overrides the deprecated `layers` parameter!",
                              DeprecationWarning)

        if net_arch is None:
            if layers is None:
                layers = {"vf": [128, 128]}
            assert "pi" not in layers
            net_arch = [layers]

        self.measure_execution_time = measure_execution_time
        self.last_execution_time = None

        with tf.variable_scope("model", reuse=reuse):
            K_num = tf.constant(self.lqr.K_num.astype(np.float32))
            K_grad = tf.constant(self.lqr._grad_K().astype(np.float32))
            weights_num = tf.constant(self.lqr.get_weights().astype(np.float32).reshape(1, -1))
            #self.lqr_K = tf.get_variable(initializer=K_num, trainable=False, name="LQR_K", use_resource=True)
            #self.lqr_K_grad = tf.get_variable(initializer=K_grad, trainable=False, name="LQR_K_grad", use_resource=True)
            #self.weights = tf.get_variable(initializer=weights_num, trainable=True, name="LQR_weights", use_resource=True)

            if obs_module_indices is not None:
                self.obs_module_indices = obs_module_indices

                vf_mask = np.array([m == "et" for m in obs_module_indices])
                lqr_mask = np.array([m == "lqr" for m in obs_module_indices])
                vf_obs = tf.boolean_mask(self.processed_obs, vf_mask, name="et_obs", axis=1)
                vf_obs.set_shape([None, sum(vf_mask)])
                lqr_obs = tf.boolean_mask(self.processed_obs, lqr_mask, name="lqr_obs", axis=1)
                lqr_obs.set_shape([None, sum(lqr_mask)])

            if feature_extraction == "cnn":
                vf_latent = cnn_extractor(vf_obs, **kwargs)
            else:
                if feature_extraction == "cnn_mlp":
                    latents = cnn_mlp_extractor(vf_obs, net_arch, act_fun, **kwargs)
                else:
                    latents = mlp_extractor(tf.layers.flatten(vf_obs), net_arch, act_fun, **kwargs)

                vf_latent = latents[1]
        with tf.variable_scope("model", reuse=tf.AUTO_REUSE, use_resource=True):
            @tf.custom_gradient
            def lqr_action(x):
                lqr_K = tf.get_variable(initializer=K_num, trainable=False, name="LQR_K", use_resource=True)
                weights = tf.get_variable(initializer=weights_num, trainable=True, name="LQR_weights", use_resource=True)
                u_lqr = -tf.matmul(x, tf.transpose(lqr_K))
                u_lqr += tf.reduce_sum(weights) * 0.0
                def grad(dy, variables=None):
                    lqr_K_grad = tf.get_variable(initializer=K_grad, trainable=False, name="LQR_K_grad", use_resource=True)
                    #dy = tf.Print(dy, [tf.reduce_mean(dy)], "dy: ")
                    return None, [tf.matmul(tf.transpose(dy), tf.matmul(x, -lqr_K_grad))]
                    #return (dy * x, [tf.reduce_mean(dy * -tf.matmul(x, lqr_K_grad), axis=0, keepdims=True)])
                return u_lqr, grad

            self.lqr_output = lqr_action(lqr_obs)
            self.weights = tf.get_variable("LQR_weights")

        with tf.variable_scope("model", reuse=reuse):
            self._value_fn = linear(vf_latent, 'vf', 1)

            self._proba_distribution, self._policy = self.pdtype.proba_distribution_from_output(self.lqr_output, std=std)
            self.q_value = self.pdtype.value_from_latent(vf_latent, init_scale=0.01, init_bias=0.0)

        #with tf.variable_scope("model", reuse=True, use_resource=True):
        #    grad = tf.gradients(self.policy, self.weights)
        #    grad2 = tf.gradients(self.policy, self.lqr_K)

        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if self.measure_execution_time:
            start_time = time.process_time()
            action = self.sess.run(self.deterministic_action if deterministic else self.action, {self.obs_ph: obs})
            self.last_execution_time = time.process_time() - start_time
            value, neglogp = self.sess.run([self.value_flat, self.neglogp],{self.obs_ph: obs})
        else:
            if deterministic:
                action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                       {self.obs_ph: obs})
            else:
                action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                       {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})


class ETMPCLQRPolicy(ActorCriticPolicy):  # TODO: check entropy, KL (that they are correct)
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, A, B, Q, R, obs_module_indices, std=1, reuse=False, layers=None, net_arch=None,
                 act_fun=tf.tanh, cnn_extractor=nature_cnn, feature_extraction="mlp", measure_execution_time=False, **kwargs):
        from stable_baselines.lqr.policy import LQR
        dist_type = MixProbabilityDistributionType
        super(ETMPCLQRPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                                scale=(feature_extraction == "cnn"), dist_type=dist_type)

        #self._kwargs_check(feature_extraction, kwargs)
        self.obs_module_indices = obs_module_indices
        self.lqr = LQR(A, B, Q, R)
        self.mpc_actions = []
        self.mpc_actions_mb = []

        if layers is not None:
            warnings.warn("Usage of the `layers` parameter is deprecated! Use net_arch instead "
                          "(it has a different semantics though).", DeprecationWarning)
            if net_arch is not None:
                warnings.warn("The new `net_arch` parameter overrides the deprecated `layers` parameter!",
                              DeprecationWarning)

        if net_arch is None:
            if layers is None:
                layers = {"vf": [128, 128], "pi": [32, 32]}
            net_arch = [layers]

        self.measure_execution_time = measure_execution_time
        self.last_execution_time = None

        with tf.variable_scope("model", reuse=reuse):
            K_num = tf.constant(self.lqr.K_num.astype(np.float32))
            K_grad = tf.constant(self.lqr._grad_K().astype(np.float32))
            weights_num = tf.constant(self.lqr.get_weights().astype(np.float32).reshape(1, -1))
            #self.lqr_K = tf.get_variable(initializer=K_num, trainable=False, name="LQR_K", use_resource=True)
            #self.lqr_K_grad = tf.get_variable(initializer=K_grad, trainable=False, name="LQR_K_grad", use_resource=True)
            #self.weights = tf.get_variable(initializer=weights_num, trainable=True, name="LQR_weights", use_resource=True)

            mpc_action_shape = [None] + list(ac_space.shape)
            mpc_action_shape[-1] -= 1
            self.mpc_action_ph = tf.placeholder(tf.float32, shape=mpc_action_shape, name="mpc_action_ph")

            et_mask = np.array([m == "et" for m in self.obs_module_indices])
            lqr_mask = np.array([m == "lqr" for m in self.obs_module_indices])
            et_obs = tf.boolean_mask(self.processed_obs, et_mask, name="et_obs", axis=1)
            et_obs.set_shape([None, sum(et_mask)])
            lqr_obs = tf.boolean_mask(self.processed_obs, lqr_mask, name="lqr_obs", axis=1)
            lqr_obs.set_shape([None, sum(lqr_mask)])

            if feature_extraction == "cnn":
                pi_latent, vf_latent = cnn_extractor(et_obs, **kwargs)
            else:
                if feature_extraction == "cnn_mlp":
                    latents = cnn_mlp_extractor(et_obs, net_arch, act_fun, **kwargs)
                else:
                    latents = mlp_extractor(tf.layers.flatten(et_obs), net_arch, act_fun, **kwargs)

                pi_latent, vf_latent = latents
        with tf.variable_scope("model", reuse=reuse, use_resource=True):
            @tf.custom_gradient
            def lqr_action(x):
                lqr_K = tf.get_variable(initializer=K_num, trainable=False, name="LQR_K", use_resource=True)
                weights = tf.get_variable(initializer=weights_num, trainable=True, name="LQR_weights", use_resource=True)
                u_lqr = -tf.matmul(x, tf.transpose(lqr_K))
                u_lqr += tf.reduce_sum(weights) * 0.0
                def grad(dy, variables=None):
                    lqr_K_grad = tf.get_variable(initializer=K_grad, trainable=False, name="LQR_K_grad", use_resource=True)
                    #dy = tf.Print(dy, [tf.reduce_mean(dy)], "dy: ")
                    return None, [tf.matmul(tf.transpose(dy), tf.matmul(x, -lqr_K_grad))]
                    #return (dy * x, [tf.reduce_mean(dy * -tf.matmul(x, lqr_K_grad), axis=0, keepdims=True)])
                return u_lqr, grad

            self.lqr_output = lqr_action(lqr_obs)
            #tf.add(tf.stop_gradient(self.mpc_action_ph), self.lqr_output, "dm_output")
            #self.weights = tf.get_variable("LQR_weights")

        with tf.variable_scope("model", reuse=reuse):
            self._value_fn = linear(vf_latent, 'vf', 1)

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, self.lqr_output, g_std=std, init_scale=0.01,
                                                           init_bias=-0.0, init_bias_vf=0.0) # TODO: set bias to 3

            # self._policy is just the et decision

        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if self.measure_execution_time:
            start_time = time.process_time()
            action = self.sess.run(self.deterministic_action if deterministic else self.action, {self.obs_ph: obs})
            self.last_execution_time = time.process_time() - start_time
            value, neglogp = self.sess.run([self.value_flat, self.neglogp],{self.obs_ph: obs})
        else:
            if deterministic:
                action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                       {self.obs_ph: obs, self.mpc_action_ph: np.zeros(shape=(obs.shape[0], *self.mpc_action_ph.shape[1:]))})
            else:
                action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp], {self.obs_ph: obs, self.mpc_action_ph: np.zeros(shape=(obs.shape[0], *self.mpc_action_ph.shape[1:]))})
        #action[0, 0] = 0  # TODO:remove (sanity check if same results as just lqr)
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})


class AHETMPCLQRPolicy(ActorCriticPolicy):  # TODO: check entropy, KL (that they are correct)
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, A, B, Q, R, obs_module_indices, time_varying=False, n_lqr=1, std=1, reuse=False, layers=None, net_arch=None,
                 act_fun=tf.tanh, cnn_extractor=nature_cnn, feature_extraction="mlp", measure_execution_time=False, **kwargs):
        from stable_baselines.lqr.policy import LQR
        dist_type = RLMPCProbabilityDistributionType
        super(AHETMPCLQRPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                                scale=(feature_extraction == "cnn"), dist_type=dist_type)

        #self._kwargs_check(feature_extraction, kwargs)
        A = [A for _ in range(n_lqr)]  # TODO: maybe check if already multienv A and B
        B = [B for _ in range(n_lqr)]

        self.time_varying = time_varying
        self.obs_module_indices = obs_module_indices
        self.lqr = LQR(A, B, Q, R, time_varying=time_varying)
        self.lqr_As = []
        self.lqr_Bs = []
        self.lqr_system_idxs = []

        #self.origKs = []
        #self.d_actions = []

        if layers is not None:
            warnings.warn("Usage of the `layers` parameter is deprecated! Use net_arch instead "
                          "(it has a different semantics though).", DeprecationWarning)
            if net_arch is not None:
                warnings.warn("The new `net_arch` parameter overrides the deprecated `layers` parameter!",
                              DeprecationWarning)

        if net_arch is None:
            if layers is None:
                layers = {"vf": [128, 128], "pi": [32, 32]}
            net_arch = [layers]

        self.measure_execution_time = measure_execution_time
        self.last_execution_time = None

        with tf.variable_scope("model", reuse=reuse):
            if self.time_varying:
                self.lqr_K_ph = tf.placeholder(tf.float32, shape=(None, *self.lqr.K_num.shape[-2:]), name="lqr_K_ph")
                self.lqr_K_grad_ph = tf.placeholder(tf.float32,
                                                    shape=(None, self.lqr.weights_size, *self.lqr.K_num.shape[-2:]),
                                                    name="lqr_K_grad_ph")
            else:
                self.lqr_K_ph = tf.placeholder(tf.float32, shape=self.lqr.K_num.shape, name="lqr_K_ph")
                self.lqr_K_grad_ph = tf.placeholder(tf.float32,
                                                    shape=(self.lqr.weights_size, *self.lqr.K_num.shape[-2:]),
                                                    name="lqr_K_grad_ph")
            weights_num = tf.constant(self.lqr.get_weights().astype(np.float32).reshape(1, -1))

            #mpc_action_shape = [None] + list(ac_space.shape)
            #mpc_action_shape[-1] -= 1
            #self.mpc_action_ph = tf.placeholder(tf.float32, shape=mpc_action_shape, name="mpc_action_ph")

            et_mask = np.array([m == "et" for m in self.obs_module_indices])
            lqr_mask = np.array([m == "lqr" for m in self.obs_module_indices])
            et_obs = tf.boolean_mask(self.processed_obs, et_mask, name="et_obs", axis=1)
            et_obs.set_shape([None, sum(et_mask)])
            lqr_obs = tf.boolean_mask(self.processed_obs, lqr_mask, name="lqr_obs", axis=1)
            lqr_obs.set_shape([None, sum(lqr_mask)])

            init_bias = kwargs.pop("init_bias", 0.0)
            init_bias_vf = kwargs.pop("init_bias_vf", 0.0)
            init_scale = kwargs.pop("init_scale", 0.01)
            max_horizon = kwargs.pop("max_horizon", 50.0)
            init_bias_horizon = kwargs.pop("init_bias_horizon", max_horizon / 2)

            if self.time_varying:  # k is first of the lqr variables
                self.lqr_k_idx = obs_module_indices.index("lqr")
            else:
                self.lqr_k_idx = None

            if feature_extraction == "cnn":
                pi_latent, vf_latent = cnn_extractor(et_obs, **kwargs)
            else:
                if feature_extraction == "cnn_mlp":
                    latents = cnn_mlp_extractor(et_obs, net_arch, act_fun, **kwargs)
                else:
                    latents = mlp_extractor(tf.layers.flatten(et_obs), net_arch, act_fun, **kwargs)

                pi_latent, vf_latent = latents
        with tf.variable_scope("model", reuse=reuse, use_resource=True):
            @tf.custom_gradient
            def lqr_action(x):
                weights = tf.get_variable(initializer=weights_num, trainable=True, name="LQR_weights", use_resource=True)
                if time_varying:
                    k, x = x[:, 0], tf.expand_dims(x[:, 1:], axis=-1)
                    # Ks = tf.gather(lqr_K, tf.cast(k, tf.int32), axis=0)
                    u_lqr = -tf.squeeze(tf.matmul(self.lqr_K_ph, x), axis=-1)
                else:
                    u_lqr = -tf.matmul(x, tf.transpose(self.lqr_K_ph))
                u_lqr += tf.reduce_sum(weights) * 0.0

                def grad(dy, variables=None):
                    # dy = tf.Print(dy, [tf.reduce_mean(dy)], "dy: ")
                    if time_varying:
                        # grad_Ks = tf.gather(lqr_K_grad, tf.cast(k,  tf.int32), axis=0)
                        return None, [tf.matmul(dy, -tf.squeeze(tf.matmul(self.lqr_K_grad_ph, tf.expand_dims(x, axis=1)), axis=[-2, -1]), transpose_a=True)]
                    else:
                        return None, [tf.matmul(dy, -tf.matmul(x, self.lqr_K_grad_ph), transpose_a=True)]
                    # return (dy * x, [tf.reduce_mean(dy * -tf.matmul(x, lqr_K_grad), axis=0, keepdims=True)])

                return u_lqr, grad

            self.lqr_output = lqr_action(lqr_obs)
            #tf.add(tf.stop_gradient(self.mpc_action_ph), self.lqr_output, "dm_output")
            #self.weights = tf.get_variable("LQR_weights")

        with tf.variable_scope("model", reuse=reuse):
            self._value_fn = linear(vf_latent, 'vf', 1)

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, self.lqr_output, g_std=std, init_scale=init_scale,
                                                           init_bias=init_bias, init_bias_vf=init_bias_vf, init_bias_horizon=init_bias_horizon, max_horizon=max_horizon) # TODO: try horizon output as tanh and multiply by limits

        if False:
            with tf.variable_scope("model", reuse=True):
                from stable_baselines.common.distributions import BernoulliProbabilityDistributionType, DiagGaussianProbabilityDistributionType
                self.pb_et = BernoulliProbabilityDistributionType(1).proba_distribution_from_latent(pi_latent, vf_latent, init_scale=init_scale, init_bias=init_bias)
                self.pb_hor = GeneralizedPoissonProbabilityDistributionType(1).proba_distribution_from_latent(pi_latent, vf_latent, init_scale=init_scale, init_bias=init_bias_horizon)
                self.pb_g = DiagGaussianProbabilityDistributionType(1).proba_distribution_from_output(self.lqr_output, std=std)

            # self._policy is just the et decision

        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if self.measure_execution_time:
            start_time = time.process_time()
            action = self.sess.run(self.deterministic_action if deterministic else self.action, {self.obs_ph: obs})
            self.last_execution_time = time.process_time() - start_time
            value, neglogp = self.sess.run([self.value_flat, self.neglogp],{self.obs_ph: obs})
        else:
            K = self.lqr.get_numeric_value("K")
            if self.time_varying:
                if obs.shape[0] > 1:
                    if isinstance(K, list):
                        t_idx = np.minimum(obs[..., self.lqr_k_idx], [len(K_i) - 1 for K_i in K]).astype(np.int32)
                    else:
                        t_idx = np.minimum(obs[..., self.lqr_k_idx], K.shape[1] - 1).astype(np.int32)
                    K = np.array([K[i][t_idx[i]] for i in range(len(K))]).reshape(obs.shape[0], *self.lqr.K.shape)
                else:
                    K = K[int(min(obs[..., self.lqr_k_idx], len(K) - 1))].reshape(1, *self.lqr.K_num.shape[1:])
            if deterministic:
                action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                       {self.obs_ph: obs, self.lqr_K_ph: K})
            else:
                action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                       {self.obs_ph: obs, self.lqr_K_ph: K})

            #self.d_actions.append(self.sess.run(self.deterministic_action, {self.obs_ph: obs, self.lqr_K_ph: K}))
            """
            if deterministic:
                action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                       {self.obs_ph: obs, self.mpc_action_ph: np.zeros(shape=(obs.shape[0], *self.mpc_action_ph.shape[1:]))})
            else:
                action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp], {self.obs_ph: obs, self.mpc_action_ph: np.zeros(shape=(obs.shape[0], *self.mpc_action_ph.shape[1:]))})
            """
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})


if tfp_import:
    class AHMPCPolicy(FeedForwardPolicy):
        def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None, net_arch=None,
                     act_fun=tf.tanh, cnn_extractor=nature_cnn, feature_extraction="mlp", measure_execution_time=False, **kwargs):
            self.dist_type = GeneralizedPoissonProbabilityDistributionType#PoissonProbabilityDistributionType,
            super(AHMPCPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse, dist_type=self.dist_type,
                                              layers=layers, net_arch=net_arch, act_fun=act_fun,
                                            feature_extraction="mlp", **kwargs)

        def step(self, obs, state=None, mask=None, deterministic=False):
            if True or self.dist_type != GeneralizedPoissonProbabilityDistributionType:
                return super().step(obs, state, mask, deterministic)
            if self.measure_execution_time:
                start_time = time.process_time()
                action = self.sess.run(self.deterministic_action if deterministic else self.action, {self.obs_ph: obs})
                self.last_execution_time = time.process_time() - start_time
                value, neglogp = self.sess.run([self.value_flat, self.neglogp], {self.obs_ph: obs})
            else:
                if deterministic:
                    action = self.sess.run(self.deterministic_action, {self.obs_ph: obs})
                    value, neglogp = self.sess.run([self.value_flat, self.neglogp],
                                                           {self.obs_ph: obs, self.proba_distribution.sample_ph: action})
                else:
                    samples_from_uniform = list(np.random.uniform(0.0, 1.0 - 1e-3, obs.shape[0]))
                    rate, delta = self.sess.run([self.proba_distribution.rate, self.proba_distribution.delta],
                                                {self.obs_ph: obs})
                    pmf_c_1 = self.proba_distribution.pmf(0, rate, delta)
                    c_p = [[pmf_c_1[i].item()] for i in range(obs.shape[0])]
                    count = 1
                    samples = []
                    while len(samples_from_uniform) > 0:
                        pmf_c_1 = self.proba_distribution.pmf(count, rate, delta, pmf_c_1)
                        for i in range(obs.shape[0]):
                            c_p[i].append(c_p[i][-1] + pmf_c_1[i].item())
                        pop_is = []
                        for i in range(len(c_p)):
                            if i < len(samples_from_uniform):
                                if c_p[i][count - 1] <= samples_from_uniform[i] <= c_p[i][count] or np.isnan(c_p[i][-1]) or (c_p[i][-1] >= 0.95 and pmf_c_1[i].item() < 1e-5):  # TODO: is gonna be wrong if there are multiple actions per obs
                                    samples.append(count)
                                    pop_is.append(i)

                        if len(pop_is) > 0:
                            for i in reversed(pop_is):
                                samples_from_uniform.pop(i)
                        count += 1
                    action = np.array(samples).reshape(obs.shape[0], -1)
                    value, neglogp = self.sess.run([self.value_flat, self.neglogp],
                                                        {self.obs_ph: obs, self.proba_distribution.sample_ph: action})
            return action, value, self.initial_state, neglogp



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

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(CnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="cnn", **_kwargs)


class CnnLstmPolicy(LstmPolicy):
    """
    Policy object that implements actor critic, using LSTMs with a CNN feature extraction

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, **_kwargs):
        super(CnnLstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                                            layer_norm=False, feature_extraction="cnn", **_kwargs)


class CnnLnLstmPolicy(LstmPolicy):
    """
    Policy object that implements actor critic, using a layer normalized LSTMs with a CNN feature extraction

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, **_kwargs):
        super(CnnLnLstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                                              layer_norm=True, feature_extraction="cnn", **_kwargs)


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

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(MlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="mlp", **_kwargs)


class MlpLstmPolicy(LstmPolicy):
    """
    Policy object that implements actor critic, using LSTMs with a MLP feature extraction

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, **_kwargs):
        super(MlpLstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                                            layer_norm=False, feature_extraction="mlp", **_kwargs)


class MlpLnLstmPolicy(LstmPolicy):
    """
    Policy object that implements actor critic, using a layer normalized LSTMs with a MLP feature extraction

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, **_kwargs):
        super(MlpLnLstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                                              layer_norm=True, feature_extraction="mlp", **_kwargs)


class CnnMlpPolicy(FeedForwardPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(CnnMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="cnn_mlp", **_kwargs)


_policy_registry = {
    ActorCriticPolicy: {
        "CnnPolicy": CnnPolicy,
        "CnnLstmPolicy": CnnLstmPolicy,
        "CnnLnLstmPolicy": CnnLnLstmPolicy,
        "MlpPolicy": MlpPolicy,
        "MlpLstmPolicy": MlpLstmPolicy,
        "MlpLnLstmPolicy": MlpLnLstmPolicy,
        "CnnMlpPolicy": CnnMlpPolicy,
        "LQRPolicy": LQRPolicy
    }
}


def get_policy_from_name(base_policy_type, name):
    """
    returns the registed policy from the base type and name

    :param base_policy_type: (BasePolicy) the base policy object
    :param name: (str) the policy name
    :return: (base_policy_type) the policy
    """
    if base_policy_type not in _policy_registry:
        raise ValueError("Error: the policy type {} is not registered!".format(base_policy_type))
    if name not in _policy_registry[base_policy_type]:
        raise ValueError("Error: unknown policy type {}, the only registed policy type are: {}!"
                         .format(name, list(_policy_registry[base_policy_type].keys())))
    return _policy_registry[base_policy_type][name]


def register_policy(name, policy):
    """
    returns the registed policy from the base type and name

    :param name: (str) the policy name
    :param policy: (subclass of BasePolicy) the policy
    """
    sub_class = None
    for cls in BasePolicy.__subclasses__():
        if issubclass(policy, cls):
            sub_class = cls
            break
    if sub_class is None:
        raise ValueError("Error: the policy {} is not of any known subclasses of BasePolicy!".format(policy))

    if sub_class not in _policy_registry:
        _policy_registry[sub_class] = {}
    if name in _policy_registry[sub_class]:
        raise ValueError("Error: the name {} is alreay registered for a different policy, will not override."
                         .format(name))
    _policy_registry[sub_class][name] = policy
