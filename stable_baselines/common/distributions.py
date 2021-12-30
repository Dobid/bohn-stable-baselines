import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
from gym import spaces

from stable_baselines.common.tf_layers import linear


class ProbabilityDistribution(object):
    """
    Base class for describing a probability distribution.
    """
    def __init__(self):
        super(ProbabilityDistribution, self).__init__()

    def flatparam(self):
        """
        Return the direct probabilities

        :return: ([float]) the probabilities
        """
        raise NotImplementedError

    def mode(self):
        """
        Returns the probability

        :return: (Tensorflow Tensor) the deterministic action
        """
        raise NotImplementedError

    def neglogp(self, x):
        """
        returns the of the negative log likelihood

        :param x: (str) the labels of each index
        :return: ([float]) The negative log likelihood of the distribution
        """
        # Usually it's easier to define the negative logprob
        raise NotImplementedError

    def kl(self, other):
        """
        Calculates the Kullback-Leibler divergence from the given probability distribution

        :param other: ([float]) the distribution to compare with
        :return: (float) the KL divergence of the two distributions
        """
        raise NotImplementedError

    def entropy(self):
        """
        Returns Shannon's entropy of the probability

        :return: (float) the entropy
        """
        raise NotImplementedError

    def sample(self):
        """
        returns a sample from the probability distribution

        :return: (Tensorflow Tensor) the stochastic action
        """
        raise NotImplementedError

    def logp(self, x):
        """
        returns the of the log likelihood

        :param x: (str) the labels of each index
        :return: ([float]) The log likelihood of the distribution
        """
        return - self.neglogp(x)


class ProbabilityDistributionType(object):
    """
    Parametrized family of probability distributions
    """

    def probability_distribution_class(self):
        """
        returns the ProbabilityDistribution class of this type

        :return: (Type ProbabilityDistribution) the probability distribution class associated
        """
        raise NotImplementedError

    def proba_distribution_from_flat(self, flat):
        """
        Returns the probability distribution from flat probabilities
        flat: flattened vector of parameters of probability distribution

        :param flat: ([float]) the flat probabilities
        :return: (ProbabilityDistribution) the instance of the ProbabilityDistribution associated
        """
        return self.probability_distribution_class()(flat)

    def proba_distribution_from_latent(self, pi_latent_vector, vf_latent_vector, init_scale=1.0, init_bias=0.0):
        """
        returns the probability distribution from latent values

        :param pi_latent_vector: ([float]) the latent pi values
        :param vf_latent_vector: ([float]) the latent vf values
        :param init_scale: (float) the initial scale of the distribution
        :param init_bias: (float) the initial bias of the distribution
        :return: (ProbabilityDistribution) the instance of the ProbabilityDistribution associated
        """
        raise NotImplementedError

    def param_shape(self):
        """
        returns the shape of the input parameters

        :return: ([int]) the shape
        """
        raise NotImplementedError

    def sample_shape(self):
        """
        returns the shape of the sampling

        :return: ([int]) the shape
        """
        raise NotImplementedError

    def sample_dtype(self):
        """
        returns the type of the sampling

        :return: (type) the type
        """
        raise NotImplementedError

    def param_placeholder(self, prepend_shape, name=None):
        """
        returns the TensorFlow placeholder for the input parameters

        :param prepend_shape: ([int]) the prepend shape
        :param name: (str) the placeholder name
        :return: (TensorFlow Tensor) the placeholder
        """
        return tf.placeholder(dtype=tf.float32, shape=prepend_shape + self.param_shape(), name=name)

    def sample_placeholder(self, prepend_shape, name=None):
        """
        returns the TensorFlow placeholder for the sampling

        :param prepend_shape: ([int]) the prepend shape
        :param name: (str) the placeholder name
        :return: (TensorFlow Tensor) the placeholder
        """
        return tf.placeholder(dtype=self.sample_dtype(), shape=prepend_shape + self.sample_shape(), name=name)


class CategoricalProbabilityDistributionType(ProbabilityDistributionType):
    def __init__(self, n_cat):
        """
        The probability distribution type for categorical input

        :param n_cat: (int) the number of categories
        """
        self.n_cat = n_cat

    def probability_distribution_class(self):
        return CategoricalProbabilityDistribution

    def proba_distribution_from_latent(self, pi_latent_vector, vf_latent_vector, init_scale=1.0, init_bias=0.0):
        pdparam = linear(pi_latent_vector, 'pi', self.n_cat, init_scale=init_scale, init_bias=init_bias)
        q_values = linear(vf_latent_vector, 'q', self.n_cat, init_scale=init_scale, init_bias=init_bias)
        return self.proba_distribution_from_flat(pdparam), pdparam, q_values

    def param_shape(self):
        return [self.n_cat]

    def sample_shape(self):
        return []

    def sample_dtype(self):
        return tf.int64


class MultiCategoricalProbabilityDistributionType(ProbabilityDistributionType):
    def __init__(self, n_vec):
        """
        The probability distribution type for multiple categorical input

        :param n_vec: ([int]) the vectors
        """
        # Cast the variable because tf does not allow uint32
        self.n_vec = n_vec.astype(np.int32)
        # Check that the cast was valid
        assert (self.n_vec > 0).all(), "Casting uint32 to int32 was invalid"

    def probability_distribution_class(self):
        return MultiCategoricalProbabilityDistribution

    def proba_distribution_from_flat(self, flat):
        return MultiCategoricalProbabilityDistribution(self.n_vec, flat)

    def proba_distribution_from_latent(self, pi_latent_vector, vf_latent_vector, init_scale=1.0, init_bias=0.0):
        pdparam = linear(pi_latent_vector, 'pi', sum(self.n_vec), init_scale=init_scale, init_bias=init_bias)
        q_values = linear(vf_latent_vector, 'q', sum(self.n_vec), init_scale=init_scale, init_bias=init_bias)
        return self.proba_distribution_from_flat(pdparam), pdparam, q_values

    def param_shape(self):
        return [sum(self.n_vec)]

    def sample_shape(self):
        return [len(self.n_vec)]

    def sample_dtype(self):
        return tf.int64


class DiagGaussianProbabilityDistributionType(ProbabilityDistributionType):
    def __init__(self, size):
        """
        The probability distribution type for multivariate Gaussian input

        :param size: (int) the number of dimensions of the multivariate gaussian
        """
        self.size = size

    def probability_distribution_class(self):
        return DiagGaussianProbabilityDistribution

    def proba_distribution_from_flat(self, flat):
        """
        returns the probability distribution from flat probabilities

        :param flat: ([float]) the flat probabilities
        :return: (ProbabilityDistribution) the instance of the ProbabilityDistribution associated
        """
        return self.probability_distribution_class()(flat)

    def proba_distribution_from_latent(self, pi_latent_vector, vf_latent_vector, init_scale=1.0, init_bias=0.0):
        mean = linear(pi_latent_vector, 'pi', self.size, init_scale=init_scale, init_bias=init_bias)
        logstd = tf.get_variable(name='pi/logstd', shape=[1, self.size], initializer=tf.zeros_initializer())
        pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
        q_values = linear(vf_latent_vector, 'q', self.size, init_scale=init_scale, init_bias=init_bias)
        return self.proba_distribution_from_flat(pdparam), mean, q_values

    def proba_distribution_from_output(self, pi_output, std=1):
        logstd = np.array([np.log(std) for i in range(self.size)]).astype(np.float32)
        logstd = tf.get_variable(initializer=tf.constant(logstd), trainable=True, name="pi/logstd")
        pd_param = tf.concat([pi_output, pi_output * 0.0 + logstd], axis=1)
        return self.proba_distribution_from_flat(pd_param), pi_output

    def value_from_latent(self, vf_latent_vector, init_scale=1.0, init_bias=0.0):
        return linear(vf_latent_vector, 'q', self.size, init_scale=init_scale, init_bias=init_bias)

    def param_shape(self):
        return [2 * self.size]

    def sample_shape(self):
        return [self.size]

    def sample_dtype(self):
        return tf.float32


class BoundedDiagGaussianProbabilityDistributionType(DiagGaussianProbabilityDistributionType):
    def __init__(self, size):
        """
        The probability distribution type for multivariate Gaussian input

        :param size: (int) the number of dimensions of the multivariate gaussian
        """
        super().__init__(size)

    def probability_distribution_class(self):
        return BoundedDiagGaussianProbabilityDistribution


class BetaProbabilityDistributionType(ProbabilityDistributionType):
    def __init__(self, size):
        """
        The probability distribution type for multivariate beta input
        :param size: (int) the number of dimensions of the multivariate beta
        """
        self.size = size

    def probability_distribution_class(self):
        return BetaProbabilityDistribution

    def proba_distribution_from_flat(self, flat):
        """
        returns the probability distribution from flat probabilities
        :param flat: ([float]) the flat probabilities
        :return: (ProbabilityDistribution) the instance of the ProbabilityDistribution associated
        """
        return self.probability_distribution_class()(flat)

    def proba_distribution_from_latent(self, pi_latent_vector, vf_latent_vector, init_scale=1.0, init_bias=0.0):
        pdparam = linear(pi_latent_vector, 'pi', self.size, init_scale=init_scale, init_bias=init_bias)
        q_values = linear(vf_latent_vector, 'q', self.size, init_scale=init_scale, init_bias=init_bias)
        return self.proba_distribution_from_flat(pdparam), pdparam, q_values

    def param_shape(self):
        return [2 * self.size]

    def sample_shape(self):
        return [self.size]

    def sample_dtype(self):
        return tf.float32


class BernoulliProbabilityDistributionType(ProbabilityDistributionType):
    def __init__(self, size):
        """
        The probability distribution type for Bernoulli input

        :param size: (int) the number of dimensions of the Bernoulli distribution
        """
        self.size = size

    def probability_distribution_class(self):
        return BernoulliProbabilityDistribution

    def proba_distribution_from_latent(self, pi_latent_vector, vf_latent_vector, init_scale=1.0, init_bias=0.0, init_bias_vf=None):
        if init_bias_vf is None:
            init_bias_vf = init_bias
        pdparam = linear(pi_latent_vector, 'pi', self.size, init_scale=init_scale, init_bias=init_bias)
        q_values = linear(vf_latent_vector, 'q', self.size, init_scale=init_scale, init_bias=init_bias_vf)
        return self.proba_distribution_from_flat(pdparam), pdparam, q_values

    def param_shape(self):
        return [self.size]

    def sample_shape(self):
        return [self.size]

    def sample_dtype(self):
        return tf.int32


class GP0GeneralizedPoissonProbabilityDistributionType(ProbabilityDistributionType):
    def __init__(self, size):
        """
        The probability distribution type for Bernoulli input

        :param size: (int) the number of dimensions of the Bernoulli distribution
        """
        self.size = size

    def probability_distribution_class(self):
        return GP0GeneralizedPoissonProbabilityDistribution

    def proba_distribution_from_latent(self, pi_latent_vector, vf_latent_vector, init_scale=1.0, init_bias=0.0,
                                       init_bias_vf=None):
        if init_bias_vf is None:
            init_bias_vf = init_bias
        rate = tf.nn.relu(linear(pi_latent_vector, 'pi_r', self.size, init_scale=init_scale * 500, init_bias=init_bias + 35 * (1 - -0.5)))
        delta = linear(pi_latent_vector, "pi_d", self.size, init_scale=init_scale, init_bias=-0.5)
        #delta = tf.get_variable(name='pi/delta', shape=[1, self.size], initializer=tf.constant_initializer(-0.5))
        q_values = linear(vf_latent_vector, 'q', self.size, init_scale=init_scale, init_bias=init_bias_vf)
        return self.proba_distribution_from_flat(rate, delta), rate, q_values

    def proba_distribution_from_flat(self, rate, delta):
        return self.probability_distribution_class()(rate, delta)

    def param_shape(self):
        return [self.size]

    def sample_shape(self):
        return [self.size]

    def sample_dtype(self):
        return tf.float32


class GeneralizedPoissonProbabilityDistributionType(ProbabilityDistributionType):
    def __init__(self, size):
        """
        The probability distribution type for Bernoulli input

        :param size: (int) the number of dimensions of the Bernoulli distribution
        """
        self.size = size
        self.max_val = 40.0
        self.min_val = 1.0

    def probability_distribution_class(self):
        return GeneralizedPoissonProbabilityDistribution

    def proba_distribution_from_latent(self, pi_latent_vector, vf_latent_vector, init_scale=1.0, init_bias=0.0, init_bias_vf=None):
        if init_bias_vf is None:
            init_bias_vf = init_bias

        w_scale = 0.5 * (self.max_val - self.min_val) * (1 - (-1)) / (1 - (-1))
        #rate, rate_w = linear(pi_latent_vector, 'pi/rate', self.size, init_scale=init_scale, init_bias=init_bias, return_w=True)#, regularizer=tf.contrib.layers.l2_regularizer(tf.constant(1e-2, dtype=tf.float32))))
        #rate = linear(pi_latent_vector, 'pi/rate', self.size, init_scale=init_scale, init_bias=init_bias, w_scale=w_scale)
        rate = linear(pi_latent_vector, 'pi/rate', self.size, init_scale=init_scale, init_bias=init_bias)
        rate = tf.tanh(rate)
        rate = (self.max_val - self.min_val) * (rate - (-1)) / (1 - (-1)) + self.min_val  # scale rate to (min, max) horizon
        #rate = tf.nn.relu(linear(pi_latent_vector, 'pi/rate', self.size, init_scale=init_scale, init_bias=init_bias))
        alpha = tf.get_variable(name='pi/alpha', shape=[1, self.size], initializer=tf.constant_initializer(-1 /( 2 * self.max_val)), constraint=lambda z: tf.clip_by_value(z, -1/ (self.max_val + 1), 1/self.max_val), trainable=False)
        #alpha = tf.constant(-0.015, shape=[1, self.size])
        q_values = linear(vf_latent_vector, 'q', self.size, init_scale=init_scale, init_bias=init_bias_vf)
        pb_dist = self.proba_distribution_from_flat(rate, alpha)
        #pb_dist.rate_w = rate_w
        return pb_dist, rate, q_values

    def proba_distribution_from_flat(self, rate, alpha):
        return self.probability_distribution_class()(rate, alpha, min_val=self.min_val, max_val=self.max_val)

    def param_shape(self):
        return [self.size]

    def sample_shape(self):
        return [self.size]

    def sample_dtype(self):
        return tf.float32



class RLMPCProbabilityDistributionType(ProbabilityDistributionType):
    def __init__(self, size):
        """
        The probability distribution type for Bernoulli input

        :param size: (int) the number of dimensions of the Bernoulli distribution
        """
        self.size = size

    def probability_distribution_class(self):
        return RLMPCProbabilityDistribution

    def proba_distribution_from_latent(self, pi_latent_vector, vf_latent_vector,  g_mean, g_std=1, init_scale=1.0, init_bias=0.0, init_bias_horizon=None, init_bias_vf=None, max_horizon=50.0, min_horizon=1.0, horizon_latent_vector=None):
        if init_bias_vf is None:
            init_bias_vf = init_bias
        if init_bias_horizon is None:
            init_bias_horizon = init_bias

        if horizon_latent_vector is None:
            horizon_latent_vector = pi_latent_vector

        etparam = linear(pi_latent_vector, 'pi/et', 1, init_scale=init_scale, init_bias=init_bias)

        w_scale = 0.5 * (max_horizon - min_horizon) * (1 - (-1)) / (1 - (-1))
        #rate = tf.nn.relu(linear(pi_latent_vector, "pi/horizon_rate", 1, init_scale=init_scale, init_bias=init_bias_horizon))
        rate = tf.tanh(linear(horizon_latent_vector, 'pi/horizon_rate', 1, init_scale=init_scale, init_bias=init_bias_horizon, w_scale=w_scale))
        rate = (max_horizon - min_horizon) * (rate - (-1)) / (1 - (-1)) + min_horizon  # scale rate to (min, max) horizon
        alpha = tf.get_variable(name='pi/horizon_alpha', shape=[1, 1], initializer=tf.constant_initializer(-1 / (2 * max_horizon)),
                                constraint=lambda z: tf.clip_by_value(z, -1 / (max_horizon + 1), 1 / max_horizon), trainable=False)

        logstd_0 = tf.get_variable(name='pi/lqr_logstd', initializer=tf.constant(np.log(g_std, dtype=np.float32), shape=[1, g_mean.shape[-1]]), trainable=True)  # TODO: consider changing to nontrainable
        g_param = tf.concat([g_mean, g_mean * 0.0 + logstd_0], axis=1)

        q_values = linear(vf_latent_vector, 'q', 1, init_scale=init_scale, init_bias=init_bias_vf)
        return self.proba_distribution_from_flat(etparam, rate, alpha, g_param, max_horizon=max_horizon, min_horizon=min_horizon), tf.concat([etparam, rate, g_mean], axis=1), q_values

    def proba_distribution_from_flat(self, et, rate, alpha, g_flat, max_horizon=50.0, min_horizon=1.0):
        return self.probability_distribution_class()(et, rate, alpha, g_flat, max_horizon=max_horizon, min_horizon=min_horizon)

    def param_shape(self):
        return [self.size]

    def sample_shape(self):
        return [self.size]

    def sample_dtype(self):
        return tf.float32


class CategoricalProbabilityDistribution(ProbabilityDistribution):
    def __init__(self, logits):
        """
        Probability distributions from categorical input

        :param logits: ([float]) the categorical logits input
        """
        self.logits = logits
        super(CategoricalProbabilityDistribution, self).__init__()

    def flatparam(self):
        return self.logits

    def mode(self):
        return tf.argmax(self.logits, axis=-1)

    def neglogp(self, x):
        # Note: we can't use sparse_softmax_cross_entropy_with_logits because
        #       the implementation does not allow second-order derivatives...
        one_hot_actions = tf.one_hot(x, self.logits.get_shape().as_list()[-1])
        return tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits,
            labels=tf.stop_gradient(one_hot_actions))

    def kl(self, other):
        a_0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        a_1 = other.logits - tf.reduce_max(other.logits, axis=-1, keepdims=True)
        exp_a_0 = tf.exp(a_0)
        exp_a_1 = tf.exp(a_1)
        z_0 = tf.reduce_sum(exp_a_0, axis=-1, keepdims=True)
        z_1 = tf.reduce_sum(exp_a_1, axis=-1, keepdims=True)
        p_0 = exp_a_0 / z_0
        return tf.reduce_sum(p_0 * (a_0 - tf.log(z_0) - a_1 + tf.log(z_1)), axis=-1)

    def entropy(self):
        a_0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        exp_a_0 = tf.exp(a_0)
        z_0 = tf.reduce_sum(exp_a_0, axis=-1, keepdims=True)
        p_0 = exp_a_0 / z_0
        return tf.reduce_sum(p_0 * (tf.log(z_0) - a_0), axis=-1)

    def sample(self):
        # Gumbel-max trick to sample
        # a categorical distribution (see http://amid.fish/humble-gumbel)
        uniform = tf.random_uniform(tf.shape(self.logits), dtype=self.logits.dtype)
        return tf.argmax(self.logits - tf.log(-tf.log(uniform)), axis=-1)

    @classmethod
    def fromflat(cls, flat):
        """
        Create an instance of this from new logits values

        :param flat: ([float]) the categorical logits input
        :return: (ProbabilityDistribution) the instance from the given categorical input
        """
        return cls(flat)


class MultiCategoricalProbabilityDistribution(ProbabilityDistribution):
    def __init__(self, nvec, flat):
        """
        Probability distributions from multicategorical input

        :param nvec: ([int]) the sizes of the different categorical inputs
        :param flat: ([float]) the categorical logits input
        """
        self.flat = flat
        self.categoricals = list(map(CategoricalProbabilityDistribution, tf.split(flat, nvec, axis=-1)))
        super(MultiCategoricalProbabilityDistribution, self).__init__()

    def flatparam(self):
        return self.flat

    def mode(self):
        return tf.stack([p.mode() for p in self.categoricals], axis=-1)

    def neglogp(self, x):
        return tf.add_n([p.neglogp(px) for p, px in zip(self.categoricals, tf.unstack(x, axis=-1))])

    def kl(self, other):
        return tf.add_n([p.kl(q) for p, q in zip(self.categoricals, other.categoricals)])

    def entropy(self):
        return tf.add_n([p.entropy() for p in self.categoricals])

    def sample(self):
        return tf.stack([p.sample() for p in self.categoricals], axis=-1)

    @classmethod
    def fromflat(cls, flat):
        """
        Create an instance of this from new logits values

        :param flat: ([float]) the multi categorical logits input
        :return: (ProbabilityDistribution) the instance from the given multi categorical input
        """
        raise NotImplementedError


class DiagGaussianProbabilityDistribution(ProbabilityDistribution):
    def __init__(self, flat):
        """
        Probability distributions from multivariate Gaussian input

        :param flat: ([float]) the multivariate Gaussian input data
        """
        self.flat = flat
        mean, logstd = tf.split(axis=len(flat.shape) - 1, num_or_size_splits=2, value=flat)
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)
        super(DiagGaussianProbabilityDistribution, self).__init__()

    def flatparam(self):
        return self.flat

    def mode(self):
        # Bounds are taken into account outside this class (during training only)
        return self.mean

    def neglogp(self, x):  # TODO: does this reduce one value?? (no, most likely over the actions dims)
        return 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.cast(tf.shape(x)[-1], tf.float32) \
               + tf.reduce_sum(self.logstd, axis=-1)

    def kl(self, other):
        assert isinstance(other, BoundedDiagGaussianProbabilityDistribution)
        return tf.reduce_sum(other.logstd - self.logstd + (tf.square(self.std) + tf.square(self.mean - other.mean)) /
                             (2.0 * tf.square(other.std)) - 0.5, axis=-1)

    def entropy(self):
        return tf.reduce_sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)

    def sample(self):
        # Bounds are taken into acount outside this class (during training only)
        # Otherwise, it changes the distribution and breaks PPO2 for instance
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean), dtype=self.mean.dtype)

    @classmethod
    def fromflat(cls, flat):
        """
        Create an instance of this from new multivariate Gaussian input

        :param flat: ([float]) the multivariate Gaussian input data
        :return: (ProbabilityDistribution) the instance from the given multivariate Gaussian input data
        """
        return cls(flat)


class BoundedDiagGaussianProbabilityDistribution(ProbabilityDistribution):
    def __init__(self, flat):
        """
        Probability distributions from multivariate Gaussian input

        :param flat: ([float]) the multivariate Gaussian input data
        """
        self.flat = flat
        mean, logstd = tf.split(axis=len(flat.shape) - 1, num_or_size_splits=2, value=flat)
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)
        super(BoundedDiagGaussianProbabilityDistribution, self).__init__()

    def flatparam(self):
        return self.flat

    def mode(self):
        # Bounds are taken into account outside this class (during training only)
        return tf.nn.tanh(self.mean)

    def neglogp(self, x):  # TODO: does this reduce one value?? (no, most likely over the actions dims)
        return 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.cast(tf.shape(x)[-1], tf.float32) \
               + tf.reduce_sum(self.logstd, axis=-1) - tf.reduce_sum(tf.log(1 - tf.square(tf.nn.tanh(x))), axis=1)

    def kl(self, other):
        assert isinstance(other, BoundedDiagGaussianProbabilityDistribution)
        return tf.reduce_sum(other.logstd - self.logstd + (tf.square(self.std) + tf.square(self.mean - other.mean)) /
                             (2.0 * tf.square(other.std)) - 0.5, axis=-1)

    def entropy(self):
        return tf.reduce_sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)

    def sample(self):
        # Bounds are taken into acount outside this class (during training only)
        # Otherwise, it changes the distribution and breaks PPO2 for instance
        return tf.nn.tanh(self.mean + self.std * tf.random_normal(tf.shape(self.mean), dtype=self.mean.dtype))

    @classmethod
    def fromflat(cls, flat):
        """
        Create an instance of this from new multivariate Gaussian input

        :param flat: ([float]) the multivariate Gaussian input data
        :return: (ProbabilityDistribution) the instance from the given multivariate Gaussian input data
        """
        return cls(flat)


class BetaProbabilityDistribution(ProbabilityDistribution):
    def __init__(self, flat):
        """
        Probability distributions from beta input
        :param flat: ([float]) the beta input data
        """
        self.flat = flat
        print(flat)
        # as per http://proceedings.mlr.press/v70/chou17a/chou17a.pdf
        self.alpha = 1.0 + tf.layers.dense(flat, flat.shape[1], activation=tf.nn.softplus)
        self.beta  = 1.0 + tf.layers.dense(flat, flat.shape[1], activation=tf.nn.softplus)
        self.dist = tf.distributions.Beta(concentration1=self.alpha, concentration0=self.beta, validate_args=True,
            allow_nan_stats=False)

    def flatparam(self):
        return self.flat

    def mode(self):
        #return tf.stack([p.mode() for p in self.flat])
        return self.dist.mode()

    def neglogp(self, x):
        return tf.reduce_sum(-self.dist.log_prob(x), axis=-1)

    def kl(self, other):
        assert isinstance(other, BetaProbabilityDistribution)
        return self.dist.kl_divergence(other.dist)

    def entropy(self):
        return self.dist.entropy()

    def sample(self):
        return self.dist.sample()

    @classmethod
    def fromflat(cls, flat):
        """
        Create an instance of this from new beta input
        :param flat: ([float]) the beta input data
        :return: (ProbabilityDistribution) the instance from the given beta input data
        """
        return cls(flat)


class BernoulliProbabilityDistribution(ProbabilityDistribution):
    def __init__(self, logits):
        """
        Probability distributions from Bernoulli input

        :param logits: ([float]) the Bernoulli input data
        """
        self.logits = logits
        self.probabilities = tf.sigmoid(logits)
        super(BernoulliProbabilityDistribution, self).__init__()

    def flatparam(self):
        return self.logits

    def mode(self):
        return tf.round(self.probabilities)

    def neglogp(self, x):
        return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
                                                                     labels=tf.cast(x, tf.float32)),
                             axis=-1)

    def kl(self, other):
        return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=other.logits,
                                                                     labels=self.probabilities), axis=-1) - \
               tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
                                                                     labels=self.probabilities), axis=-1)

    def entropy(self):
        return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
                                                                     labels=self.probabilities), axis=-1)

    def sample(self):
        samples_from_uniform = tf.random_uniform(tf.shape(self.probabilities))
        return tf.cast(math_ops.less(samples_from_uniform, self.probabilities), tf.float32)

    @classmethod
    def fromflat(cls, flat):
        """
        Create an instance of this from new Bernoulli input

        :param flat: ([float]) the Bernoulli input data
        :return: (ProbabilityDistribution) the instance from the given Bernoulli input data
        """
        return cls(flat)


class GP0GeneralizedPoissonProbabilityDistribution(ProbabilityDistribution):   # GP-0 (Consul, 1973 and 1989)
    def __init__(self, rate, delta):
        """
        Probability distributions from Bernoulli input

        :param logits: ([float]) the Bernoulli input data
        """
        self.rate = rate
        self.delta = delta
        self.sample_ph = tf.placeholder(tf.float32, shape=rate.shape, name="gpd_sample_ph")
        super(GP0GeneralizedPoissonProbabilityDistribution, self).__init__()

    def flatparam(self):
        return self.rate, self.delta

    def mean(self):
        return self.rate / (1 - self.delta)  # TODO: possibly round and/or cast to integer

    def variance(self):
        return self.mean() / tf.square(1 - self.delta)

    def mode(self):  # mode is the point with the highest pmf, think will just set to mean for now
        return self.mean()

    def pmf(self, x, rate, delta, pmf_x_1=None):
        if x == 0:
            return np.exp(-rate)
        elif pmf_x_1 is None:
            return rate * np.power(rate + delta * x, x - 1) * np.exp(-rate - delta * x) / np.math.factorial(x)
        else:
            x = float(x)
            return np.exp(np.log(pmf_x_1) + (x - 1) * np.log(rate + delta * x) - (x - 2) * np.log(rate + delta * (x - 1)) - np.log(x) - delta)

            N = np.power(rate + delta * x, x - 1)
            D = (np.power(rate + delta * (x - 1), x - 2)) * x * np.exp(delta)
            if not np.any(np.isinf(N)) or np.any(np.isnan(N)):  # TODO: fix, should go through each
                return pmf_x_1 * N / D
            else:
                lp = np.log(pmf_x_1) - (x - 1) * np.log(rate + delta * x) + x * np.log(rate + delta * (x + 1)) - delta - np.log(x + 1)
                return np.exp(lp)

    def neglogp(self, x):  # TODO: can maybe even remove the last term since it doesnt depend on the model parameters
        return tf.reduce_sum(tf.log(self.rate) + (x - 1) * tf.log(self.rate + self.delta * x) - (self.rate + self.delta * x) - tf.lgamma(x + 1), axis=-1)

    def kl(self, other):
        raise NotImplementedError

    def entropy(self):  # assuming GPR has same entropy as Poisson but where var != mean
        var = self.variance()
        return 0.5 * tf.log(2 * np.pi * np.e * var) - 1 / (12 * var) - 1 / (24 * tf.square(var)) - 19 / (360 * tf.math.pow(var, 3))

    def sample(self):
        #samples_from_uniform = tf.random_uniform(tf.shape(self.rate))
        #sample = 1
        #cdf = self.pmf(sample)
        #while samples_from_uniform < cdf:
        #    cdf += self.pmf(sample)
        #return tf.cast(sample, tf.float32)
        return self.sample_ph

    @classmethod
    def fromflat(cls, rate, dispersion):
        """
        Create an instance of this from new Bernoulli input

        :param flat: ([float]) the Bernoulli input data
        :return: (ProbabilityDistribution) the instance from the given Bernoulli input data
        """
        return cls(rate, dispersion)


class GeneralizedPoissonProbabilityDistribution(ProbabilityDistribution):   # GP-2 (Famoye, 1993/ Wang 1997)
    def __init__(self, rate, alpha, min_val=1.0, max_val=40.0):
        """
        Probability distributions from Bernoulli input

        :param logits: ([float]) the Bernoulli input data
        """
        self.rate = rate
        self.alpha = alpha
        self.min_val = min_val
        self.max_val = max_val
        super(GeneralizedPoissonProbabilityDistribution, self).__init__()

    def flatparam(self):
        return self.rate, self.alpha

    def mean(self):
        return self.rate

    def variance(self):
        return self.rate * tf.square(1 + self.alpha * self.rate)

    def mode(self):  # mode is the point with the highest pmf, think will just set to mean for now
        return tf.ceil(self.mean())

    def neglogp(self, x):  # TODO: can maybe even remove the last term since it doesnt depend on the model parameters
        return -tf.reduce_sum(x * tf.log(self.rate / (1 + self.alpha * self.rate)) + (x - 1) * tf.log(1 + self.alpha * x) - (self.rate * (1 + self.alpha * x) / (1 + self.alpha * self.rate)) - tf.lgamma(x + 1), axis=-1)

    def kl(self, other):
        raise NotImplementedError

    def entropy(self):  # assuming GPR has same entropy as Poisson but where var != mean
        var = self.variance()
        return 0.5 * tf.log(2 * np.pi * np.e * var) - 1 / (12 * var) - 1 / (24 * tf.square(var)) - 19 / (360 * tf.math.pow(var, 3))

    def sample(self):  # Normal approximation for sampling
        samples_from_normal = tf.random_normal(tf.shape(self.rate))
        return tf.clip_by_value(tf.floor(self.mean() + tf.sqrt(self.variance()) * samples_from_normal + 0.5), self.min_val, self.max_val)  # TODO: not sure if correct to limit to max. No, it is correct because the distribution is only defined for 1 + alpha * sampled > 0.

    @classmethod
    def fromflat(cls, rate, dispersion):
        """
        Create an instance of this from new Bernoulli input

        :param flat: ([float]) the Bernoulli input data
        :return: (ProbabilityDistribution) the instance from the given Bernoulli input data
        """
        return cls(rate, dispersion)


class RLMPCProbabilityDistribution(ProbabilityDistribution):
    def __init__(self, b_logits, horizon_rate, horizon_alpha, g_flat, max_horizon=50.0, min_horizon=1.0):
        """
        Probability distributions from Mixed Distribution of Bernoulli and Gaussian input

        :param logits: ([float]) the Bernoulli input data
        """
        self.b_logits = b_logits
        self.b_probabilities = tf.sigmoid(b_logits)
        self.g_flat = g_flat

        self.horizon_gpd = GeneralizedPoissonProbabilityDistribution(horizon_rate, horizon_alpha, min_val=min_horizon, max_val=max_horizon)

        mean, logstd = tf.split(axis=len(g_flat.shape) - 1, num_or_size_splits=2, value=g_flat)
        self.g_mean = mean  # Need to modify mode and neglogp og Gaussian (by multiplying by et) so dont use Gaussian class
        self.g_logstd = logstd
        self.g_std = tf.exp(logstd)

        self.include_horizon_neglogp = True

        super(RLMPCProbabilityDistribution, self).__init__()

    def flatparam(self):
        return self.b_logits, self.g_flat, self.nb_logits, self.nb_counts

    def mode(self):
        #et = tf.round(self.b_probabilities)
        samples_from_uniform = tf.random_uniform(tf.shape(self.b_probabilities))
        et = tf.cast(math_ops.less(samples_from_uniform, self.b_probabilities), tf.float32)
        return tf.concat([et, self.horizon_gpd.mode(), (1 - et) * self.g_mean], axis=1)

    def neglogp(self, x):  # TODO: should it be positive/negative (or both?).
        et, horizon, u = x[:, 0], tf.expand_dims(x[:, 1], 1), x[:, 2:]
        self.etneglogp = etneglogp = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.b_logits,
                                                                          labels=tf.expand_dims(et, 1), name="etneglogp"),
                             axis=-1)

        self.g_neglogp = g_neglogp = 0.5 * tf.reduce_sum(tf.square((u - self.g_mean * (1 - tf.expand_dims(et, 1))) / self.g_std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.cast(tf.shape(u)[-1], tf.float32) \
               + tf.reduce_sum(self.g_logstd, axis=-1)

        if self.include_horizon_neglogp:
            self.horizonneglogp = horizonneglogp = self.horizon_gpd.neglogp(horizon)  # TODO: should it be zero when not recomputing? And should it simply be added?

            return etneglogp + horizonneglogp + g_neglogp
        else:
            return etneglogp + g_neglogp

    def kl(self, other):
        raise NotImplementedError
        return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=other.logits,
                                                                     labels=self.probabilities), axis=-1) - \
               tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
                                                                     labels=self.probabilities), axis=-1)

    def entropy(self):  # TODO: might be overestimating (i.e. counting mutual information between distributions several times). Can it be negative?
        et_ent = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.b_logits,
                                                                       labels=self.b_probabilities), axis=-1)
        #et = tf.round(self.b_probabilities)

        g_ent = tf.reduce_sum(self.g_logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)# * (1 - et)
        if self.include_horizon_neglogp:
            horizon_ent = self.horizon_gpd.entropy()  # * et
            return et_ent + horizon_ent + g_ent
        else:
            return et_ent + g_ent

    def sample(self):
        samples_from_uniform = tf.random_uniform(tf.shape(self.b_probabilities))
        et = tf.cast(math_ops.less(samples_from_uniform, self.b_probabilities), tf.float32)
        horizon = self.horizon_gpd.sample()
        u = self.g_mean * (1 - et) + self.g_std * tf.random_normal(tf.shape(self.g_mean),
                                                       dtype=self.g_mean.dtype)
        return tf.concat([et, horizon, u], axis=-1)

    @classmethod
    def fromflat(cls, flat):
        """
        Create an instance of this from new Bernoulli input

        :param flat: ([float]) the Bernoulli input data
        :return: (ProbabilityDistribution) the instance from the given Bernoulli input data
        """
        raise NotImplementedError
        return cls(flat)



def make_proba_dist_type(ac_space, dist_type=None):
    """
    return an instance of ProbabilityDistributionType for the correct type of action space

    :param ac_space: (Gym Space) the input action space
    :return: (ProbabilityDistributionType) the appropriate instance of a ProbabilityDistributionType
    """
    if isinstance(ac_space, spaces.Box):
        assert len(ac_space.shape) == 1, "Error: the action space must be a vector"
        if dist_type is None:
            return DiagGaussianProbabilityDistributionType(ac_space.shape[0])
        else:
            return dist_type(ac_space.shape[0])
    elif isinstance(ac_space, spaces.Discrete):
        if dist_type is None:
            return CategoricalProbabilityDistributionType(ac_space.n)
        else:
            return dist_type(ac_space.n)
    elif isinstance(ac_space, spaces.MultiDiscrete):
        if dist_type is None:
            return MultiCategoricalProbabilityDistributionType(ac_space.nvec)
        else:
            return dist_type(ac_space.nvec)
    elif isinstance(ac_space, spaces.MultiBinary):
        if dist_type is None:
            return BernoulliProbabilityDistributionType(ac_space.n)
        else:
            return dist_type(ac_space.n)
    else:
        raise NotImplementedError("Error: probability distribution, not implemented for action space of type {}."
                                  .format(type(ac_space)) +
                                  " Must be of type Gym Spaces: Box, Discrete, MultiDiscrete or MultiBinary.")


def shape_el(tensor, index):
    """
    get the shape of a TensorFlow Tensor element

    :param tensor: (TensorFlow Tensor) the input tensor
    :param index: (int) the element
    :return: ([int]) the shape
    """
    maybe = tensor.get_shape()[index]
    if maybe is not None:
        return maybe
    else:
        return tf.shape(tensor)[index]
