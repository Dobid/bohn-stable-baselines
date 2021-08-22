import time

import gym
import numpy as np
import tensorflow as tf

from stable_baselines import logger
from stable_baselines.common import explained_variance, ActorCriticRLModel, tf_util, SetVerbosity, TensorboardWriter
from stable_baselines.common.runners import AbstractEnvRunner
from stable_baselines.common.policies import ActorCriticPolicy, RecurrentActorCriticPolicy
from stable_baselines.common.schedules import get_schedule_fn
from stable_baselines.common.tf_util import total_episode_reward_logger
from stable_baselines.common.math_util import safe_mean
from collections import deque
class KLDivergenceException(Exception):
    pass

class PPO2(ActorCriticRLModel):
    """
    Proximal Policy Optimization algorithm (GPU version).
    Paper: https://arxiv.org/abs/1707.06347

    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) Discount factor
    :param n_steps: (int) The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param ent_coef: (float) Entropy coefficient for the loss calculation
    :param learning_rate: (float or callable) The learning rate, it can be a function
    :param vf_coef: (float) Value function coefficient for the loss calculation
    :param max_grad_norm: (float) The maximum value for the gradient clipping
    :param lam: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param nminibatches: (int) Number of training minibatches per update. For recurrent policies,
        the number of environments run in parallel should be a multiple of nminibatches.
    :param noptepochs: (int) Number of epoch when optimizing the surrogate
    :param cliprange: (float or callable) Clipping parameter, it can be a function
    :param cliprange_vf: (float or callable) Clipping parameter for the value function, it can be a function.
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        then `cliprange` (that is used for the policy) will be used.
        IMPORTANT: this clipping depends on the reward scaling.
        To deactivate value function clipping (and recover the original PPO implementation),
        you have to pass a negative value (e.g. -1).
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    """
    def __init__(self, policy, env, gamma=0.99, n_steps=128, ent_coef=0.01, learning_rate=2.5e-4, vf_coef=0.5,
                 max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4, cliprange=0.2, target_kl=None, cliprange_vf=None,
                 verbose=0, spatial_CAPS_coef=None, temporal_CAPS_coef=None, CAPS_idxs=None, frame_skip=None, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, time_aware=False,
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None):

        self.learning_rate = learning_rate
        self.cliprange = cliprange
        self.cliprange_vf = cliprange_vf
        self.n_steps = n_steps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.lam = lam
        self.nminibatches = nminibatches
        self.noptepochs = noptepochs
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log
        self.target_kl = target_kl

        self.spatial_CAPS_coef = spatial_CAPS_coef
        self.temporal_CAPS_coef = temporal_CAPS_coef
        self.frame_skip = frame_skip
        
        self.time_aware = time_aware
        self.action_ph = None
        self.advs_ph = None
        self.rewards_ph = None
        self.old_neglog_pac_ph = None
        self.old_vpred_ph = None
        self.learning_rate_ph = None
        self.ent_coef_ph = None
        self.clip_range_ph = None
        self.entropy = None
        self.vf_loss = None
        self.pg_loss = None
        self.approxkl = None
        self.clipfrac = None
        self._train = None
        self.loss_names = None
        self.train_model = None
        self.act_model = None
        self.value = None
        self.n_batch = None
        self.summary = None

        super().__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=True,
                         _init_setup_model=_init_setup_model, policy_kwargs=policy_kwargs,
                         seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)

        self.active_sampling = False

        if _init_setup_model:
            self.setup_model()

    def _make_runner(self):
        return Runner(env=self.env, model=self, n_steps=self.n_steps,
                      gamma=self.gamma, lam=self.lam, time_aware=self.time_aware, frame_skip=self.frame_skip)

    def _get_pretrain_placeholders(self, get_vf=False):
        policy = self.act_model
        if isinstance(self.action_space, gym.spaces.Discrete):
            return policy.obs_ph, self.action_ph, policy.policy
        if get_vf:
            return policy.obs_ph, self.action_ph, policy.deterministic_action, self.train_model.obs_ph, self.rewards_ph, self.train_model._value_flat
        else:
            return policy.obs_ph, self.action_ph, policy.deterministic_action

    def setup_model(self):
        with SetVerbosity(self.verbose):

            assert issubclass(self.policy, ActorCriticPolicy), "Error: the input policy for the PPO2 model must be " \
                                                               "an instance of common.policies.ActorCriticPolicy."

            self.n_batch = self.n_envs * self.n_steps

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(self.seed)
                self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)

                n_batch_step = None
                n_batch_train = None
                if issubclass(self.policy, RecurrentActorCriticPolicy):
                    assert self.n_envs % self.nminibatches == 0, "For recurrent policies, "\
                        "the number of environments run in parallel should be a multiple of nminibatches."
                    n_batch_step = self.n_envs
                    n_batch_train = self.n_batch // self.nminibatches

                act_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                        n_batch_step, reuse=False, **self.policy_kwargs)
                with tf.variable_scope("train_model", reuse=True,
                                       custom_getter=tf_util.outer_scope_getter("train_model")):
                    train_model = self.policy(self.sess, self.observation_space, self.action_space,
                                              self.n_envs // self.nminibatches, self.n_steps, n_batch_train,
                                              reuse=True, **self.policy_kwargs)

                    if self.frame_skip:
                        self.train_model_v = self.policy(self.sess, self.observation_space, self.action_space,
                                              self.n_envs // self.nminibatches, self.n_steps, n_batch_train,
                                              reuse=True, **self.policy_kwargs)

                    if self.spatial_CAPS_coef is not None:
                        self.train_model_scasp = self.policy(self.sess, self.observation_space, self.action_space,
                                                  self.n_envs // self.nminibatches, self.n_steps, n_batch_train,
                                                  reuse=True, **self.policy_kwargs)
                    if self.temporal_CAPS_coef is not None:
                        self.train_model_tcasp = self.policy(self.sess, self.observation_space, self.action_space,
                                               self.n_envs // self.nminibatches, self.n_steps, n_batch_train,
                                               reuse=True, **self.policy_kwargs)
                with tf.variable_scope("loss", reuse=False):
                    self.action_ph = train_model.pdtype.sample_placeholder([None], name="action_ph")
                    self.advs_ph = tf.placeholder(tf.float32, [None], name="advs_ph")
                    self.rewards_ph = tf.placeholder(tf.float32, [None], name="rewards_ph")
                    self.old_neglog_pac_ph = tf.placeholder(tf.float32, [None], name="old_neglog_pac_ph")
                    self.old_vpred_ph = tf.placeholder(tf.float32, [None], name="old_vpred_ph")
                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")
                    self.ent_coef_ph = tf.placeholder(tf.float32, [], name="ent_coef_ph")
                    self.clip_range_ph = tf.placeholder(tf.float32, [], name="clip_range_ph")

                    neglogpac = train_model.proba_distribution.neglogp(self.action_ph)
                    self.entropy = tf.reduce_mean(train_model.proba_distribution.entropy())

                    if self.frame_skip:
                        vpred = self.train_model_v.value_flat
                    else:
                        vpred = train_model.value_flat

                    # Value function clipping: not present in the original PPO
                    if self.cliprange_vf is None:
                        # Default behavior (legacy from OpenAI baselines):
                        # use the same clipping as for the policy
                        self.clip_range_vf_ph = self.clip_range_ph
                        self.cliprange_vf = self.cliprange
                    elif isinstance(self.cliprange_vf, (float, int)) and self.cliprange_vf < 0:
                        # Original PPO implementation: no value function clipping
                        self.clip_range_vf_ph = None
                    else:
                        # Last possible behavior: clipping range
                        # specific to the value function
                        self.clip_range_vf_ph = tf.placeholder(tf.float32, [], name="clip_range_vf_ph")

                    if self.clip_range_vf_ph is None:
                        # No clipping
                        if self.frame_skip:
                            vpred_clipped = self.train_model_v.value_flat
                        else:
                            vpred_clipped = train_model.value_flat
                    else:
                        # Clip the different between old and new value
                        # NOTE: this depends on the reward scaling
                        vpred_clipped = self.old_vpred_ph + \
                            tf.clip_by_value(vpred - self.old_vpred_ph,
                                             - self.clip_range_vf_ph, self.clip_range_vf_ph)

                    vf_losses1 = tf.square(vpred - self.rewards_ph)
                    vf_losses2 = tf.square(vpred_clipped - self.rewards_ph)
                    self.vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

                    ratio = tf.exp(self.old_neglog_pac_ph - neglogpac)
                    pg_losses = -self.advs_ph * ratio
                    pg_losses2 = -self.advs_ph * tf.clip_by_value(ratio, 1.0 - self.clip_range_ph, 1.0 +
                                                                  self.clip_range_ph)
                    self.pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
                    self.approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - self.old_neglog_pac_ph))
                    self.clipfrac = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.0),
                                                                      self.clip_range_ph), tf.float32))

                    loss = self.pg_loss - self.entropy * self.ent_coef_ph + self.vf_loss * self.vf_coef

                    # CAPS
                    if self.spatial_CAPS_coef is not None:
                        spatial_CAPS = self.spatial_CAPS_coef * tf.nn.l2_loss(train_model.policy - self.train_model_scasp.policy, name="spatial_CAPS_loss")
                        loss += spatial_CAPS
                    if self.temporal_CAPS_coef is not None:
                        temporal_CAPS = self.temporal_CAPS_coef * tf.nn.l2_loss(train_model.policy - self.train_model_tcasp.policy, name="temporal_CAPS_loss")
                        loss += temporal_CAPS
                    #loss = self.pg_loss + self.vf_loss * self.vf_coef

                    tf.summary.scalar('entropy_loss', self.entropy)
                    tf.summary.scalar('policy_gradient_loss', self.pg_loss)
                    tf.summary.scalar('value_function_loss', self.vf_loss)
                    tf.summary.scalar('approximate_kullback-leibler', self.approxkl)
                    tf.summary.scalar('clip_factor', self.clipfrac)
                    tf.summary.scalar('loss', loss)

                    with tf.variable_scope('model'):
                        self.params = tf.trainable_variables()
                        if self.full_tensorboard_log:
                            for var in self.params:
                                tf.summary.histogram(var.name, var)
                    grads = tf.gradients(loss, self.params)
                    if self.max_grad_norm is not None:
                        grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
                    grads = list(zip(grads, self.params))
                    self.grads = grads

                trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph, epsilon=1e-5, beta1=0.9)

                if hasattr(train_model, "lqr_output") and getattr(train_model, "lqr_lr_multiplier", None) is not None:
                    trainer_lqr = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph * train_model.lqr_lr_multipler, epsilon=1e-5, beta1=0.9)
                    train_op1 = trainer.apply_gradients([g for g in grads if "LQR" not in g[1].name])
                    train_op2 = trainer_lqr.apply_gradients([g for g in grads if "LQR" in g[1].name])
                    self._train = tf.group(train_op1, train_op2)
                else:
                    self._train = trainer.apply_gradients(grads)

                if hasattr(train_model, "lqr_output"):
                    with tf.variable_scope("model", reuse=True):
                        train_model.lqr_weights = tf.get_variable("LQR_weights")

                self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

                with tf.variable_scope("input_info", reuse=False):
                    tf.summary.scalar('discounted_rewards', tf.reduce_mean(self.rewards_ph))
                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))
                    tf.summary.scalar('advantage', tf.reduce_mean(self.advs_ph))
                    tf.summary.scalar('clip_range', tf.reduce_mean(self.clip_range_ph))
                    if hasattr(train_model.proba_distribution, "probabilities"):
                        tf.summary.scalar("prob1", tf.reduce_mean(train_model.proba_distribution.probabilities))
                        tf.summary.scalar("prob1var", tf.math.reduce_std(train_model.proba_distribution.probabilities))
                    if hasattr(train_model.proba_distribution, "rate"):
                        tf.summary.scalar("rate", tf.reduce_mean(train_model.proba_distribution.rate))
                        tf.summary.scalar("rate_var", tf.math.reduce_std(train_model.proba_distribution.rate))
                    if hasattr(train_model.proba_distribution, "horizon_gpd"):
                        tf.summary.scalar("horizon_mean", tf.reduce_mean(train_model.proba_distribution.horizon_gpd.rate))
                        tf.summary.scalar("horizon_var", tf.math.reduce_std(train_model.proba_distribution.horizon_gpd.rate))
                        tf.summary.scalar("mpc_compute", tf.reduce_mean(train_model.proba_distribution.b_probabilities))
                        tf.summary.scalar("mpc_compute_var", tf.math.reduce_std(train_model.proba_distribution.b_probabilities))
                    if self.clip_range_vf_ph is not None:
                        tf.summary.scalar('clip_range_vf', tf.reduce_mean(self.clip_range_vf_ph))
                    tf.summary.scalar("explained_variance", tf.reduce_mean(1 - tf.math.reduce_variance(vpred - self.rewards_ph) / tf.math.reduce_variance(self.rewards_ph)))

                    if self.spatial_CAPS_coef is not None:
                        tf.summary.scalar("spatial_CAPS_loss", spatial_CAPS)
                    if self.temporal_CAPS_coef is not None:
                        tf.summary.scalar("temporal_CAPS_loss", temporal_CAPS)

                    tf.summary.scalar('old_neglog_action_probability', tf.reduce_mean(self.old_neglog_pac_ph))
                    tf.summary.scalar('old_value_pred', tf.reduce_mean(self.old_vpred_ph))

                    if self.full_tensorboard_log:
                        tf.summary.histogram('discounted_rewards', self.rewards_ph)
                        tf.summary.histogram('learning_rate', self.learning_rate_ph)
                        tf.summary.histogram('advantage', self.advs_ph)
                        tf.summary.histogram('clip_range', self.clip_range_ph)
                        tf.summary.histogram('old_neglog_action_probability', self.old_neglog_pac_ph)
                        tf.summary.histogram('old_value_pred', self.old_vpred_ph)
                        if tf_util.is_image(self.observation_space):
                            tf.summary.image('observation', train_model.obs_ph)
                        else:
                            tf.summary.histogram('observation', train_model.obs_ph)

                if self.full_tensorboard_log:
                    for var in tf_util.get_trainable_vars("model"):
                        tf.summary.histogram("weights/" + var.name, var)
                    for var in grads:
                        if var[0] is not None:
                            tf.summary.histogram("grads/" + var[1].name, var[0])

                self.train_model = train_model
                self.act_model = act_model
                self.step = act_model.step
                self.proba_step = act_model.proba_step
                self.value = act_model.value
                self.initial_state = act_model.initial_state
                tf.global_variables_initializer().run(session=self.sess)  # pylint: disable=E1101
                
                self.summary = tf.summary.merge_all()

    def _train_step(self, learning_rate, cliprange, ent_coef, obs, returns, masks, actions, values, neglogpacs, update,
                    writer, states=None, cliprange_vf=None):
        """
        Training of PPO2 Algorithm

        :param learning_rate: (float) learning rate
        :param cliprange: (float) Clipping factor
        :param obs: (np.ndarray) The current observation of the environment
        :param returns: (np.ndarray) the rewards
        :param masks: (np.ndarray) The last masks for done episodes (used in recurent policies)
        :param actions: (np.ndarray) the actions
        :param values: (np.ndarray) the values
        :param neglogpacs: (np.ndarray) Negative Log-likelihood probability of Actions
        :param update: (int) the current step iteration
        :param writer: (TensorFlow Summary.writer) the writer for tensorboard
        :param states: (np.ndarray) For recurrent policies, the internal state of the recurrent model
        :return: policy gradient loss, value function loss, policy entropy,
                approximation of kl divergence, updated clipping range, training update operation
        :param cliprange_vf: (float) Clipping factor for the value function
        """
        advs = returns - values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        td_map = {self.train_model.obs_ph: obs, self.action_ph: actions,
                  self.advs_ph: advs, self.rewards_ph: returns,
                  self.learning_rate_ph: learning_rate, self.clip_range_ph: cliprange,
                  self.old_neglog_pac_ph: neglogpacs, self.old_vpred_ph: values, self.ent_coef_ph: ent_coef}
        if states is not None:
            td_map[self.train_model.states_ph] = states
            td_map[self.train_model.dones_ph] = masks

        if cliprange_vf is not None and cliprange_vf >= 0:
            td_map[self.clip_range_vf_ph] = cliprange_vf


        if hasattr(self.train_model, "lqr_K_ph"):
            if self.train_model.time_varying:
                td_map[self.train_model.lqr_K_ph] = self.train_model.lqr_Ks
                td_map[self.train_model.lqr_K_grad_ph] = self.train_model.lqr_K_grads
            else:
                td_map[self.train_model.lqr_K_ph] = self.train_model.lqr.K_num
                td_map[self.train_model.lqr_K_grad_ph] = np.transpose(self.train_model.lqr._grad_K(), (2, 0, 1))


        if self.spatial_CAPS_coef is not None:
            td_map[self.train_model_scasp.obs_ph] = np.random.normal(obs, 0.01)  # TODO: fix for LQR

        if self.temporal_CAPS_coef is not None:
            tp1_mbinds = self.mbinds
            tp1_mbinds[(self.mbinds < self.n_batch - 1) & (self.mbinds % len(self.mbinds) != 0)] += 1
            td_map[self.train_model_tcasp.obs_ph] = self.train_model_tcasp.obss[tp1_mbinds]

        if self.frame_skip is not None:
            td_map[self.rewards_ph] = self.train_model.rewards_n
            td_map[self.old_vpred_ph] = self.train_model.values
            td_map[self.train_model_v.obs_ph] = self.train_model.vf_obs

        if states is None:
            update_fac = max(self.n_batch // self.nminibatches // self.noptepochs, 1)
        else:
            update_fac = max(self.n_batch // self.nminibatches // self.noptepochs // self.n_steps, 1)


        #grads = self.sess.run([v[0] for v in self.grads if v[0] is not None], td_map)
        #for g_i, g in enumerate(grads):
        #    if np.any(np.isnan(g)):
        #        print("Grad is nan for: {}".format(self.grads[g_i][1].name))

        if writer is not None:
            # run loss backprop with summary, but once every 10 runs save the metadata (memory, compute time, ...)
            if self.full_tensorboard_log and (1 + update) % 10 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                    [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train],
                    td_map, options=run_options, run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata, 'step%d' % (update * update_fac))
            else:
                summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                    [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train],
                    td_map)
            writer.add_summary(summary, (update * update_fac))
        else:
            policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                [self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train], td_map)

        return policy_loss, value_loss, policy_entropy, approxkl, clipfrac

    def learn(self, total_timesteps, callback=None, log_interval=1, tb_log_name="PPO2",
              reset_num_timesteps=True):
        # Transform to callable if needed
        self.learning_rate = get_schedule_fn(self.learning_rate)
        self.cliprange = get_schedule_fn(self.cliprange)
        self.ent_coef = get_schedule_fn(self.ent_coef)
        cliprange_vf = get_schedule_fn(self.cliprange_vf)

        samples = deque(maxlen=5 * self.env.num_envs * 10)

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        callback = self._init_callback(callback)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn()

            t_first_start = time.time()
            n_updates = total_timesteps // self.n_batch

            callback.on_training_start(locals(), globals())

            for update in range(1, n_updates + 1):
                assert self.n_batch % self.nminibatches == 0, ("The number of minibatches (`nminibatches`) "
                                                               "is not a factor of the total number of samples "
                                                               "collected per rollout (`n_batch`), "
                                                               "some samples won't be used."
                                                               )
                batch_size = self.n_batch // self.nminibatches
                t_start = time.time()
                frac = 1.0 - (update - 1.0) / n_updates
                lr_now = self.learning_rate(frac)
                ent_coef_now = self.ent_coef(frac)
                cliprange_now = self.cliprange(frac)
                cliprange_vf_now = cliprange_vf(frac)

                callback.on_rollout_start()
                # true_reward is the reward without discount
                rollout = self.runner.run(callback)
                # Unpack
                obs, returns, masks, actions, values, neglogpacs, states, ep_infos, true_reward = rollout
                callback.on_rollout_end()

                # Early stopping due to the callback
                if not self.runner.continue_training:
                    break


                if getattr(self.train_model, "time_varying", False):
                    self.train_model.lqr_As = np.array(self.train_model.lqr_As)
                    self.train_model.lqr_Bs = np.array(self.train_model.lqr_Bs)
                    self.train_model.lqr_system_idxs = np.array(self.train_model.lqr_system_idxs)

                    self.train_model.old_Q = np.copy(self.train_model.lqr.get_numeric_value("Q"))
                    self.train_model.old_R = np.copy(self.train_model.lqr.get_numeric_value("R"))

                if getattr(self, "action_hist_sum", None) is not None:
                    action_hist_sum = self.sess.run(self.action_hist_sum, {self.action_ph: actions})
                    writer.add_summary(action_hist_sum, self.num_timesteps)

                if self.temporal_CAPS_coef is not None:
                    self.train_model_tcasp.obss = obs


                self.ep_info_buf.extend(ep_infos)
                mb_loss_vals = []
                if states is None:  # nonrecurrent version
                    update_fac = max(self.n_batch // self.nminibatches // self.noptepochs, 1)
                    inds = np.arange(self.n_batch)
                    try:
                        for epoch_num in range(self.noptepochs):
                            np.random.shuffle(inds)
                            if epoch_num > 0:  # recompute advantages on every epoch
                                values = self.act_model.value(obs)
                                last_values = self.act_model.value(self.runner.obs)
                                new_advs = swap_and_flatten(self.runner.get_advantages(true_reward, values, masks, last_values))
                                returns = new_advs + values
                            for start in range(0, self.n_batch, batch_size):
                                timestep = self.num_timesteps // update_fac + ((epoch_num *
                                                                                self.n_batch + start) // batch_size)
                                end = start + batch_size
                                mbinds = inds[start:end]
                                self.mbinds = mbinds
                                slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                                if hasattr(self.train_model, "mpc_action_ph") and False:
                                    self.train_model.mpc_actions_mb = self.train_model.mpc_actions[mbinds]
                                if getattr(self.train_model, "time_varying", False):
                                    unique_system_idxs, unique_inverse_idxs = np.unique(self.train_model.lqr_system_idxs[mbinds], return_inverse=True)  # TODO: the correct Ks and stuff are not extracted
                                    self.train_model.lqr.set_weights(self.sess.run(self.train_model.lqr_weights), compute_lqr=False)
                                    self.train_model.lqr.set_numeric_value({"A": self.train_model.lqr_As[unique_system_idxs], "B": self.train_model.lqr_Bs[unique_system_idxs]})
                                    grad_Ks = self.train_model.lqr._grad_K()
                                    lqr_Ks = self.train_model.lqr.get_numeric_value("K")
                                    t_idxs = obs[mbinds, self.train_model.lqr_k_idx].astype(np.int32)
                                    t_idxs = [min(len(lqr_Ks[idx]) - 1, t_idxs[i]) for i, idx in enumerate(unique_inverse_idxs)]
                                    if isinstance(lqr_Ks, list):
                                        self.train_model.lqr_Ks = np.array([lqr_Ks[idx][t_idxs[i]] for i, idx in enumerate(unique_inverse_idxs)])   # TODO: dont know if this works
                                        self.train_model.lqr_K_grads = np.transpose(np.array([grad_Ks[idx][t_idxs[i]] for i, idx in enumerate(unique_inverse_idxs)]), axes=(0, 3, 1, 2))
                                    else:
                                        self.train_model.lqr_Ks = lqr_Ks[unique_inverse_idxs, t_idxs, ...]
                                        self.train_model.lqr_K_grads = np.transpose(grad_Ks[unique_inverse_idxs, t_idxs, ...], axes=(0, 3, 1, 2))
                                    #self.sess.run([self.train_model.lqr_K_var.assign(Ks), self.train_model.lqr_K_grad_var.assign(grad_Ks)])

                                mb_loss_vals.append(self._train_step(lr_now, cliprange_now, ent_coef_now, *slices, writer=writer,
                                                                     update=timestep, cliprange_vf=cliprange_vf_now))
                                if hasattr(self.train_model, "lqr_output") and not getattr(self.train_model, "time_varying", False):
                                    self.train_model.lqr.set_weights(self.sess.run(self.train_model.lqr_weights))
                                    #K_num, K_grad = self.train_model.lqr.K_num, self.train_model.lqr._grad_K()
                                    #self.sess.run([self.train_model.lqr_K_var.assign(K_num), self.train_model.lqr_K_grad_var.assign(K_grad)])
                                if self.target_kl is not None and mb_loss_vals[-1][-2] > self.target_kl:
                                    raise KLDivergenceException
                    except KLDivergenceException:
                        print("Update stopped early (it {}/{}) because approx kl divergence is bigger than target kl {}/{}".format(epoch_num, self.noptepochs, mb_loss_vals[-1][-2], self.target_kl))

                            #if hasattr(self.train_model, "mpc_action_ph"): # TODO: update here if std becomes trainable
                else:  # recurrent version
                    update_fac = max(self.n_batch // self.nminibatches // self.noptepochs // self.n_steps, 1)
                    assert self.n_envs % self.nminibatches == 0
                    env_indices = np.arange(self.n_envs)
                    flat_indices = np.arange(self.n_envs * self.n_steps).reshape(self.n_envs, self.n_steps)
                    envs_per_batch = batch_size // self.n_steps
                    for epoch_num in range(self.noptepochs):
                        np.random.shuffle(env_indices)
                        for start in range(0, self.n_envs, envs_per_batch):
                            timestep = self.num_timesteps // update_fac + ((epoch_num *
                                                                            self.n_envs + start) // envs_per_batch)
                            end = start + envs_per_batch
                            mb_env_inds = env_indices[start:end]
                            mb_flat_inds = flat_indices[mb_env_inds].ravel()
                            slices = (arr[mb_flat_inds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                            mb_states = states[mb_env_inds]
                            mb_loss_vals.append(self._train_step(lr_now, cliprange_now, ent_coef_now, *slices, update=timestep,
                                                                 writer=writer, states=mb_states,
                                                                 cliprange_vf=cliprange_vf_now))
                if hasattr(self.train_model, "lqr_output"):
                    self.act_model.lqr.set_weights(self.sess.run(self.train_model.lqr_weights))
                    #K_num, K_grad = self.act_model.lqr.K_num, self.act_model.lqr._grad_K()
                    #self.sess.run([self.act_model.lqr_K.assign(K_num), self.act_model.lqr_K_grad.assign(K_grad)])
                    if getattr(self.train_model, "time_varying", False):
                        self.train_model.lqr_As, self.train_model.lqr_Bs = list(self.act_model.lqr.get_numeric_value("A")), list(self.act_model.lqr.get_numeric_value("B"))
                        if self.n_envs == 1:
                            self.train_model.lqr_As, self.train_model.lqr_Bs = [self.train_model.lqr_As], [self.train_model.lqr_Bs]
                        self.train_model.lqr_system_idxs = [[i for i in range(self.n_envs)]]

                    Q = self.train_model.lqr.get_numeric_value("Q")
                    R = self.train_model.lqr.get_numeric_value("R")
                    if not getattr(self.train_model, "time_varying", False):
                        K = self.train_model.lqr.get_numeric_value("K")
                        data = zip(["Q", "R", "K"], [Q, R, K])
                    else:
                        data = zip(["Q", "R"], [Q, R])

                    for k, v in data:
                        v_flat = v.flatten()
                        for v_i in range(v_flat.shape[0]):
                            if len(v.shape) == 1:
                                r, c = 1, ""
                            else:
                                r, c = v_i // v.shape[1] + 1, v_i % v.shape[1] + 1
                            summary = tf.Summary.Value(tag="LQR/{}_{}{}".format(k, r, c), simple_value=v_flat[v_i])
                            writer.add_summary(tf.Summary(value=[summary]), self.num_timesteps)

                    if hasattr(self.train_model, "mpc_action_ph") and False:
                        self.env.env_method("update_lqr", Q=Q, R=R)
                    if False:
                        for param in self.params:
                            if param.name == "model/pi/et_logstd:0":
                               std = np.squeeze(np.exp(self.sess.run(param)), axis=0)
                               self.env.env_method("set_action_noise_properties", [{"scale": std[i]} for i in range(std.shape[0])])
                               break

                loss_vals = np.mean(mb_loss_vals, axis=0)
                t_now = time.time()
                fps = int(self.n_batch / (t_now - t_start))

                if writer is not None:
                    total_episode_reward_logger(self.episode_reward,
                                                true_reward.reshape((self.n_envs, self.n_steps)),
                                                masks.reshape((self.n_envs, self.n_steps)),
                                                writer, self.num_timesteps)

                if self.verbose >= 1 and (update % log_interval == 0 or update == 1):
                    explained_var = explained_variance(values, returns)
                    logger.logkv("serial_timesteps", update * self.n_steps)
                    logger.logkv("n_updates", update)
                    logger.logkv("total_timesteps", self.num_timesteps)
                    logger.logkv("fps", fps)
                    logger.logkv("explained_variance", float(explained_var))
                    if len(self.ep_info_buf) > 0 and len(self.ep_info_buf[0]) > 0:
                        logger.logkv('ep_reward_mean', safe_mean([ep_info['r'] for ep_info in self.ep_info_buf]))
                        logger.logkv('ep_len_mean', safe_mean([ep_info['l'] for ep_info in self.ep_info_buf]))
                    logger.logkv('time_elapsed', t_start - t_first_start)
                    for (loss_val, loss_name) in zip(loss_vals, self.loss_names):
                        logger.logkv(loss_name, loss_val)
                    logger.dumpkvs()

                if self.active_sampling:
                    if len(samples) < samples.maxlen:
                        samples.extendleft(self.env.get_reset_data())
                    else:
                        resets = np.count_nonzero(masks)
                        if resets > 0:
                            sample_obs = [s["obs"] for s in samples]
                            scores = self.train_model.get_critic_discrepancy(sample_obs)
                            samples = deque(
                                [sample for _, sample in sorted(zip(scores, samples), key=lambda pair: pair[0])],
                                maxlen=5 * self.env.num_envs * 10)
                            scenarios = []
                            for i in range(self.env.num_envs * resets):
                                scenarios.append(samples.popleft()["initial_state"])

                            self.env.add_scenarios(scenarios)

                            samples.extendleft(self.env.get_reset_data())
            callback.on_training_end()
            return self

    def save(self, save_path, cloudpickle=False):
        data = {
            "gamma": self.gamma,
            "n_steps": self.n_steps,
            "vf_coef": self.vf_coef,
            "ent_coef": self.ent_coef,
            "max_grad_norm": self.max_grad_norm,
            "learning_rate": self.learning_rate,
            "lam": self.lam,
            "nminibatches": self.nminibatches,
            "noptepochs": self.noptepochs,
            "cliprange": self.cliprange,
            "cliprange_vf": self.cliprange_vf,
            "verbose": self.verbose,
            "policy": self.policy,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_envs": self.n_envs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs,
            "num_timesteps": self.num_timesteps
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)


class Runner(AbstractEnvRunner):
    def __init__(self, *, env, model, n_steps, gamma, lam, time_aware=False, frame_skip=None):
        """
        A runner to learn the policy of an environment for a model

        :param env: (Gym environment) The environment to learn from
        :param model: (Model) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        :param gamma: (float) Discount factor
        :param lam: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        """
        super().__init__(env=env, model=model, n_steps=n_steps)
        self.lam = lam
        self.gamma = gamma
        self.time_aware = time_aware
        self.frame_skip = frame_skip
        if getattr(model.act_model, "time_varying", False):  # TODO: fix for n_envs > 1
            systems = env.env_method("get_linearized_mpc_model_over_prediction")
            model.act_model.lqr.set_numeric_value({"A": [s[0] for s in systems], "B": [s[1] for s in systems]})
            for i in range(self.n_envs):
                As, Bs = systems[i]
                model.train_model.lqr_As.append(As)  # TODO: fix for n_envs > 1
                model.train_model.lqr_Bs.append(Bs)
                #model.train_model.lqr_system_idxs[i].append(len(self.model.train_model.lqr_As) - 1)
            model.train_model.lqr_system_idxs = [[i for i in range(self.n_envs)]]

    def _run(self):
        """
        Run a learning step of the model

        :return:
            - observations: (np.ndarray) the observations
            - rewards: (np.ndarray) the rewards
            - masks: (numpy bool) whether an episode is over or not
            - actions: (np.ndarray) the actions
            - values: (np.ndarray) the value function output
            - negative log probabilities: (np.ndarray)
            - states: (np.ndarray) the internal states of the recurrent policies
            - infos: (dict) the extra information of the model
        """
        # mb stands for minibatch
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs, mb_envterms = [], [], [], [], [], [], []
        mb_states = self.states
        ep_infos = []

        self.model.train_model.vf_obs = []
        self.model.train_model.values = []
        self.model.train_model.rewards_n = []

        for _ in range(self.n_steps):
            if self.frame_skip is not None and _ % self.frame_skip != 0:
                __, values, self.states, __ = self.model.step(self.obs, self.states, self.dones)
                #neglogpacs = self.model.sess.run(self.model.neglogpac, {self.model.train_model.obs_ph: self.obs, self.model.action_ph: actions})
            else:
                actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
                self.model.train_model.vf_obs.append(self.obs.copy())
                self.model.train_model.values.append(values.copy())
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.env.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
            if self.frame_skip:
                self.obs, rewards, self.dones, infos = self.env.step(clipped_actions)
                if _ > 0 and _ % self.frame_skip != 0:  # TODO: does not treat dones
                    self.model.train_model.rewards_n[-1] += rewards
                else:
                    self.model.train_model.rewards_n.append(np.copy(rewards))
                #self.obs[:, self.obs.shape[1] // 2:] = newobs[:, self.obs.shape[1] // 2:]
            else:
                self.obs[:], rewards, self.dones, infos = self.env.step(clipped_actions)
            if getattr(self.model.train_model, "time_varying", False) or getattr(self.model.act_model, "train_mpc_value_fn", False):
                system_idxs = []
                for i in range(self.n_envs):
                    if self.dones[i] or "As" in infos[i]:
                        if self.dones[i]:
                            As, Bs = self.env.env_method("get_linearized_mpc_model_over_prediction", indices=i)[0]
                        else:
                            As, Bs = infos[i].pop("As"), infos[i].pop("Bs")
                        self.model.act_model.lqr.set_numeric_value({"A": As, "B": Bs}, indices=i)
                        #self.model.sess.run(self.model.train_model.lqr_K_var.assign(self.model.train_model.lqr.K_num))
                        self.model.train_model.lqr_As.append(As)
                        self.model.train_model.lqr_Bs.append(Bs)
                        #self.model.train_model.lqr_system_idxs[i].append(len(self.model.train_model.lqr_As) - 1)  # TODO: ensure same length as the other data (it is n_env longer, but i think that is correct, i.e. it has the system from the last step we got the obs from but havent calculated the action for yet).
                        system_idxs.append(len(self.model.train_model.lqr_As) - 1)
                    else:
                        #self.model.train_model.lqr_system_idxs[i].append(self.model.train_model.lqr_system_idxs[i][-1])  # TODO: think this should work but need to ensrue that obs[:, ..] will have same order as this
                        system_idxs.append(self.model.train_model.lqr_system_idxs[-1][i])
                self.model.train_model.lqr_system_idxs.append(system_idxs)

            self.model.num_timesteps += self.n_envs

            if self.callback is not None:
                # Abort training early
                if self.callback.on_step() is False:
                    self.continue_training = False
                    # Return dummy values
                    return [None] * 9

            env_terms = []
            for info in infos:
                maybe_ep_info = info.get('episode')
                termination_reason = info.get("termination", None)
                if maybe_ep_info is not None:
                    ep_infos.append(maybe_ep_info)
                env_terms.append(termination_reason is not None and termination_reason != "steps")
            env_terms = np.array(env_terms)
            mb_envterms.append(env_terms)
            mb_rewards.append(rewards)
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)


        # discount/bootstrap off value fn
        mb_advs = np.zeros_like(mb_rewards)
        true_reward = np.copy(mb_rewards)
        self.env_terms = mb_envterms
        last_gae_lam = 0

        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                if self.time_aware and False:  # i dont think this works
                    nextnonterminal = 1.0 - env_terms
                else:
                    nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                if self.time_aware and False:  # i dont think this works
                    nextnonterminal = 1.0 - mb_envterms[step + 1]
                else:
                    nextnonterminal = 1.0 - mb_dones[step + 1]
                nextvalues = mb_values[step + 1]
            delta = mb_rewards[step] + self.gamma * nextvalues * nextnonterminal - mb_values[step]
            mb_advs[step] = last_gae_lam = delta + self.gamma * self.lam * nextnonterminal * last_gae_lam
        mb_returns = mb_advs + mb_values

        mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward = \
            map(swap_and_flatten, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward))  # TODO: because of swap and flatten, lqr stuff is probably in the wrong order

        if getattr(self.model.train_model, "time_varying", False):
            self.model.train_model.lqr_system_idxs = swap_and_flatten(np.array(self.model.train_model.lqr_system_idxs[:-1]))

            #self.model.act_model.origKs = swap_and_flatten(np.array(self.model.act_model.origKs))
            #self.model.act_model.d_actions = swap_and_flatten(np.array(self.model.act_model.d_actions))

        if self.frame_skip is not None:
            self.model.train_model.vf_obs = swap_and_flatten(np.array(self.model.train_model.vf_obs))
            self.model.train_model.values = swap_and_flatten(np.array(self.model.train_model.values))
            self.model.train_model.rewards_n = swap_and_flatten(np.array(self.model.train_model.rewards_n))
            self.model.train_model.rewards_n = mb_returns[::self.frame_skip]

        return mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_states, ep_infos, true_reward

    def get_advantages(self, rewards, values, dones, last_values):
        def deswap_and_flatten(vec):
            vec = vec.reshape(self.n_envs, self.n_steps)
            return vec.swapaxes(0, 1)
        last_gae_lam = 0
        if len(rewards.shape) == 1:
            rewards = deswap_and_flatten(rewards)
            values = deswap_and_flatten(values)
            dones = deswap_and_flatten(dones)
        advs = np.zeros_like(rewards)
        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                if self.time_aware and False:
                    nextnonterminal = 1.0 - self.env_terms[-1]
                else:
                    nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                if self.time_aware and False:
                    nextnonterminal = 1.0 - self.env_terms[step + 1]
                else:
                    nextnonterminal = 1.0 - dones[step + 1]
                nextvalues = values[step + 1]
            delta = rewards[step] + self.gamma * nextvalues * nextnonterminal - values[step]
            advs[step] = last_gae_lam = delta + self.gamma * self.lam * nextnonterminal * last_gae_lam

        return advs


# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def swap_and_flatten(arr):
    """
    swap and then flatten axes 0 and 1

    :param arr: (np.ndarray)
    :return: (np.ndarray)
    """
    shape = arr.shape
    return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])
