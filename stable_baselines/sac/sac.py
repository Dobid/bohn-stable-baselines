import sys
import time
import warnings

import numpy as np
import tensorflow as tf

from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.math_util import safe_mean, unscale_action, scale_action
from stable_baselines.common.schedules import get_schedule_fn
from stable_baselines.common.buffers import ReplayBuffer
from stable_baselines.sac.policies import SACPolicy, AHMPCPolicy
from stable_baselines import logger

from stable_baselines.her.replay_buffer import HindsightExperienceReplayWrapper

import copy

class SAC(OffPolicyRLModel):
    """
    Soft Actor-Critic (SAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup) and from the Softlearning repo
    (https://github.com/rail-berkeley/softlearning/)
    Paper: https://arxiv.org/abs/1801.01290
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

    :param policy: (SACPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, LnMlpPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) the discount factor
    :param learning_rate: (float or callable) learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress (from 1 to 0)
    :param buffer_size: (int) size of the replay buffer
    :param batch_size: (int) Minibatch size for each gradient update
    :param tau: (float) the soft update coefficient ("polyak update", between 0 and 1)
    :param ent_coef: (str or float) Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param train_freq: (int) Update the model every `train_freq` steps.
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param target_update_interval: (int) update the target network every `target_network_update_freq` steps.
    :param gradient_steps: (int) How many gradient update after each step
    :param target_entropy: (str or float) target entropy when learning ent_coef (ent_coef = 'auto')
    :param action_noise: (ActionNoise) the action noise type (None by default), this can help
        for hard exploration problem. Cf DDPG for the different action noise type.
    :param random_exploration: (float) Probability of taking a random action (as in an epsilon-greedy strategy)
        This is not needed for SAC normally but can help exploring when using HER + SAC.
        This hack was present in the original OpenAI Baselines repo (DDPG + HER)
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        Note: this has no effect on SAC logging for now
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    """

    def __init__(self, policy, env, gamma=0.99, learning_rate=3e-4, buffer_size=50000, buffer_type=ReplayBuffer,
                 learning_starts=100, train_freq=1, batch_size=64,
                 tau=0.005, reward_scale=1, ent_coef='auto', target_update_interval=1, action_l2_scale=None,
                 gradient_steps=1, target_entropy='auto', action_noise=None,
                 random_exploration=0.0, spatial_similarity_coef=None, temporal_similarity_coef=None, verbose=0, write_freq=1, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False,
                 seed=None, n_cpu_tf_sess=None, time_aware=False):

        super(SAC, self).__init__(policy=policy, env=env, replay_buffer=None, verbose=verbose, write_freq=write_freq,
                                  policy_base=SACPolicy, requires_vec_env=False, policy_kwargs=policy_kwargs,
                                  seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)
        self.buffer_type = buffer_type
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.tau = tau
        # In the original paper, same learning rate is used for all networks
        # self.policy_lr = learning_rate
        # self.qf_lr = learning_rate
        # self.vf_lr = learning_rate
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.action_l2_scale = action_l2_scale
        self.target_update_interval = target_update_interval
        self.gradient_steps = gradient_steps
        self.gamma = gamma
        self.action_noise = action_noise
        self.random_exploration = random_exploration
        self.reward_scale = reward_scale
        self.spatial_similarity_coef = spatial_similarity_coef
        self.temporal_similarity_coef = temporal_similarity_coef
        self.similar_obs_ph = None


        self.value_fn = None
        self.graph = None
        self.replay_buffer = None
        self.sess = None
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self.params = None
        self.summary = None
        self.policy_tf = None
        self.target_entropy = target_entropy
        self.full_tensorboard_log = full_tensorboard_log

        self.obs_target = None
        self.target_policy_tf = None
        self.actions_ph = None
        self.rewards_ph = None
        self.terminals_ph = None
        self.observations_ph = None
        self.action_target = None
        self.next_observations_ph = None
        self.value_target = None
        self.step_ops = None
        self.target_update_op = None
        self.infos_names = None
        self.entropy = None
        self.target_params = None
        self.learning_rate_ph = None
        self.processed_obs_ph = None
        self.processed_next_obs_ph = None
        self.log_ent_coef = None

        self.time_aware = time_aware

        self.train_extra_phs = {}

        if _init_setup_model:
            self.setup_model()

    def _get_pretrain_placeholders(self):
        policy = self.policy_tf
        # Rescale
        deterministic_action = unscale_action(self.action_space, self.deterministic_action)
        return policy.obs_ph, self.actions_ph, deterministic_action

    def setup_model(self):
        with SetVerbosity(self.verbose):
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(self.seed)
                self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)

                with tf.variable_scope("input", reuse=False):
                    # Create policy and target TF objects
                    self.policy_tf = self.policy(self.sess, self.observation_space, self.action_space,
                                                 **self.policy_kwargs)
                    self.target_policy_tf = self.policy(self.sess, self.observation_space, self.action_space,
                                                **self.policy_kwargs)

                    if hasattr(self.policy_tf, "extra_phs"):
                        for ph_name in self.policy_tf.extra_phs:
                            if "target_" in ph_name:
                                self.train_extra_phs[ph_name] = getattr(self.target_policy_tf,
                                                                        ph_name.replace("target_", "") + "_ph")
                            else:
                                self.train_extra_phs[ph_name] = getattr(self.policy_tf, ph_name + "_ph")

                    # Initialize Placeholders
                    self.observations_ph = self.policy_tf.obs_ph
                    # Normalized observation for pixels
                    self.processed_obs_ph = self.policy_tf.processed_obs
                    self.next_observations_ph = self.target_policy_tf.obs_ph
                    self.processed_next_obs_ph = self.target_policy_tf.processed_obs
                    self.action_target = self.target_policy_tf.action_ph
                    self.terminals_ph = tf.placeholder(tf.float32, shape=(None, 1), name='terminals')
                    self.rewards_ph = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
                    self.actions_ph = tf.placeholder(tf.float32, shape=(None,) + self.action_space.shape,
                                                     name='actions')
                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")

                    if issubclass(self.policy, AHMPCPolicy) and self.policy_tf.use_mpc_value_fn:
                        self.mpc_state_ph = self.policy_tf.mpc_state_ph
                        self.next_mpc_state_ph = self.policy_tf.mpc_next_state_ph
                        if self.policy_tf.train_mpc_value_fn:
                            self.mpc_rewards_ph = tf.placeholder(tf.float32, shape=(None, 1), name="mpc_rewards")
                            self.mpc_n_step_ph = tf.placeholder(tf.float32, shape=(None, 1), name="mpc_n_step")
                            self.mpc_terminals_ph = tf.placeholder(tf.float32, shape=(None, 1), name='mpc_terminals')
                            self.train_extra_phs.update({"mpc_rewards": self.mpc_rewards_ph, "mpc_state": self.mpc_state_ph,
                                                        "mpc_next_state": self.next_mpc_state_ph,
                                                         "mpc_n_step": self.mpc_n_step_ph,
                                                         "mpc_terminals": self.mpc_terminals_ph})

                replay_buffer_kw = {"extra_data_names": []}
                if self.time_aware:
                    replay_buffer_kw["extra_data_names"].append("bootstrap")
                if self.replay_buffer is None:
                    if issubclass(self.policy, AHMPCPolicy) and self.policy_tf.train_mpc_value_fn:
                        self.mpc_replay_buffer = self.buffer_type(self.buffer_size, extra_data_names=replay_buffer_kw["extra_data_names"])
                    else:
                        replay_buffer_kw["extra_data_names"].extend(self.train_extra_phs.keys())
                    self.replay_buffer = self.buffer_type(self.buffer_size, **replay_buffer_kw)

                with tf.variable_scope("model", reuse=False):
                    # Create the policy
                    # first return value corresponds to deterministic actions
                    # policy_out corresponds to stochastic actions, used for training
                    # logp_pi is the log probability of actions taken by the policy
                    self.deterministic_action, policy_out, logp_pi = self.policy_tf.make_actor(self.processed_obs_ph)
                    # Monitor the entropy of the policy,
                    # this is not used for training
                    self.entropy = tf.reduce_mean(self.policy_tf.entropy)
                    #  Use two Q-functions to improve performance by reducing overestimation bias.
                    qf1, qf2, value_fn = self.policy_tf.make_critics(self.processed_obs_ph, self.actions_ph,
                                                                     create_qf=True, create_vf=True)
                    qf1_pi, qf2_pi, _ = self.policy_tf.make_critics(self.processed_obs_ph,
                                                                    policy_out, create_qf=True, create_vf=False,
                                                                    reuse=True)

                    if issubclass(self.policy, AHMPCPolicy) and self.policy_tf.use_mpc_value_fn:
                        self.mpc_value_fn = self.policy_tf.make_mpc_value_fn(self.mpc_state_ph)
                        if not self.policy_tf.use_mpc_vf_target:
                            self.mpc_value_fn_term_state = self.policy_tf.make_mpc_value_fn(self.next_mpc_state_ph, reuse=True)

                    # Target entropy is used when learning the entropy coefficient
                    if self.target_entropy == 'auto':
                        # automatically set target entropy if needed
                        self.target_entropy = -np.prod(self.action_space.shape).astype(np.float32)
                    else:
                        # Force conversion
                        # this will also throw an error for unexpected string
                        self.target_entropy = float(self.target_entropy)

                    # The entropy coefficient or entropy can be learned automatically
                    # see Automating Entropy Adjustment for Maximum Entropy RL section
                    # of https://arxiv.org/abs/1812.05905
                    if isinstance(self.ent_coef, str) and self.ent_coef.startswith('auto'):
                        # Default initial value of ent_coef when learned
                        init_value = 1.0
                        if '_' in self.ent_coef:
                            init_value = float(self.ent_coef.split('_')[1])
                            assert init_value > 0., "The initial value of ent_coef must be greater than 0"

                        self.log_ent_coef = tf.get_variable('log_ent_coef', dtype=tf.float32,
                                                            initializer=np.log(init_value).astype(np.float32))
                        self.ent_coef = tf.exp(self.log_ent_coef)
                    else:
                        # Force conversion to float
                        # this will throw an error if a malformed string (different from 'auto')
                        # is passed
                        self.ent_coef = float(self.ent_coef)

                with tf.variable_scope("target", reuse=False):
                    # Create the value network
                    _, _, value_target = self.target_policy_tf.make_critics(self.processed_next_obs_ph,
                                                                         create_qf=False, create_vf=True)
                    self.value_target = value_target

                    if issubclass(self.policy, AHMPCPolicy) and self.policy_tf.use_mpc_vf_target:
                        self.mpc_value_fn_term_state = self.target_policy_tf.make_mpc_value_fn(self.next_mpc_state_ph)

                if self.temporal_similarity_coef is not None:
                    with tf.variable_scope("model", reuse=True):
                        self.deterministic_action_next, _, _ = self.policy_tf.make_actor(self.processed_next_obs_ph, reuse=True)

                if self.spatial_similarity_coef is not None:
                    self.similar_obs_ph = tf.placeholder(tf.float32, shape=self.observations_ph.shape, name="similar_obs_ph")
                    self.train_extra_phs["similar_obs"] = self.similar_obs_ph
                    with tf.variable_scope("model", reuse=True):
                        self.deterministic_action_similar, _, _ = self.policy_tf.make_actor(self.similar_obs_ph, reuse=True)

                with tf.variable_scope("loss", reuse=False):
                    # Take the min of the two Q-Values (Double-Q Learning)
                    min_qf_pi = tf.minimum(qf1_pi, qf2_pi)

                    # Target for Q value regression
                    q_backup = tf.stop_gradient(
                        self.rewards_ph +
                        (1 - self.terminals_ph) * self.gamma * self.value_target
                    )

                    # Compute Q-Function loss
                    # TODO: test with huber loss (it would avoid too high values)
                    qf1_loss = 0.5 * tf.reduce_mean((q_backup - qf1) ** 2)
                    qf2_loss = 0.5 * tf.reduce_mean((q_backup - qf2) ** 2)

                    if issubclass(self.policy, AHMPCPolicy) and self.policy_tf.train_mpc_value_fn:
                        mpc_value_fn_backup = tf.stop_gradient(self.mpc_rewards_ph + (1 - self.mpc_terminals_ph) * self.mpc_value_fn_term_state * self.policy_tf.mpc_gamma ** self.mpc_n_step_ph)
                        self.mpc_value_fn_loss = mpc_value_fn_loss = 0.5 * tf.reduce_mean((mpc_value_fn_backup - self.mpc_value_fn) ** 2)
                        self.mpc_value_fn_backup = mpc_value_fn_backup

                    # Compute the entropy temperature loss
                    # it is used when the entropy coefficient is learned
                    ent_coef_loss, entropy_optimizer = None, None
                    if not isinstance(self.ent_coef, float):
                        ent_coef_loss = -tf.reduce_mean(
                            self.log_ent_coef * tf.stop_gradient(logp_pi + self.target_entropy))
                        entropy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)

                    # Compute the policy loss
                    # Alternative: policy_kl_loss = tf.reduce_mean(logp_pi - min_qf_pi)
                    policy_kl_loss = tf.reduce_mean(self.ent_coef * logp_pi - qf1_pi)

                    # NOTE: in the original implementation, they have an additional
                    # regularization loss for the Gaussian parameters
                    # this is not used for now
                    # policy_loss = (policy_kl_loss + policy_regularization_loss)
                    policy_loss = policy_kl_loss
                    if self.action_l2_scale is not None:
                        action_loss = self.action_l2_scale * tf.nn.l2_loss(self.policy_tf.policy_pre_activation)
                        policy_loss += action_loss
                    if self.spatial_similarity_coef is not None:
                        spatial_similarity_loss = self.spatial_similarity_coef * tf.nn.l2_loss(self.deterministic_action - self.deterministic_action_similar, name="spatial_similarity_loss") / self.action_space.shape[0]
                        policy_loss += spatial_similarity_loss

                    if self.temporal_similarity_coef is not None:
                        temporal_similarity_loss = self.temporal_similarity_coef * tf.nn.l2_loss(self.deterministic_action - self.deterministic_action_next, name="temporal_similarity_loss") / self.action_space.shape[0]
                        policy_loss += temporal_similarity_loss


                    # Target for value fn regression
                    # We update the vf towards the min of two Q-functions in order to
                    # reduce overestimation bias from function approximation error.
                    v_backup = tf.stop_gradient(min_qf_pi - self.ent_coef * logp_pi)
                    value_loss = 0.5 * tf.reduce_mean((value_fn - v_backup) ** 2)

                    values_losses = qf1_loss + qf2_loss + value_loss
                    if issubclass(self.policy, AHMPCPolicy) and self.policy_tf.train_mpc_value_fn:
                        mpc_value_fn_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                        self.mpc_value_fn_train_op = mpc_value_fn_train_op = mpc_value_fn_optimizer.minimize(mpc_value_fn_loss, var_list=tf_util.get_trainable_vars("model/mpc_value_fns"))

                    # Policy train op
                    # (has to be separate from value train op, because min_qf_pi appears in policy_loss)
                    policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    policy_train_op = policy_optimizer.minimize(policy_loss, var_list=tf_util.get_trainable_vars('model/pi'))

                    # Value train op
                    value_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    values_params = tf_util.get_trainable_vars('model/values_fn')

                    source_params = tf_util.get_trainable_vars("model/values_fn/vf")
                    target_params = tf_util.get_trainable_vars("target/values_fn/vf")
                    if issubclass(self.policy, AHMPCPolicy) and self.policy_tf.use_mpc_vf_target:
                        source_params += tf_util.get_trainable_vars("model/mpc_value_fns")
                        target_params += tf_util.get_trainable_vars("target/mpc_value_fns")


                    # Polyak averaging for target variables
                    self.target_update_op = [
                        tf.assign(target, (1 - self.tau) * target + self.tau * source)
                        for target, source in zip(target_params, source_params)
                    ]
                    # Initializing target to match source variables
                    target_init_op = [
                        tf.assign(target, source)
                        for target, source in zip(target_params, source_params)
                    ]

                    # Control flow is used because sess.run otherwise evaluates in nondeterministic order
                    # and we first need to compute the policy action before computing q values losses
                    with tf.control_dependencies([policy_train_op]):
                        train_values_op = value_optimizer.minimize(values_losses, var_list=values_params)

                        self.infos_names = ['policy_loss', 'qf1_loss', 'qf2_loss', 'value_loss', 'entropy']
                        # All ops to call during one training step
                        self.step_ops = [policy_loss, qf1_loss, qf2_loss,
                                         value_loss, qf1, qf2, value_fn, logp_pi,
                                         self.entropy, policy_train_op, train_values_op]
                        if issubclass(self.policy, AHMPCPolicy) and self.policy_tf.train_mpc_value_fn:
                            self.step_ops.append(mpc_value_fn_train_op)

                        # Add entropy coefficient optimization operation if needed
                        if ent_coef_loss is not None:
                            with tf.control_dependencies([train_values_op]):
                                ent_coef_op = entropy_optimizer.minimize(ent_coef_loss, var_list=self.log_ent_coef)
                                self.infos_names += ['ent_coef_loss', 'ent_coef']
                                self.step_ops += [ent_coef_op, ent_coef_loss, self.ent_coef]

                    # Monitor losses and entropy in tensorboard
                    tf.summary.scalar('policy_loss', policy_loss)
                    tf.summary.scalar('qf1_loss', qf1_loss)
                    tf.summary.scalar('qf2_loss', qf2_loss)
                    tf.summary.scalar('value_loss', value_loss)
                    tf.summary.scalar('entropy', self.entropy)
                    tf.summary.scalar("std", tf.reduce_mean(self.policy_tf.std))
                    if ent_coef_loss is not None:
                        tf.summary.scalar('ent_coef_loss', ent_coef_loss)
                        tf.summary.scalar('ent_coef', self.ent_coef)
                    if issubclass(self.policy, AHMPCPolicy) and self.policy_tf.train_mpc_value_fn:
                        tf.summary.scalar("mpc_value_fn_loss", mpc_value_fn_loss)
                    if self.spatial_similarity_coef is not None:
                        tf.summary.scalar("spatial_similarity_loss", spatial_similarity_loss)
                    if self.temporal_similarity_coef is not None:
                        tf.summary.scalar("temporal_similarity_loss", temporal_similarity_loss)
                    if self.action_l2_scale:
                        tf.summary.scalar("action_l2_loss", action_loss)

                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))

                # Retrieve parameters that must be saved
                self.params = tf_util.get_trainable_vars("model")
                self.target_params = target_params

                # Initialize Variables and target network
                with self.sess.as_default():
                    self.sess.run(tf.global_variables_initializer())
                    self.sess.run(target_init_op)
                    if getattr(self.policy_tf, "mpc_value_fn_path", None) is not None:
                        self.load_parameters_by_name(self.policy_tf.mpc_value_fn_path, [p.name for p in tf_util.get_trainable_vars("model/mpc_value_fns")])  # TODO: ensure only loading mpc value fn params

                self.summary = tf.summary.merge_all()

    def _train_step(self, step, writer, learning_rate):
        # Sample a batch from the replay buffer
        batch = self.replay_buffer.sample(self.batch_size, env=self._vec_normalize_env)
        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones, *batch_extra = batch
        if len(batch_extra) > 0:
            batch_extra = batch_extra[0]
        if issubclass(self.policy, AHMPCPolicy) and self.policy_tf.train_mpc_value_fn:
            batch_mpc = self.mpc_replay_buffer.sample(self.batch_size, n_step=32, gamma=self.policy_tf.mpc_gamma)
            batch_extra["mpc_state"] = batch_mpc[0]
            batch_extra["mpc_rewards"] = batch_mpc[2]
            batch_extra["mpc_next_state"] = batch_mpc[3]
            batch_extra["mpc_n_step"] = batch_mpc[5].get("n_step", np.ones((256, 1)))
            batch_extra["mpc_terminals"] = batch_mpc[4]
        if self.spatial_similarity_coef is not None:
            batch_extra["similar_obs"] = np.random.normal(batch_obs, 0.01)

        feed_dict = {
            self.observations_ph: batch_obs,
            self.actions_ph: batch_actions,
            self.next_observations_ph: batch_next_obs,
            self.rewards_ph: batch_rewards.reshape(self.batch_size, -1),
            self.terminals_ph: batch_dones.reshape(self.batch_size, -1),
            self.learning_rate_ph: learning_rate
        }

        for k, v in self.train_extra_phs.items():
            if len(batch_extra[k].shape) == 1:
                feed_dict[v] = batch_extra[k].reshape(self.batch_size, -1)
            else:
                feed_dict[v] = batch_extra[k]

        # out  = [policy_loss, qf1_loss, qf2_loss,
        #         value_loss, qf1, qf2, value_fn, logp_pi,
        #         self.entropy, policy_train_op, train_values_op]

        # Do one gradient step
        # and optionally compute log for tensorboard
        if writer is not None:
            out = self.sess.run([self.summary] + self.step_ops, feed_dict)
            summary = out.pop(0)
            writer.add_summary(summary, step)
        else:
            out = self.sess.run(self.step_ops, feed_dict)

        # Unpack to monitor losses and entropy
        policy_loss, qf1_loss, qf2_loss, value_loss, *values = out
        # qf1, qf2, value_fn, logp_pi, entropy, *_ = values
        entropy = values[4]

        if self.log_ent_coef is not None:
            ent_coef_loss, ent_coef = values[-2:]
            return policy_loss, qf1_loss, qf2_loss, value_loss, entropy, ent_coef_loss, ent_coef

        return policy_loss, qf1_loss, qf2_loss, value_loss, entropy

    def learn(self, total_timesteps, callback=None,
              log_interval=4, tb_log_name="SAC", reset_num_timesteps=True, replay_wrapper=None, random_sampler=None):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        callback = self._init_callback(callback)

        action_space = copy.deepcopy(self.action_space)
        vectorize_objects = self.n_envs == 1 and not issubclass(type(self.env), VecEnv)
        if not vectorize_objects:
            action_space.sample = lambda: np.array([self.action_space.sample() for _ in range(self.n_envs)])
            action_space.low = np.repeat(action_space.low[np.newaxis, ...], self.n_envs, axis=0)
            action_space.high = np.repeat(action_space.high[np.newaxis, ...], self.n_envs, axis=0)
            action_space.shape = action_space.low.shape

        if replay_wrapper is not None:
            self.replay_buffer = replay_wrapper(self.replay_buffer)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:

            self._setup_learn()

            # Transform to callable if needed
            self.learning_rate = get_schedule_fn(self.learning_rate)
            # Initial learning rate
            current_lr = self.learning_rate(1)

            initial_step = self.num_timesteps

            start_time = time.time()
            episode_rewards = [[0.0] for _ in range(self.n_envs)]
            episode_successes = []
            if self.action_noise is not None:
                self.action_noise.reset()
            obs = self.env.reset()
            if vectorize_objects:
                obs = [obs]
            # Retrieve unnormalized observation for saving into the buffer
            if self._vec_normalize_env is not None:
                obs_ = self._vec_normalize_env.get_original_obs().squeeze()

            n_updates = 0
            infos_values = [[] for _ in range(self.n_envs)]
            done = [[False] for _ in range(self.n_envs)]
            episode_data = [[] for _ in range(self.n_envs)]

            callback.on_training_start(locals(), globals())
            callback.on_rollout_start()

            for step in range(initial_step, total_timesteps, self.n_envs):
                # Before training starts, randomly sample actions
                # from a uniform distribution for better exploration.
                # Afterwards, use the learned policy
                # if random_exploration is set to 0 (normal setting)
                if self.num_timesteps < self.learning_starts or np.random.rand() < self.random_exploration:
                    # actions sampled from action space are from range specific to the environment
                    # but algorithm operates on tanh-squashed actions therefore simple scaling is used
                    if random_sampler is None:
                        unscaled_action = action_space.sample()
                    else:
                        unscaled_action = random_sampler(obs)
                    #unscaled_action = np.array([[1]])
                    #unscaled_action = np.random.uniform(25, 50, size=(1,))
                    action = scale_action(action_space, unscaled_action)
                else:
                    if self.n_envs == 1:
                        step_obs = obs[0]
                    else:
                        step_obs = obs
                    if self.n_envs > 1:
                        action = self.policy_tf.step(step_obs, deterministic=False)
                    else:
                        action = self.policy_tf.step(step_obs[None], deterministic=False)
                    if vectorize_objects:
                        action = action.flatten()
                    # Add noise to the action (improve exploration,
                    # not needed in general)
                    if self.action_noise is not None:
                        action = np.clip(action + self.action_noise(), -1, 1)
                    # inferred actions need to be transformed to environment action_space before stepping
                    unscaled_action = unscale_action(action_space, action)

                assert action.shape == action_space.shape

                new_obs, reward, done, info = self.env.step(unscaled_action)
                if vectorize_objects:
                    new_obs = [new_obs]
                    reward = [reward]
                    done = [done]
                    info = [info]
                    action = [action]

                reward = [r / self.reward_scale for r in reward]

                self.num_timesteps += self.n_envs

                self.info = info

                # Only stop training if return value is False, not when it is None. This is for backwards
                # compatibility with callbacks that have no return statement.
                if callback.on_step() is False:
                    break

                # Store only the unnormalized version
                if self._vec_normalize_env is not None:
                    new_obs_ = self._vec_normalize_env.get_original_obs().squeeze()
                    reward_ = self._vec_normalize_env.get_original_reward().squeeze()

                obs_, new_obs_, reward_ = obs, new_obs, reward

                # Store transition in the replay buffer.
                extra_data = [{} for _ in range(self.n_envs)]
                if self.time_aware:
                    for env_i in range(self.n_envs):
                        bootstrap = True
                        if done[env_i]:
                            info_time_limit = info[env_i].get("TimeLimit.truncated", None)
                            bootstrap = info[env_i].get("termination", None) == "steps" or \
                                        (info_time_limit is not None and info_time_limit)
                        extra_data[env_i]["bootstrap"] = bootstrap

                if hasattr(self.policy, "collect_data"):
                    policy_data = self.policy_tf.collect_data(locals(), globals())
                    for env_i in range(self.n_envs):
                        extra_data[env_i].update(policy_data[env_i])

                for env_i in range(self.n_envs):
                    #extra_data[env_i].update(info[env_i].get("data", {}))
                    if issubclass(self.policy, AHMPCPolicy) and self.policy_tf.train_mpc_value_fn:
                        mpc_extra_data = copy.deepcopy(extra_data[env_i])
                        self.mpc_replay_buffer.add(info[env_i]["data"]["mpc_state"], np.array([0]), info[env_i]["data"]["mpc_rewards"], info[env_i]["data"]["mpc_next_state"], float(done[env_i]), **mpc_extra_data)
                    if isinstance(self.replay_buffer, HindsightExperienceReplayWrapper):
                        self.replay_buffer.add(obs_[env_i], action[env_i], reward_[env_i], new_obs_[env_i],
                                               float(done[env_i]), env_i=env_i, **extra_data[env_i])
                    else:
                        self.replay_buffer.add(obs_[env_i], action[env_i], reward_[env_i], new_obs_[env_i],
                                               float(done[env_i]), **extra_data[env_i])

                for env_i in range(self.n_envs):
                    episode_data[env_i].append({"obs": obs, "action": action, "reward": reward, "obs_tp1": new_obs, "done": done, **extra_data[env_i]})

                obs = new_obs
                # Save the unnormalized observation
                if self._vec_normalize_env is not None:
                    obs_ = new_obs_

                # Retrieve reward and episode length if using Monitor wrapper
                for env_i in range(self.n_envs):
                    maybe_ep_info = info[env_i].get('episode')
                    if maybe_ep_info is not None:
                        self.ep_info_buf.extend([maybe_ep_info])

                if writer is not None:
                    # Write reward per episode to tensorboard
                    ep_reward = np.array([reward_]).reshape((self.n_envs, -1))
                    ep_done = np.array([done]).reshape((self.n_envs, -1))
                    tf_util.total_episode_reward_logger(self.episode_reward, ep_reward,
                                                        ep_done, writer, self.num_timesteps)

                if (self.num_timesteps // self.n_envs) % self.train_freq == 0:
                    callback.on_rollout_end()

                    mb_infos_vals = []
                    # Update policy, critics and target networks
                    for grad_step in range(self.gradient_steps):
                        # Break if the warmup phase is not over
                        # or if there are not enough samples in the replay buffer
                        if not self.replay_buffer.can_sample(self.batch_size) \
                           or self.num_timesteps < self.learning_starts:
                            break
                        n_updates += 1
                        # Compute current learning_rate
                        frac = 1.0 - step / total_timesteps
                        current_lr = self.learning_rate(frac)
                        # Update policy and critics (q functions)
                        step_writer = writer if grad_step % self.write_freq == 0 else None
                        mb_infos_vals.append(self._train_step(step, step_writer, current_lr))
                        # Update target network
                        if (step + grad_step) % self.target_update_interval == 0:
                            # Update target network
                            self.sess.run(self.target_update_op)
                    # Log losses and entropy, useful for monitor training
                    if len(mb_infos_vals) > 0:
                        infos_values = np.mean(mb_infos_vals, axis=0)

                    callback.on_rollout_start()

                for env_i in range(self.n_envs):
                    episode_rewards[env_i][-1] += reward[env_i]
                    if done[env_i]:
                        if self.action_noise is not None:
                            if self.n_envs > 1:
                                self.action_noise.reset(env_i)
                            else:
                                self.action_noise.reset()
                        obs[env_i] = self.env.reset(indices=env_i)
                        if vectorize_objects:
                            obs = [obs]
                        episode_rewards[env_i].append(0.0)
                        episode_data[env_i] = []

                        maybe_is_success = info[env_i].get('is_success')
                        if maybe_is_success is not None:
                            episode_successes.append(float(maybe_is_success))

                num_episodes = sum([len(ep_rews) for ep_rews in episode_rewards])
                # Display training infos
                if self.verbose >= 1 and done[0] and log_interval is not None and len(episode_rewards[0]) % log_interval == 0:
                    if len(episode_rewards[0][-101:-1]) == 0:
                        mean_reward = -np.inf
                    else:
                        mean_reward = round(float(np.mean([np.mean(er[-101:-1]) for er in episode_rewards])), 1)

                    fps = int(step * self.n_envs / (time.time() - start_time))

                    logger.logkv("episodes", num_episodes)
                    logger.logkv("mean 100 episode reward", mean_reward)
                    if len(self.ep_info_buf) > 0 and len(self.ep_info_buf[0]) > 0:
                        logger.logkv('ep_rewmean', safe_mean([ep_info['r'] for ep_info in self.ep_info_buf]))
                        logger.logkv('eplenmean', safe_mean([ep_info['l'] for ep_info in self.ep_info_buf]))
                    logger.logkv("n_updates", n_updates)
                    logger.logkv("current_lr", current_lr)
                    logger.logkv("fps", fps)
                    logger.logkv('time_elapsed', int(time.time() - start_time))
                    if len(episode_successes) > 0:
                        logger.logkv("success rate", np.mean(episode_successes[-100:]))
                    if len(infos_values) > 0:
                        for (name, val) in zip(self.infos_names, infos_values):
                            logger.logkv(name, val)
                    logger.logkv("total timesteps", self.num_timesteps)
                    #logger.logkv("std", self.sess.run(self.policy_tf.std, feed_dict={self.observations_ph: obs[0].reshape(1, -1)})[0])
                    logger.dumpkvs()
                    # Reset infos:
                    infos_values = []
            callback.on_training_end()
            return self

    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        if actions is not None:
            raise ValueError("Error: SAC does not have action probabilities.")

        warnings.warn("Even though SAC has a Gaussian policy, it cannot return a distribution as it "
                      "is squashed by a tanh before being scaled and outputed.")

        return None

    def predict(self, observation, state=None, mask=None, deterministic=True):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        actions = self.policy_tf.step(observation, deterministic=deterministic)
        actions = actions.reshape((-1,) + self.action_space.shape)  # reshape to the correct action shape
        actions = unscale_action(self.action_space, actions) # scale the output for the prediction

        if not vectorized_env:
            actions = actions[0]

        return actions, None

    def get_parameter_list(self):
        return (self.params +
                self.target_params)

    def save(self, save_path, cloudpickle=False, save_replay_buffer=False):
        data = {
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "train_freq": self.train_freq,
            "batch_size": self.batch_size,
            "tau": self.tau,
            "ent_coef": self.ent_coef if isinstance(self.ent_coef, float) else 'auto',
            "target_entropy": self.target_entropy,
            # Should we also store the replay buffer?
            # this may lead to high memory usage
            # with all transition inside
            # "replay_buffer": self.replay_buffer
            "gamma": self.gamma,
            "verbose": self.verbose,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "policy": self.policy,
            "n_envs": self.n_envs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "action_noise": self.action_noise,
            "random_exploration": self.random_exploration,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs,
            "num_timesteps": self.num_timesteps,
            "time_aware": self.time_aware
        }
        if save_replay_buffer:
            data["replay_buffer"] = self.replay_buffer
            if hasattr(self, "mpc_replay_buffer"):
                data["mpc_replay_buffer"] = self.mpc_replay_buffer

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)

    def get_q_value_discrepancies(self, inputs):
        if len(inputs.shape) == 1: # TODO: or matrix shape and len(shape) == 2
            inputs = np.expand_dims(inputs, axis=0)

        qf1, qf2 = self.sess.run([self.policy_tf.qf1, self.policy_tf.qf2], feed_dict={self.observations_ph: np.expand_dims(inputs, axis=0)})

    def export(self, export_path):
        with self.graph.as_default():
            tf.saved_model.simple_save(self.sess, export_path, inputs={"obs": self.policy_tf.obs_ph},
                                       outputs={"action": self.policy_tf.deterministic_policy})
            print("Exported model to {}".format(export_path))
