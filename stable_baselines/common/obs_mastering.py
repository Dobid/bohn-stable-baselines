import tensorflow as tf


class ObsMaster:
    def __init__(self, obs_dim, actor):
        self.nn = tf.layers.dense(2 * obs_dim, obs_dim, activation=tf.tanh)

        pass

    def get_loss(self):
        pass