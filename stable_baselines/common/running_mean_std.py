import numpy as np


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=(), update_every=1, mean_mask=None, var_mask=None):
        """
        calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: (float) helps with arithmetic issues
        :param shape: (tuple) the shape of the data stream's output
        """
        if len(shape) > 1:
            print("WARNING: CALLED RUNNIGMEANSTD WITH SHAPE WITH 2 OR MORE DIMENSIONS, WILL ONLY USE LAST DIMENSION")
            shape = shape[-1]
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon
        self.mean_mask = mean_mask
        self.var_mask = var_mask
        self.update_every = update_every
        self.buf = []

    def update(self, arr):
        if len(arr.shape) > 1 and arr.shape[0] == 1:
            self.buf.append(arr)
            if len(self.buf) < self.update_every:
                return
            else:
                arr = np.concatenate(self.buf)
                self.buf = []
        if len(self.mean.shape) == 0:
            batch_mean = np.mean(arr)
            batch_var = np.var(arr)
        else:
            batch_mean = np.mean(arr, axis=tuple((i for i in range(len(arr.shape) - 1))))
            batch_var = np.var(arr, axis=tuple((i for i in range(len(arr.shape) - 1))))
        if self.mean_mask is not None:
            batch_mean[..., self.mean_mask] = 0
        if self.var_mask is not None:
            batch_var[self.var_mask] = 1
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count
        
        
class RunningMeanStdSerial(object):
    def __init__(self, epsilon=1e-4, shape=(), mean_mask=None):
        """
        calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: (float) helps with arithmetic issues
        :param shape: (tuple) the shape of the data stream's output
        """
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon
        self.mean_mask = mean_mask
        self.m2 = np.zeros(shape, "float64")

    def update(self, arr):
        delta = arr - self.mean
        self.count += 1
        
        self.mean += delta / self.count
        delta2 = arr - self.mean
        self.m2 += delta * delta2
        self.var = self.m2 / self.count
        
