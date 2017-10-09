import numpy as np
import tensorflow as tf
from scipy.stats import norm
from all_parameters import AllParameters
import math


class NeuralActivity(object):
    """Abstract object representing a time series of neural activity
    """

    def __call__(self):
        """Run neural activity generation process by the parameters.

        Returns:
            A time series of neural activity.
        """


class BasicBlockDesign(NeuralActivity):

    def __init__(self, blo_des_pa=None, batch_size=None):
        if blo_des_pa is None:
            self.blo_des_pa = BlockDesignParameters
        elif isinstance(blo_des_pa, BlockDesignParameters):
            self.blo_des_pa = blo_des_pa
        else:
            raise TypeError('Type of blo_des_pa must be BlockDesignParameters!')

        if batch_size is None:
            self.batch_size = 128
        else:
            self.batch_size = batch_size

    def __call__(self):
        batch_size = self.batch_size
        cycles = self.blo_des_pa.cycles
        stimulu_last = self.blo_des_pa.stimulu_last
        rest_last = self.blo_des_pa.rest_last
        step_size = self.blo_des_pa.step_size
        cycle_length = round((rest_last + stimulu_last) / step_size)
        neural_activity = np.ones(shape=[cycles * cycle_length, self.batch_size])
        for cycle in range(cycles):
            for index in np.arange(0, round(rest_last / step_size)):
                neural_activity[cycle * cycle_length + index, :] = 0

        return neural_activity


class RandomBlockDesign(NeuralActivity):

    def __init__(self, blo_des_pa=None, batch_size=None):
        if blo_des_pa is None:
            self.blo_des_pa = BlockDesignParameters
        elif isinstance(blo_des_pa, BlockDesignParameters):
            self.blo_des_pa = blo_des_pa
        else:
            raise TypeError('Type of blo_des_pa must be BlockDesignParameters!')

        if batch_size is None:
            self.batch_size = 128
        else:
            self.batch_size = batch_size

    def __call__(self):
        batch_size = self.batch_size
        cycles = self.blo_des_pa.cycles

        random_rate = self.blo_des_pa.random_rate
        if random_rate < 0 or random_rate > 1:
            raise TypeError('Random rate is illgal!')

        stimulu_last = self.blo_des_pa.stimulu_last
        rest_last = self.blo_des_pa.rest_last
        step_size = self.blo_des_pa.step_size
        cycle_length = round((rest_last + stimulu_last) / step_size)
        neural_activity = np.ones(shape=[cycles * cycle_length, self.batch_size])
        for cycle in range(cycles):
            for j in range(batch_size):
                rest_last = self.blo_des_pa.rest_last * (1 - random_rate) + \
                               cycle_length * np.random.uniform() * random_rate
                for index in np.arange(0, round(rest_last)):
                    neural_activity[cycle * cycle_length + index, j] = 0

        return neural_activity


class BlockDesignParameters:
    cycles, step_size, stimulu_last, rest_last, random_rate = AllParameters().get_block_design_parameters()

    def __init__(self, blo_des_pa):
        self.cycles = blo_des_pa.cycles
        self.step_size = blo_des_pa.step_size
        self.stimulu_last = blo_des_pa.stimulu_last
        self.rest_last = blo_des_pa.stimulu_interval
        self.downsampling_rate = blo_des_pa.downsampling_rate
        self.batch_size = blo_des_pa.batch_size


class BasicGaussianBump(NeuralActivity):

    def __init__(self, gau_bum_pa=None, batch_size=None):
        if gau_bum_pa is None:
            self.gau_bum_pa = GaussianBumpParameters(batch_size=batch_size, type='basic')
        elif isinstance(gau_bum_pa, GaussianBumpParameters):
            self.gau_bum_pa = gau_bum_pa
        else:
            raise TypeError('Type of blo_des_pa must be BlockDesign Parameters!')

        if batch_size is None:
            self.batch_size = 128
        else:
            self.batch_size = batch_size

    def __call__(self):
        batch_size = self.batch_size
        time_length = self.gau_bum_pa.time_length
        step_size = self.gau_bum_pa.step_size
        mu = self.gau_bum_pa.mu
        sigma = self.gau_bum_pa.sigma
        cycle = self.gau_bum_pa.cycle
        sequence_length = round(time_length / step_size)
        neural_activity = np.zeros(shape=[sequence_length, batch_size])
        for j in range(batch_size):
            for k in range(cycle):
                for i in range(sequence_length):
                    neural_activity[i, j] += sigma[k] * math.exp(-math.pow(i*step_size-mu[k], 2)/4)

        return neural_activity


class RandomGaussianBump(NeuralActivity):

    def __init__(self, gau_bum_pa=None, batch_size=None):
        if gau_bum_pa is None:
            self.gau_bum_pa = GaussianBumpParameters(batch_size=batch_size, type='random')
        elif isinstance(gau_bum_pa, GaussianBumpParameters):
            self.gau_bum_pa = gau_bum_pa
        else:
            raise TypeError('Type of blo_des_pa must be BlockDesign Parameters!')

        if batch_size is None:
            self.batch_size = 128
        else:
            self.batch_size = batch_size

    def __call__(self):
        batch_size = self.batch_size
        time_length = self.gau_bum_pa.time_length
        step_size = self.gau_bum_pa.step_size
        mu = self.gau_bum_pa.mu
        sigma = self.gau_bum_pa.sigma
        cycle = self.gau_bum_pa.cycle
        sequence_length = round(time_length / step_size * cycle)
        neural_activity = np.zeros(shape=[sequence_length, batch_size])
        for i in range(sequence_length):
            for j in range(batch_size):
                for k in range(cycle):
                    neural_activity[i, j] += sigma[k] * math.exp(math.pow(i*step_size-mu[k], 2)/12)

        return neural_activity

class GaussianBumpParameters:
    # Cycle Num
    cycle , time_length, step_size,  mu, sigma = AllParameters().get_gaussian_bump_parameters()

    def __init__(self, batch_size=128, type='basic'):
        if type is 'basic':
            pass
        elif type is 'random':
            self.mu = np.zeros(shape=[batch_size])
            for i in range(batch_size):
                self.mu[i] = 25 + np.random.randint(low=-8, high=8)