import numpy as np
import tensorflow as tf
from scipy.stats import norm


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
        stimulu_interval = self.blo_des_pa.stimulu_interval
        step_size = self.blo_des_pa.step_size
        cycle_length = round((stimulu_interval + stimulu_last) / step_size)
        neural_activity = np.zeros(shape=[cycles * cycle_length, self.batch_size])
        for cycle in range(cycles):
            for index in np.arange(0, round(stimulu_last / step_size)):
                neural_activity[cycle * cycle_length + index, :] = 1

        return neural_activity


class BlockDesignParameters:
    cycles = 5
    step_size = 0.1
    stimulu_last = 30
    stimulu_interval = 30

    def __init__(self, blo_des_pa):
        self.cycles = blo_des_pa.cycles
        self.step_size = blo_des_pa.step_size
        self.stimulu_last = blo_des_pa.stimulu_last
        self.stimulu_interval = blo_des_pa.stimulu_interval
        self.downsampling_rate = blo_des_pa.downsampling_rate
        self.batch_size = blo_des_pa.batch_size


class BasicGaussianBump(NeuralActivity):

    def __init__(self, gau_bum_pa=None, batch_size=None):
        if gau_bum_pa is None:
            self.gau_bum_pa = GaussianBumpParameters
        elif isinstance(gau_bum_pa, GaussianBumpParameters):
            self.gau_bum_pa = gau_bum_pa
        else:
            raise TypeError('Type of blo_des_pa must be BlockDesignParameters!')

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

        sequence_length = round(time_length / step_size)
        neural_activity = np.zeros(shape=[sequence_length, batch_size])
        for i in range(sequence_length):
            neural_activity[i, :] = norm.pdf(i * step_size, loc=mu, scale=sigma)

        return neural_activity


class GaussianBumpParameters:
    # Time Length (s)
    time_length = 60
    # Step Size (s)
    step_size = 0.1
    # Mu
    mu = 25
    # Sigma
    sigma = 6

    def __init__(self, inp_pa):
        if isinstance(inp_pa):
            self.time_length = inp_pa.time_length
            self.step_size = inp_pa.step_size
            self.mu = inp_pa.mu
            self.sigma = inp_pa.sigma