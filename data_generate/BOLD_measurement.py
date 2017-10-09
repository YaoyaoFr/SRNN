import numpy as np
from data_generate.generation_parameters import ExperimentParameters, BiophysicalParameters


class BOLDMeasurement:

    def __init__(self, batch_size=128, bio_pa=None, exp_pa=None):
        self.batch_size = batch_size
        if bio_pa is None:
            bio_pa = BiophysicalParameters()
        self.bio_pa = bio_pa
        if exp_pa is None:
            exp_pa = ExperimentParameters()
        self.exp_pa = exp_pa

    def BOLD_generation(self, state, is_exp=True):
        """
        Generate BOLD measurement
        :param state: An ndarry with the shape of [time_sequence_length, batch_size, state_size]
        :return: measured BOLD signal
        """

        phi = self.bio_pa.phi
        V0 = self.bio_pa.V0
        TE = 0.04
        r0 = 25
        nu0 = 40.3
        epsi = np.exp(0.02);

        k1 = 4.3 * nu0 * phi * TE
        k2 = epsi * r0 * phi * TE;
        k3 = 1 - epsi
        shape = np.shape(state)
        # 指数化状态变量
        if is_exp:
            state = np.exp(state)
        if len(shape) >= 3:
            [time_sequence_length, batch_size, state_size] = shape
            shape = [time_sequence_length, batch_size, 1]
            if state_size is 4:
                BOLD = V0 * (k1 * (1 - state[:, :, 3]) + k2 * (1 - state[:, :, 3] / state[:, :, 2]) + k3 * (1 - state[:, :, 2]))
            elif state_size is 2:
                BOLD = V0 * (k1 * (1 - state[:, :, 1]) + k2 * (1 - state[:, :, 1] / state[:, :, 0]) + k3 * (1 - state[:, :, 0]))

        elif len(shape) >= 2:
            [batch_size, state_size] = shape
            shape = [batch_size, 1]
            if state_size is 4:
                BOLD = V0 * (k1 * (1 - state[:, 3]) + k2 * (1 - state[:, 3] / state[:, 2]) + k3 * (1 - state[:, 2]))
            elif state_size is 2:
                BOLD = V0 * (k1 * (1 - state[:, 1]) + k2 * (1 - state[:, 1] / state[:, 0]) + k3 * (1 - state[:, 0]))

        BOLD = np.reshape(BOLD, newshape=shape)
        BOLD = np.nan_to_num(BOLD)
        return BOLD

    def BOLD_noisy(self, BOLD, exp_pa=None):
        """

        :param BOLD: A time series with shape of [time_sequence_length, batch_size, BOLD_size]
        :return:
        """
        if exp_pa is None:
            exp_pa = self.exp_pa

        observation_noise = exp_pa.observation_noise
        shape = np.shape(BOLD)

        noisy_BOLD = BOLD + np.random.normal(loc=0, scale=observation_noise, size=shape)
        return noisy_BOLD

    def BOLD_downsampling(self, BOLD):
        """
        Downsampling with the BOLD signal
        :param BOLD: A time series with the shape of [time_sequence_length, batch_size, BOLD_size(1 usually)]
        :return: A time series with the shape of [downsampled_sequence_length, batch_size, BOLD_size]
        """

        Vg = self.exp_pa.Vg
        TR = self.exp_pa.TR

        [time_sequence_length, batch_size, BOLD_size] = np.shape(BOLD)

        downsample_interval = Vg * TR
        downsample_index = np.arange(0, time_sequence_length, downsample_interval)

        downsampled_BOLD = BOLD[downsample_index, :, :]
        return downsampled_BOLD

    def BOLD_interpolate(self, downsampled_BOLD):
        """
        Interpolate with the downsampled BOLD signal
        :param downsampled_BOLD: A time series with the shape of [downsampled_sequence_length, batch_size, BOLD_size]
        :return: A time series with the shape of [time_sequence_length, batch_size, BOLD_size]
        """

        Vg = self.exp_pa.Vg
        TR = self.exp_pa.TR

        [downsampled_sequence_length, batch_size, BOLD_size] = np.shape(downsampled_BOLD)

        interpolate_index = np.arange(0, downsampled_sequence_length, 1 / (Vg * TR))
        downsampled_index = np.arange(0, downsampled_sequence_length)
        interpolate_BOLD = np.zeros(shape=[len(interpolate_index), batch_size, BOLD_size])
        for i in range(batch_size):
            interpolate_BOLD[:, i, 0] = np.interp(interpolate_index, downsampled_index,
                                                  np.reshape(downsampled_BOLD[:, i, 0], [downsampled_sequence_length, ]))
        return interpolate_BOLD

    def BOLD_observation(self, states, is_noisy=True, is_downsampling=True, is_interpolate=True, exp_pa=None):
        if states is None:
            raise TypeError('states cannot be empty!')

        BOLD = self.BOLD_generation(state=states)
        if is_noisy:
            noisy_BOLD = self.BOLD_noisy(BOLD)
        else:
            noisy_BOLD = BOLD

        if is_downsampling:
            downsampled_BOLD = self.BOLD_downsampling(noisy_BOLD)
        else:
            downsampled_BOLD = noisy_BOLD

        if is_interpolate:
            interpolated_BOLD = self.BOLD_interpolate(downsampled_BOLD)
        else:
            interpolated_BOLD = downsampled_BOLD

        return BOLD, interpolated_BOLD

