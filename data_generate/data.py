from data_generate.BOLD_measurement import BOLDMeasurement
from all_parameters import AllParameters
import rnn_structure.functions as fun
import scipy.io as sio
import numpy as np


class Data:

    pre_MA_neural = None
    pre_hemodynamic_state = None
    hemodynamic_state = None
    pre_BOLD = None
    raw_BOLD = None
    noisy_interpolated_BOLD = None
    pre_neural = None
    neural = None
    hidden_state = None
    los = None
    lr = None
    time_sequence_length = None

    def __init__(self, pre_MA_neural = None, pre_hemodynamic_state=None, hemodynamic_state=None, pre_BOLD=None,
                 raw_BOLD=None, noisy_interpolated_BOLD=None, pre_neural=None, neural=None, hidden_state=None,
                 los=None, lr=None, time_sequence_length=None):
        # All Data Shape = [time_sequence_length, batch_size, data_size]
        self.pre_MA_neural = pre_MA_neural
        self.pre_hemodynamic_state = pre_hemodynamic_state
        self.hemodynamic_state = hemodynamic_state
        self.pre_BOLD = pre_BOLD
        self.raw_BOLD = raw_BOLD
        self.noisy_interpolated_BOLD = noisy_interpolated_BOLD
        self.pre_neural = pre_neural
        self.neural = neural
        self.hidden_state = hidden_state
        self.los = los
        self.lr = lr
        self.time_sequence_length = time_sequence_length

    def read_data(self, data_path=None):
        if data_path is None:
            data_path = AllParameters().data_path

        data_path = AllParameters().data_path
        train_data_size = AllParameters().train_data_size
        test_data_size = AllParameters().test_data_size

        d = sio.loadmat(data_path)
        raw_BOLD = d['BOLD']
        neural = d['neural']
        hemodynamic_state = d['state']
        [time_sequence_length, batch_size] = np.shape(raw_BOLD)
        self.neural = np.reshape(neural, newshape=[time_sequence_length, batch_size])
        self.hemodynamic_state = np.reshape(hemodynamic_state, [time_sequence_length, batch_size, 4])
        self.raw_BOLD = np.reshape(raw_BOLD, newshape=[time_sequence_length, batch_size, 1])
        self.noisy_interpolated_BOLD = BOLDMeasurement().BOLD_noisy(BOLD=self.raw_BOLD)

    def set(self, pre_MA_neural = None, pre_hemodynamic_state=None, hemodynamic_state=None, pre_BOLD=None,
                 raw_BOLD=None, noisy_interpolated_BOLD=None, pre_neural=None, neural=None, hidden_state=None,
            los=None, lr=None, time_sequence_length=None):
        if pre_MA_neural is not None:
            self.pre_MA_neural = pre_MA_neural
        if pre_hemodynamic_state is not None:
            self.pre_hemodynamic_state = pre_hemodynamic_state
        if hemodynamic_state is not None:
            self.hemodynamic_state = hemodynamic_state
        if pre_BOLD is not None:
            self.pre_BOLD = pre_BOLD
        if raw_BOLD is not None:
            self.raw_BOLD = raw_BOLD
        if noisy_interpolated_BOLD is not None:
            self.noisy_interpolated_BOLD = noisy_interpolated_BOLD
        if pre_neural is not None:
            self.pre_neural = pre_neural
        if neural is not None:
            self.neural = neural
        if hidden_state is not None:
            self.hidden_state = hidden_state
        if los is not None:
            self.los = los
        if lr is not None:
            self.lr = lr
        if time_sequence_length is not None:
            self.time_sequence_length = time_sequence_length

    def post_calculation(self, cal_los=False, is_interpolated=False):
        self.pre_BOLD = BOLDMeasurement().BOLD_generation(state=self.pre_hemodynamic_state, is_exp=True)
        self.pre_neural = fun.cal_neural_activity(hemodynamic_state=self.pre_hemodynamic_state, is_exp=True,
                                                  is_interpolated=is_interpolated)
        self.pre_MA_neural = fun.move_average(self.pre_neural, circle=True)

        if cal_los:
            self.los = np.mean(np.square(self.pre_hemodynamic_state - self.hemodynamic_state[:, self.test_index, :]),
                               axis=1)

        self.time_sequence_length = np.shape(self.neural)[0]

    def concatenate(self, data):
        if not isinstance(data, Data):
            raise TypeError('Input Type Error!')
        if self.neural is not None:
            self.neural = np.concatenate((self.neural, data.neural), axis=1)
        else:
            self.neural = data.neural

        if self.hemodynamic_state is not None:
            self.hemodynamic_state = np.concatenate((self.hemodynamic_state, data.hemodynamic_state), axis=1)
        else:
            self.hemodynamic_state = data.hemodynamic_state

        if self.noisy_interpolated_BOLD is not None:
            self.noisy_interpolated_BOLD = np.concatenate((self.noisy_interpolated_BOLD, data.noisy_interpolated_BOLD), axis=1)
        else:
            self.noisy_interpolated_BOLD = data.noisy_interpolated_BOLD

        if self.raw_BOLD is not None:
            self.raw_BOLD = np.concatenate((self.raw_BOLD, data.raw_BOLD), axis=1)
        else:
            self.raw_BOLD = data.raw_BOLD

    def get_result(self, batch_size, index=None):
        if index is None:
            index = self.train_index
        BOLD = np.concatenate((np.reshape(self.raw_BOLD[:, index, :], newshape=[1, self.time_sequence_length, batch_size, 1]),
                               np.reshape(self.pre_BOLD, newshape=[1, self.time_sequence_length, batch_size, 1])),
                              axis=0)

        state = np.concatenate((np.reshape(self.hemodynamic_state[:, index, :],
                                           newshape=[1, self.time_sequence_length, batch_size, 4]),
                               np.reshape(self.pre_hemodynamic_state,
                                          newshape=[1, self.time_sequence_length, batch_size, 4])),
                               axis=0)

        neural = np.concatenate((np.reshape(self.neural[:, index], newshape=[1, self.time_sequence_length, batch_size, 1]),
                                 np.reshape(self.pre_MA_neural, newshape=[1, self.time_sequence_length, batch_size, 1])),
                                axis=0)

        [s, f, v, q] = [np.reshape(state[:, :, :, i], newshape=[2, self.time_sequence_length, batch_size, 1])
                        for i in range(4)]

        return [neural, s, f, v, q, BOLD]

    def preprocess_data(self, output=None):
        if output is None:
            output = self.hemodynamic_state

        [time_sequence_length, batch_size, data_size] = np.shape(output)
        preprocessed_data = np.zeros([time_sequence_length + 2, batch_size, data_size])
        preprocessed_data[2:time_sequence_length+2, :, 0] = output[:, :, 0]
        preprocessed_data[1:time_sequence_length+1, :, 1] = output[:, :, 1]
        preprocessed_data[0:time_sequence_length, :, 2:4] = output[:, :, 2:4]
        return preprocessed_data[0:time_sequence_length, :, :]

    def postprocess_data(self, output):
        [time_sequence_length, batch_size, data_size] = np.shape(output)
        postprocessed_data = np.zeros([time_sequence_length, batch_size, data_size])
        postprocessed_data[0:time_sequence_length-2, :, 0] = output[2:time_sequence_length, :, 0]
        postprocessed_data[0:time_sequence_length-1, :, 1] = output[1:time_sequence_length, :, 1]
        postprocessed_data[:, :, 2:4] = output[0:time_sequence_length, :, 2:4]
        return postprocessed_data