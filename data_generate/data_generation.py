from data_generate.BOLD_measurement import *
from data_generate.hemodynamic_model import *
from data_generate.neural_activity import *
from data_generate.data import Data
from all_parameters import AllParameters
import scipy.io as sio
import matlab.engine


class DataGeneration:

    def data_generation_python(self, data=None):
        """
        :param data:
        :return:
        """
        bio_pa = BiophysicalParameters()
        data_type, random_level, batch_size = AllParameters().get_generation_parameters()

        if data_type is None or random_level is None:
            raise TypeError('The type of data and level of random must be specific.')

        if data_type is 'block':
            if random_level is 'basic':
                neural_activity = BasicBlockDesign(batch_size=batch_size)
            elif random_level is 'random':
                neural_activity = RandomBlockDesign(batch_size=batch_size)
        elif data_type is 'gaussian':
            if random_level is 'basic':
                neural_activity = BasicGaussianBump(batch_size=batch_size)
            elif random_level is 'random':
                neural_activity = RandomGaussianBump(batch_size=batch_size)

        neural_activity = neural_activity()

        states = HemodynamicModel(bio_pa=bio_pa, batch_size=batch_size).\
            dynamic_hemodynamic_odeint(neural=neural_activity)

        raw_BOLD, BOLD = BOLDMeasurement(bio_pa=bio_pa, batch_size=batch_size).BOLD_observation(states=states)
        if data is None:
            data = Data(neural=neural_activity, hemodynamic_state=states, raw_BOLD=raw_BOLD, noisy_interpolated_BOLD=BOLD)
        else:
            data = data.set(neural=neural_activity, hemodynamic_state=states, raw_BOLD=raw_BOLD, noisy_interpolated_BOLD=BOLD)
        return data

    def read_data(self, data=None):
        data_path = AllParameters().data_path
        train_data_size = AllParameters().train_data_size
        test_data_size = AllParameters().test_data_size

        d = sio.loadmat(data_path)
        raw_BOLD = d['BOLD']
        neural = d['neural']
        hemodynamic_state = d['state']
        [time_sequence_length, batch_size] = np.shape(raw_BOLD)
        neural = np.reshape(neural, newshape=[time_sequence_length, batch_size])
        hemodynamic_state = np.reshape(hemodynamic_state, [time_sequence_length, batch_size, 4])
        raw_BOLD = np.reshape(raw_BOLD, newshape=[time_sequence_length, batch_size, 1])
        noisy_interpolated_BOLD = BOLDMeasurement().BOLD_noisy(BOLD=raw_BOLD)

        if data is None:
            data = Data(neural=neural, hemodynamic_state=hemodynamic_state, raw_BOLD=raw_BOLD, noisy_interpolated_BOLD=noisy_interpolated_BOLD)
        else:
            data.set(neural=neural, hemodynamic_state=hemodynamic_state, raw_BOLD=raw_BOLD, noisy_interpolated_BOLD=noisy_interpolated_BOLD)
        return data

    def data_generation_matlab(self, data=None, inputs=None, eng=None):
        if inputs is None:
            inputs = list()
            inputs.append([[7, 25, 34, 56], [1, 0.7, 0.9, 0.2]])
            inputs.append([[7, 25, 34, 56], [0.7, 1, 0.2, 0.9]])

        if eng is None:
            eng = matlab.engine.start_matlab()

        inputs = matlab.double(inputs)
        data_temp = eng.generation(inputs)
        time_sequence_length = int(data_temp['N'])
        raw_BOLD = np.reshape(data_temp['BOLD'], newshape=[time_sequence_length, len(inputs), 1])
        hemodynamic_state = np.reshape(data_temp['state'], newshape=[time_sequence_length, len(inputs), 4])
        neural = np.reshape(data_temp['neural'], newshape=[time_sequence_length, len(inputs), 1])

        if data is None:
            data = Data(neural=neural, hemodynamic_state=hemodynamic_state, raw_BOLD=raw_BOLD, noisy_interpolated_BOLD=raw_BOLD)
        else:
            data.concatenate(Data(neural=neural, hemodynamic_state=hemodynamic_state, raw_BOLD=raw_BOLD, noisy_interpolated_BOLD=raw_BOLD))

        return data

