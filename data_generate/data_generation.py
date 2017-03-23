from data_generate.BOLD_measurement import *
from data_generate.generation_parameters import *
from data_generate.hemodynamic_model import *
from data_generate.neural_activity import *

class DataGeneration:

    def data_generation(self, data_type='block', random_level='basic', batch_size=None):
        """
        :param data_type: Option:   'block', 'gaussian'
        :param random_level: Option:    'basic', 'random'
        :return:
        """
        if data_type is None or random_level is None:
            raise TypeError('The type of data and level of random must be specific.')

        if data_type is 'block':
            if random_level is 'basic':
                neural_activity = BasicBlockDesign(batch_size=batch_size)
        elif data_type is 'gaussian':
            if random_level is 'basic':
                neural_activity = BasicGaussianBump(batch_size=batch_size)

        neural_activity = neural_activity()

        states = HemodynamicModel(batch_size=batch_size).dynamic_hemodynamic_odeint(neural=neural_activity)

        raw_BOLD, BOLD = BOLDMeasurement(batch_size=batch_size).BOLD_observation(states=states)

        return neural_activity, states, raw_BOLD, BOLD
