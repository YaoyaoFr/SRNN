import numpy as np
from data_generate.hemodynamic_model import BiophysicalParameters
from data_generate.experiment_parameters import ExperimentParameters


def gen_BOLD(state):
    """
    Generate BOLD measurement
    :param state: An ndarry with the shape of [time_sequence_length, batch_size, state_size]
    :return: measured BOLD signal
    """

    phi = BiophysicalParameters().phi

    [time_sequence_length, batch_size, state_size] = np.shape(state)

    V0 = 0.04
    k1 = 7 * phi
    k2 = 2
    k3 = 2 * phi - 0.2

    if state_size is 4:
        BOLD = V0 * (k1 * (1 - state[:, :, 3]) + k2 * (1 - state[:, :, 3] / state[:, :, 2]) + k3 * (1 - state[:, :, 2]))
    elif state_size is 2:
        BOLD = V0 * (k1 * (1 - state[:, :, 1]) + k2 * (1 - state[:, :, 1] / state[:, :, 0]) + k3 * (1 - state[:, :, 0]))
    BOLD = np.reshape(BOLD, [time_sequence_length, batch_size, 1])

    return BOLD

