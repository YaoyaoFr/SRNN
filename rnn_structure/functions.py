import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy import interpolate
from scipy.stats import norm
from data_generate.generation_parameters import BiophysicalParameters, ExperimentParameters
import scipy.io as sio
from data_generate.BOLD_measurement import BOLDMeasurement
from all_parameters import AllParameters

def move_average(data=None, step_size=5, circle=False):
    axis = 0
    if data is None:
        raise TypeError('Data is Empty!')

    shape = np.shape(data)
    if axis > len(shape):
        raise TypeError('Axis out of Bound!')

    if not isinstance(step_size, int):
        raise TypeError('Step Size must be Integer!')

    if np.mod(step_size, 2) != 1:
        raise TypeError('Step Size must be Odd!')

    move_range = int((step_size-1)/2)
    new_data = np.zeros(shape)
    for i in range(shape[axis]):
        for j in range(i-move_range, i+move_range+1):
            if 0 <= j < shape[axis]:
                new_data[i] += data[j]
    new_data /= step_size
    return new_data


def cal_scale(loc=0, confidence_level=0.95, confidence_bound=0.25, confidence_interval=None, edge_type='bilateral'):
    """
    Caculate the standard deviation by mean value, confidence level and confidence interval
    :param loc:
    :param confidence_level:
    :param confidence_bound:
    :param confidence_interval:
    :param edge_type:
    :return: Standard deviation
    """
    if edge_type == 'bilateral':
        confidence_edge = (1 + confidence_level) / 2
    elif edge_type == 'unilateral':
        confidence_edge = confidence_level

    if confidence_interval is not None:
        loc = np.mean(confidence_interval)

    percent_point = norm.ppf(confidence_edge, loc)
    scale = np.abs(confidence_bound - loc) / percent_point
    return scale


def cal_FWTM(loc=0, scale=1, time_threshold=100, time_length=None, step_size=0.1, percent_of_apmlitude=0.1):
    """
    Caculate Full-Width at Tenth Maximum
    :param loc:
    :param scale:
    :param time_threshold: millisecond
    :param time_length: The edge of Width
    :param step_size:
    :param percent_of_apmlitude:
    :return:The Interval of Full-Width at Tenth Maximum
    """
    if time_length is None:
        raise TypeError('Time Length is None!')
    h = 1 / (np.sqrt(2 * np.pi) * scale)
    h *= percent_of_apmlitude
    HWTM = np.sqrt(-2 * np.square(scale) * np.log(h * scale * np.sqrt(2 * np.pi)))
    left = int(round(max(loc - time_threshold, loc - HWTM, 0) / step_size))
    right = int(round(min(loc + time_threshold, loc + HWTM, time_length) / step_size))
    FWTM = np.arange(left, right)
    return FWTM


def gen_output_state(neural=None, hemodynamic_state=None, i=None, batch_size=128):
    state_size = 4
    neural_size = 1
    output_state = np.zeros([batch_size, state_size])
    """
    if i >= 3:
        output_state[:, 0] = neural[i - 3, :, 0]
        """
    if i >= 2:
        output_state[:, 0] = hemodynamic_state[i - 2, :, 0]
    if i >= 1:
        output_state[:, 1] = hemodynamic_state[i - 1, :, 1]
    output_state[:, 2:4] = hemodynamic_state[i, :, 2:4]
    return output_state


def reset_output_state(output_state=None, neural=None, hemodynamic_state=None, i=None,batch_size=128):
    """
    if i >= 3:
         neural[i - 3, :, 0] = output_state[:, 0]
         """
    if i >= 2:
        hemodynamic_state[i - 2, :, 0] = output_state[:, 0]
    if i >= 1:
        hemodynamic_state[i - 1, :, 1] = output_state[:, 1]
    hemodynamic_state[i, :, 2:4] = output_state[:, 2:4]
    return neural, hemodynamic_state


def cal_neural_activity(hemodynamic_state=None, bio_pa=None, exp_pa=None, is_exp=False, is_interpolated=False):
    if bio_pa is None:
        bio_pa = BiophysicalParameters
    else:
        bio_pa = bio_pa

    if exp_pa is None:
        exp_pa = ExperimentParameters
    else:
        exp_pa = exp_pa

    s = hemodynamic_state[:, :, 0]
    if is_exp:
        f = np.exp(hemodynamic_state[:, :, 1])

    [time_sequence_length, batch_size, state_size] = np.shape(hemodynamic_state)

    delta = exp_pa.TR

    if is_interpolated:
        Vg = 50
        TR = AllParameters().TR
        interpolate_index = np.arange(0, time_sequence_length, 1 / (Vg * TR))
        downsampled_index = np.arange(0, time_sequence_length)
        interpolated_s = np.zeros(shape=[len(interpolate_index), batch_size])
        interpolated_f = np.zeros(shape=[len(interpolate_index), batch_size])
        for i in range(batch_size):
            interpolated_s[:, i] = np.interp(interpolate_index, downsampled_index,
                                                  np.reshape(s[:, i],
                                                             [time_sequence_length, ]))
            interpolated_f[:, i] = np.interp(interpolate_index, downsampled_index,
                                             np.reshape(f[:, i],
                                                        [time_sequence_length, ]))
        time_sequence_length = len(interpolate_index)
        delta = TR / Vg
        s = interpolated_s
        f = interpolated_f

    neural_activity = np.zeros([time_sequence_length, batch_size])
    for i in np.arange(start=0, stop=time_sequence_length-1):
        for j in range(batch_size):
            neural_activity[i, j] = ((s[i+1, j] - s[i, j]) / delta +
                                     s[i, j] * bio_pa.kappa +
                                        (f[i, j]-1)*bio_pa.gamma) \
                                       / bio_pa.epsilon

    neural_activity = np.reshape(neural_activity, newshape=[time_sequence_length, batch_size, 1])

    if is_interpolated:
        downsampled_index = downsampled_index * Vg * TR
        neural_activity = neural_activity[downsampled_index, :]

    return neural_activity


def read_data(path=None):
    raw_BOLD = sio.loadmat(path)['Y']
    exp_pa = ExperimentParameters()
    exp_pa.TR=2
    raw_BOLD = np.reshape(raw_BOLD, [np.shape(raw_BOLD)[0], 1, 1])
    raw_BOLD = BOLDMeasurement(batch_size=1, exp_pa=exp_pa).BOLD_interpolate(raw_BOLD)
    return raw_BOLD