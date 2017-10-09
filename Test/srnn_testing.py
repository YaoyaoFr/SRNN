import tensorflow as tf
import numpy as np
from data_generate.data_generation import DataGeneration
from rnn_structure.RNNStructure import SRNNStructure
import rnn_structure.functions as fun
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import scipy.io as sio
from tensorflow.python.ops.seq2seq import *
from all_parameters import AllParameters
from log.log import Log
from data_generate.data import Data


# SRNN Parameters
batch_size = AllParameters().get_testing_parameters()

# Saver
with tf.Session() as sess:
    srnn_structure = SRNNStructure(sess=sess)
    log = Log(sess=sess)

    saver = tf.train.Saver(tf.all_variables())
    init_op = tf.global_variables_initializer()
    log.restore(saver=saver)

    test_data = Data(hidden_state=srnn_structure.zero_state(batch_size=batch_size))

    # Generate New Noisy Data
    test_data.read_data()

    #  Testing
    test_data = srnn_structure.test_time_sequence(test_data, is_noisy=False)

    test_data.post_calculation(cal_los=True)
    log.write_error(test_data.los, 0, test_data.lr, False)
    log.save_mat(test_data, 109999, False, save_name='random_parameter_0')

