import tensorflow as tf
import numpy as np
from data_generate.data_generation import DataGeneration
from rnn_structure.RNNStructure import SRNNStructure
from rnn_structure.RNNStructure import RNNStructureParameters
import rnn_structure.functions as fun
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import scipy.io as sio
from tensorflow.python.ops.seq2seq import *
from all_parameters import AllParameters
from log.log import Log
from data_generate.data import Data


def rnn_filter():
    # SRNN Parameters
    need_init, restore_epoch, read_data, batch_size, input_size, is_train, train_modules, get_data = AllParameters().get_filter_parameters()

    # Saver
    with tf.Session() as sess:
        srnn_structure = SRNNStructure(sess=sess)
        log = Log(sess=sess)

        saver = tf.train.Saver(tf.all_variables())
        init_op = tf.global_variables_initializer()
        epoch = 0
        if need_init:
            sess.run(init_op)
            print('Initialize successful.')
        else:
            log.restore(saver=saver)
            save_path = AllParameters().save_path
            load_path = AllParameters().load_path
            if save_path == load_path:
                epoch = restore_epoch

            for i in range(len(srnn_structure.RNNs)):
                srnn_structure.RNNs[i].lr = 0.5

        train_data = Data(hidden_state=srnn_structure.zero_state(batch_size=batch_size))
        # Generate New Noisy Data
        if get_data == 0:
            train_data = DataGeneration().data_generation_python(train_data)
        elif get_data == 1:
            train_data = DataGeneration().data_generation_matlab(train_data)
        # Read Data
        elif get_data == 2:
            train_data.read_data()
        print('Data generated.')
        test_data = train_data

        # Choose Mode Type
        while epoch < 15000:


            # Hidden State
            train_data.set(hidden_state=srnn_structure.zero_state(batch_size=batch_size))
            test_data.set(hidden_state=srnn_structure.zero_state(batch_size=batch_size))
            # Training
            if is_train:
                train_data = srnn_structure.train_time_sequence(train_data, train_modules=train_modules)
                print()
                train_data.post_calculation()
                log.write_error(train_data.los, epoch, train_data.lr, True)
                log.save_mat(train_data, epoch, True)

            # Saveing Neural Network Weights
            log.save_model(epoch, saver)

            #  Testing
            test_data = srnn_structure.test_time_sequence(test_data)

            test_data.post_calculation(cal_los=True)
            log.write_error(test_data.los, epoch, test_data.lr, False)
            log.save_mat(test_data, epoch, False)

            # Iterate Epoch
            epoch += 1

            yield test_data, test_data.test_index

subplot, subplot_name, batch_nums = AllParameters().get_result_parameters()
fig = plt.figure()

axs = list()
lines = list()
k = 1
for i, name in enumerate(subplot_name):
    for j, batch in enumerate(batch_nums):
        ax = fig.add_subplot(len(subplot_name), len(batch_nums), k)
        ax.set_xlabel('t')
        ax.set_ylabel(name + '-' + str(batch))
        axs.append(ax)
        lines.append([axs[k-1].plot([], [], lw=2)[0] for _ in range(2)])
        k = k + 1


def init():
    for ax, line in zip(axs, lines):
        ax.set_ylim(-0.1, 0.1)
        ax.set_xlim(0, 64)
        ax.grid()
        for l in line:
            l.set_data([], [])

    return lines,


def run(data):
    data, index = data
    batch_size = AllParameters().batch_size
    time_length = data.time_sequence_length

    all_data = data.get_result(batch_size, index=index)
    datas = []
    for i, flag in enumerate(subplot):
        if flag:
            datas.append(all_data[i])

    #
    k = 0
    for i, _ in enumerate(subplot_name):
        for j, _ in enumerate(batch_nums):
            data = datas[i][:, :, j, 0]
            ax = axs[k]

            # Axis Range
            min_value = np.min(data)
            min_value -= min_value / 10
            max_value = np.max(data)
            max_value += max_value / 10
            ax.set_ylim(min_value, max_value)

            tdata = np.arange(0, time_length)

            for l, d in zip(lines[k], data):
                l.set_data(tdata, d[0:len(tdata)])

            ax.figure.canvas.draw()

            k = k +1

    return lines,


ani = animation.FuncAnimation(fig, run, rnn_filter, blit=False,
                              repeat=False, init_func=init)
plt.show()
