import tensorflow as tf
import numpy as np
from data_generate.data_generation import DataGeneration
from data_generate.BOLD_measurement import BOLDMeasurement
from tensorflow.python.ops.rnn_cell import BasicLSTMCell, BasicRNNCell
from rnn_structure.RNNStructure import RNNStructure, RNNStructureParameters
import rnn_structure.functions as fun
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import os


def RNN_structure(input_size, output_size, batch_size, hidden_size, cell_type='LSTM'):
    #   Recurrent Neural Network Cell
    if cell_type is 'LSTM':
        cell = BasicLSTMCell(hidden_size, state_is_tuple=True)
    elif cell_type is 'RNN':
        cell = BasicRNNCell(hidden_size)

    # Place holder
    input_place = tf.placeholder(dtype=tf.float64, shape=[batch_size, input_size],
                                      name='Placeholder/Input')
    output_place = tf.placeholder(dtype=tf.float64, shape=[batch_size, output_size],
                                       name='Placeholder/Output')

    #   Network Variable
    hidden_state = cell.zero_state(batch_size=batch_size, dtype=tf.float64)

    #   Feedforward Process
    value, hidden_state_c = cell(input_place, hidden_state)
    return cell, input_place, output_place, hidden_state_c, hidden_state, value


def RNN_Block_filter():
    # Programe parameters
    need_init = True
    batch_size = 16
    plan_path = 'F:/Data/Exp_result/2017.3.8/block_design'
    if not os.path.exists(plan_path):
        os.makedirs(plan_path)
    incorrect_path = plan_path+'\\incorrect.txt'

    # Size of variable vectors
    BOLD_size = 1
    v_size = 1
    q_size = 1
    f_size = 1
    s_size = 1
    u_size = 1
    v_q_size = 2

    # RNN CELLs
    vq_hidden_size = 20
    f_hidden_size = 15
    s_hidden_size = 15
    u_hidden_size = 10
    state_size = 4
    neural_size = 1

    input_sizes = [BOLD_size, vq_hidden_size, f_hidden_size]
    hidden_sizes = [vq_hidden_size, f_hidden_size, s_hidden_size]
    output_sizes = [v_q_size, f_size, s_size]
    var_scopes = ['v_q', 'f', 's']

    Pas=list()
    for i in range(3):
        Pas.append(RNNStructureParameters(input_size=input_sizes[i], output_size=output_sizes[i],
                                          hidden_size=hidden_sizes[i], batch_size=batch_size, var_scope=var_scopes[i]))

    [v_q_structure, f_structure, s_structure] = [RNNStructure(Pa=Pa) for Pa in Pas]

    # Saver
    saver = tf.train.Saver(tf.all_variables())
    init_op = tf.initialize_all_variables()
    restore_epoch = 0

    with tf.Session() as sess:

        if need_init:
            init = sess.run(init_op)
            file_incorrect = open(incorrect_path, 'w')
            print('Initialize successful.')
        else:
            restore_epoch = 390
            file_incorrect = open(incorrect_path, 'a')
            saver.restore(sess, plan_path+'/model'+str(restore_epoch)+'.ckpt')
            print('Model restored.')
        epoch = restore_epoch

        # Choose Mode Type
        test_data_type = 'Block'

        # Train Variable

        while True:
            # Generate New Noisy Data
            [block_neural, block_hemodynamic_state, block_raw_BOLD, block_noisy_interpolated_BOLD] = \
                DataGeneration().data_generation(data_type='block', batch_size=batch_size)
            [bump_neural, bump_hemodynamic_state, bump_BOLD, _] = \
                DataGeneration().data_generation(data_type='gaussian', batch_size=batch_size)

            block_sequence_length = len(block_neural)
            bump_sequence_length = len(bump_neural)

            if test_data_type is 'Block':
                [test_neural, test_hemodynamic_state, test_raw_BOLD, test_noisy_interpolated_BOLD] = \
                    DataGeneration().data_generation(data_type='block', batch_size=batch_size)
                test_sequence_length = block_sequence_length
            elif test_data_type is 'Bump':
                [test_neural, test_hemodynamic_state, test_noisy_interpolated_BOLD, _] = \
                    DataGeneration().data_generation(data_type='gaussian', batch_size=batch_size)
                test_sequence_length = bump_sequence_length

            #   Train by Bump Data
            bump_los = np.zeros([bump_sequence_length, 4])
            bump_pre_neural = np.zeros([np.shape(bump_hemodynamic_state)[0], np.shape(bump_hemodynamic_state)[1], 1])
            bump_pre_hemodynamic_state = np.zeros(np.shape(bump_hemodynamic_state))
            bump_pre_BOLD = np.zeros([bump_sequence_length, batch_size, BOLD_size])

            vq_bump_state = v_q_structure.zero_state(sess)
            f_bump_state = f_structure.zero_state(sess)
            s_bump_state = s_structure.zero_state(sess)

            for i in range(bump_sequence_length):
                output_state = fun.gen_output_state(bump_neural, bump_hemodynamic_state, batch_size=batch_size, i=i)

                # V and Q
                q_val, vq_pre, bump_los[i, 2:4], vq_bump_state, lr = v_q_structure.train_time_point(
                    sess=sess, input=np.reshape(bump_BOLD[i, :, 0], [batch_size, BOLD_size]),
                    output=output_state[:, 2:4], state=vq_bump_state)

                # F
                _, f_pre, bump_los[i, 1], f_bump_state, _ = f_structure.train_time_point(
                    sess=sess, input=vq_bump_state.h, output=np.reshape(output_state[:, 1], [batch_size, 1]),
                    state=f_bump_state
                )

                # S
                _, s_pre, bump_los[i, 0], s_bump_state, _ = s_structure.train_time_point(
                    sess=sess, input=f_bump_state.h, output=np.reshape(output_state[:, 0], [batch_size, 1]),
                    state= s_bump_state
                )
                output = np.concatenate((s_pre, f_pre, vq_pre), axis=1)
                bump_pre_BOLD[i] = BOLDMeasurement().BOLD_generation(output)
                bump_pre_neural, bump_pre_hemodynamic_state = \
                    fun.reset_output_state(output, bump_pre_neural, bump_pre_hemodynamic_state, i, batch_size)

            #   Train by Block Data
            block_los = np.zeros([block_sequence_length, 4])
            block_pre_neural = np.zeros([block_sequence_length, batch_size, neural_size])
            block_pre_hemodynamic_state = np.zeros([block_sequence_length, batch_size, state_size])
            block_pre_BOLD = np.zeros([block_sequence_length, batch_size, BOLD_size])

            vq_block_state = v_q_structure.zero_state(sess)
            f_block_state = f_structure.zero_state(sess)
            s_block_state = s_structure.zero_state(sess)

            for i in range(block_sequence_length):
                output_state = fun.gen_output_state(block_neural, block_hemodynamic_state, batch_size=batch_size, i=i)
                # V and Q
                vq_val, vq_pre, block_los[i, 2:4], vq_block_state, _ = v_q_structure.train_time_point(
                    sess=sess, input= np.reshape(block_noisy_interpolated_BOLD[i, :, 0], [batch_size, BOLD_size]),
                    output=output_state[:, 2:4], state=vq_block_state
                )

                # F
                _, f_pre, block_los[i, 1], f_block_state, _ = f_structure.train_time_point(
                    sess=sess, input=vq_block_state.h, output=np.reshape(output_state[:, 1], [batch_size, 1]),
                    state=f_block_state
                )

                # S
                _, s_pre, block_los[i, 0], s_block_state, _ = s_structure.train_time_point(
                    sess=sess, input=f_block_state.h, output=np.reshape(output_state[:, 0], [batch_size, 1]),
                    state=s_block_state
                )

                output = np.concatenate((s_pre, f_pre, vq_pre), axis=1)
                block_pre_BOLD[i] = BOLDMeasurement().BOLD_generation(output)
                block_pre_neural, block_pre_hemodynamic_state = \
                    fun.reset_output_state(output, block_pre_neural, block_pre_hemodynamic_state, i, batch_size)

            #test_predicted_neural = fun.move_average(test_predicted_neural, step_size=5)

            #   Test by Test Data Type
            test_los = np.zeros([test_sequence_length, 4])
            test_pre_neural = np.zeros([test_sequence_length, batch_size, neural_size])
            test_pre_hemodynamic_state = np.zeros([test_sequence_length, batch_size, state_size])
            test_pre_BOLD = np.zeros([test_sequence_length, batch_size, BOLD_size])

            vq_test_state = v_q_structure.zero_state(sess)
            f_test_state = f_structure.zero_state(sess)
            s_test_state = s_structure.zero_state(sess)

            for i in range(test_sequence_length):
                output_state = fun.gen_output_state(test_neural, test_hemodynamic_state, batch_size=batch_size,
                                                    i=i)
                # V and Q
                vq_val, vq_pre, test_los[i, 2:4], vq_test_state, _= \
                    v_q_structure.train_time_point(
                        sess=sess,
                        input = np.reshape(test_noisy_interpolated_BOLD[i, :, 0], [batch_size, BOLD_size]),
                        state=vq_test_state,
                        output = output_state[:, 2:4]
                    )

                # F
                _, f_pre, test_los[i, 1], f_test_state, _ = f_structure.train_time_point(
                        sess=sess,
                        state= f_test_state,
                        input= vq_test_state.h,
                        output=np.reshape(output_state[:, 1], [batch_size, 1])
                )

                # S
                _, s_pre, test_los[i, 0], s_test_state, _ =s_structure.train_time_point(
                        sess=sess,
                        state=s_test_state,
                        input=f_test_state.h,
                        output=np.reshape(output_state[:, 0], [batch_size, 1])
                    )

                output = np.concatenate((s_pre, f_pre, vq_pre), axis=1)
                test_pre_BOLD[i] = BOLDMeasurement().BOLD_generation(output)
                test_pre_neural, test_pre_hemodynamic_state = \
                    fun.reset_output_state(output, test_pre_neural, test_pre_hemodynamic_state, i, batch_size)


            # Show data by Show Type

            block_los = np.mean(block_los, 0)
            block_error_string = 'Block:  Epoch - {:2d}  MSE:  s - {:10f}  f - {:10f}  v - {:10f} q - {:10f}  lr - {:10f} ' \
                .format(epoch + 1, block_los[0], block_los[1], block_los[2], block_los[3], lr)
            print(block_error_string)

            bump_los = np.mean(bump_los, 0)
            bump_error_string = 'Bump:   Epoch - {:2d}  MSE:  s - {:10f}  f - {:10f}  v - {:10f} q - {:10f}  lr - {:10f} ' \
                .format(epoch + 1, bump_los[0], bump_los[1], bump_los[2], bump_los[3], lr)
            print(bump_error_string)

            test_los = np.mean(test_los, 0)
            test_error_string = 'Test:   Epoch - {:2d}  MSE:  s - {:10f}  f - {:10f}  v - {:10f} q - {:10f}  lr - {:10f} ' \
                .format(epoch + 1, test_los[0], test_los[1], test_los[2], test_los[3], lr)
            print(test_error_string)
            print('\n')


            yield [test_pre_hemodynamic_state, test_hemodynamic_state], [test_pre_BOLD, test_noisy_interpolated_BOLD]

            # Writer Accuracy Info
            file_incorrect = open(incorrect_path, 'a')
            file_incorrect.write(block_error_string+'\n')
            file_incorrect.write(bump_error_string+'\n')
            file_incorrect.close()


            # Saveing Neural Network Weights
            if (epoch+1) % 50 == 0:
                save_path = plan_path+'/model'+str(epoch+1)+'.ckpt'
                save_path = saver.save(sess, save_path)
                print('Model saved in file', save_path)

            # Iterate Epoch
            epoch += 1

tdata = []
fig = plt.figure()
subplot_name = ['BOLD', 's', 'f', 'v', 'q']
axs = list()
lines = list()
for i in range(len(subplot_name)):
    ax = fig.add_subplot(2, 3, i+1)
    ax.set_xlabel('t')
    ax.set_ylabel(subplot_name[i])
    axs.append(ax)
    if i < 6:
        lines.append([axs[i].plot([], [], lw=2)[0] for _ in range(2)])
    else:
        lines.append(axs[i].plot([], [], lw=2))


def init():
    for ax, line in zip(axs, lines):
        ax.set_ylim(-0.1, 0.1)
        ax.set_xlim(0, 100)
        ax.grid()
        for l in line:
            l.set_data([], [])

    return lines,


def run(data):
    state, BOLD = data
    # Data Pre-processing
    # state
    state = np.array(state)
    s = state[:, :, 0, 0]
    f = state[:, :, 0, 1]
    v = state[:, :, 0, 2]
    q = state[:, :, 0, 3]
    del state

    # BOLD
    BOLD = np.array(BOLD)
    BOLD = BOLD[:, :, 0]

    datas = [BOLD, s, f, v, q]

    # Neural
    for ax, line, data, type in zip(axs, lines, datas, subplot_name):
        '''
        if type is 'Accuracy':
            pass
            accuracy = list()
            ax.set_ylim(-0.1, 1.1)
            predicted_neural = data[0]
            true_neural = data[1]
            for pre_n, tru_n in zip(predicted_neural, true_neural):
                n = np.sum(np.round(pre_n) == tru_n)
                accuracy.append(n / np.shape(pre_n)[0])
            if len(accuracy) > max_len:
                del accuracy[0]
            tdata = np.arange(t, t + len(accuracy) * 0.5, 0.5)
            ax.set_xlim(min(tdata) - 10, max(tdata) + 10)
            line[0].set_data(tdata, accuracy)
            ax.figure.canvas.draw()
        else:
        '''
        # Axis Range
        min_value = np.min(data)
        min_value -= min_value / 10

        max_value = np.max(data)
        max_value += max_value / 10

        ax.set_ylim(min_value, max_value)

        """
        ax.set_xlim(0, inp_pa.time_length)
        tdata = np.arange(0, inp_pa.time_length, inp_pa.step_size)
        """
        time_length = 59
        ax.set_xlim(0, time_length)
        tdata = np.arange(0, time_length + 0.1, 0.1)
        for l, d in zip(line, data):
            l.set_data(tdata, d[0:len(tdata)])
        ax.figure.canvas.draw()
    return lines,

ani = animation.FuncAnimation(fig, run, RNN_Block_filter, blit=False,
                              repeat=False, init_func=init)
plt.show()
