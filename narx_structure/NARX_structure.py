import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class NARXStructure:
    output_size = 1

    def __init__(self, Pa=None):
        if Pa is None:
            Pa = NARXParameters()
        self.hidden_num = Pa.hidden_num
        self.input_TDL = Pa.input_TDL
        self.output_TDL = Pa.output_TDL
        self.batch_size = Pa.batch_size

        self.input_size = self.input_TDL + self.output_TDL + 1
        self.output_size = 1
        self.input_place = tf.placeholder(dtype=tf.float64, shape=[self.batch_size, self.input_size],
                                          name='Input_place')
        self.output_place = tf.placeholder(dtype=tf.float64, shape=[self.batch_size, self.output_size], name='Output_place')

        self.hidden_W = tf.Variable(tf.truncated_normal(shape=[self.input_size, self.hidden_num], dtype=tf.float64),
                                  dtype=tf.float64, name='Hidden_weight')
        self.hidden_b = tf.Variable(tf.constant(0.1, shape=[self.hidden_num, ], dtype=tf.float64), dtype=tf.float64,
                                    name='Hidden_bias')

        self.hidden_state = tf.tanh(
            tf.add(tf.matmul(self.input_place, self.hidden_W), self.hidden_b), name='Hidden_state')

        self.output_W = tf.Variable(tf.truncated_normal(shape=[self.hidden_num, self.output_size], dtype=tf.float64),
                                    dtype=tf.float64, name='Output_weight')
        self.output_b = tf.Variable(tf.constant(0.1, shape=[self.output_size, ], dtype=tf.float64), dtype=tf.float64,
                                    name='Output_bias')

        self.prediction = tf.add(tf.matmul(self.hidden_state, self.output_W), self.output_b, name='Prediction')

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(0.001, global_step, decay_steps=3000000000, decay_rate=0.96, staircase=True)
        self.loss = tf.reduce_mean(tf.square(tf.sub(self.prediction, self.output_place)), 0)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate, name='Optimizer')
        self.minimizer = self.optimizer.minimize(self.loss, global_step=global_step, name='Minimize')

        self.init_op = tf.global_variables_initializer()

    def train_time_point(self, input_value, output_value, sess):
            [pre, los, _] = sess.run([self.prediction, self.loss, self.minimizer],
                                     feed_dict={
                                         self.input_place: input_value,
                                         self.output_place: output_value
                                     })
            return pre, los

    def train_time_sequence(self, sess, input, output):
        '''
        Train NARX network by a time sequence of data
        :param input: Input data with the shape of [time_sequence_length, batch_size, input_size]
        :param output: Output data with the shape of [time_sequence_length, batch_size, output_size=1]
        :return: A time sequence of output, MSE
        '''
        [time_sequence_length, _, input_size] = np.shape(input)
        for time in range(time_sequence_length):
            input_left_boundary = time - self.input_TDL
            if input_left_boundary < 0:
                input_data = np.concatenate((np.zeros(shape=[-input_left_boundary, self.batch_size, input_size]),
                                             input[0:time+1, :, :]), axis=0)
            else:
                input_data = input[input_left_boundary:time+1, :, :]

            output_left_boundary = time - self.output_TDL
            output_right_boundary = time - 1
            if output_left_boundary < 0:
                input_data = np.concatenate((input_data,
                                             np.zeros(shape=[-output_left_boundary, self.batch_size, input_size])), axis=0)
                if output_right_boundary > 0:
                    input_data = np.concatenate((input_data, self.output[0:time, :, :]), axis=0)
            else:
                input_data = np.concatenate((input_data,
                                             self.output[output_left_boundary:time, :, :]))

            input_data = np.reshape(input_data, newshape=[self.batch_size, self.input_size])

            prediction, loss = self.train_time_point(
                input_value=input_data,
                output_value=np.reshape(output[time, :, :], newshape=[self.batch_size, self.output_size]),
                sess=sess)
            prediction = np.reshape(prediction, [1, self.batch_size, self.output_size])
            loss = np.reshape(loss, [1, self.output_size])
            if time is 0:
                self.output = prediction
                self.los = loss
            else:
                self.output = np.concatenate((self.output, prediction), axis=0)
                self.los = np.concatenate((self.los, loss), axis=0)
            pass

        '''
        fig, ax = plt.subplots()
        ax.set_xlim([0, time_sequence_length])
        ax.set_ylim([np.min(output), np.max(output)])
        lines = [ax.plot([], [], lw=2)[0] for _ in range(2)]
        lines[0].set_data(np.arange(time_sequence_length), np.reshape(self.output, newshape=[time_sequence_length]))
        lines[1].set_data(np.arange(time_sequence_length), np.reshape(output, newshape=[time_sequence_length]))
        plt.show()
        '''
        return self.output, self.los

class NARXParameters():

    def __init__(self, hidden_num, input_TDL, output_TDL, batch_size=128):
        self.hidden_num = hidden_num
        self.input_TDL = input_TDL
        self.output_TDL = output_TDL
        self.batch_size = batch_size

