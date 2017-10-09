from tensorflow.python.ops.rnn_cell import BasicLSTMCell, BasicRNNCell
import tensorflow as tf
import numpy as np
import random


class RNNStructureParameters:

    def __init__(self, input_size=None, output_size=None, hidden_size=None, batch_size=None, cell_type=None,
                 scope=None, learning_rate=None, decay_step=None, decay_rate=None, train_batch_size=None,
                 var_scope=None):

        if input_size is not None:
            self.input_size = input_size
        if output_size is not None:
            self.output_size = output_size
        if hidden_size is not None:
            self.hidden_size = hidden_size
        if batch_size is not None:
            self.batch_size = batch_size
        if cell_type is not None:
            self.cell_type = cell_type
        if scope is not None:
            self.scope = scope
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if decay_step is not None:
            self.decay_step = decay_step
        if decay_rate is not None:
            self.decay_rate = decay_rate
        if var_scope is not None:
            self.var_scope = var_scope
        if train_batch_size is not None:
            self.train_batch_size = train_batch_size


class RNNStructure:
    pa = RNNStructureParameters()
    input_size = pa.input_size
    output_size = pa.output_size
    hidden_size = pa.hidden_size
    cell_type = pa.cell_type
    batch_size = pa.batch_size
    lr = pa.learning_rate
    decay_step = pa.decay_step
    decay_rate = pa.decay_rate
    var_scope = pa.var_scope
    train_batch_size = pa.train_batch_size

    def set_parameters(self, pa=None):
        if pa is not None:
            self.input_size = pa.input_size
            self.output_size = pa.output_size
            self.hidden_size = pa.hidden_size
            self.cell_type = pa.cell_type
            self.batch_size = pa.batch_size
            self.lr = pa.learning_rate
            self.decay_step = pa.decay_step
            self.decay_rate = pa.decay_rate
            self.var_scope = pa.var_scope
            self.train_batch_size = pa.train_batch_size

    def __init__(self, pa=None, sess=None):
        # Check Arguments
        if pa is not None:
            self.set_parameters(pa=pa)

        if sess is not None:
            self.sess = sess

        with tf.variable_scope(self.var_scope+'_'+self.cell_type):
            #   Recurrent Neural Network Cell
            if self.cell_type is 'LSTM':
                self.cell = BasicLSTMCell(self.hidden_size, state_is_tuple=True)
            elif self.cell_type is 'RNN':
                self.cell = BasicRNNCell(self.hidden_size)

            #   Place holder
            self.input_place = tf.placeholder(dtype=tf.float64, shape=[None, self.batch_size, self.input_size])
            self.output_place = tf.placeholder(dtype=tf.float64, shape=[None, self.batch_size, self.output_size])
            self.hidden_state_place = self.cell.zero_state(batch_size=self.batch_size, dtype=tf.float64)

            #  Output
            self.W_output = tf.Variable(tf.truncated_normal(shape=[self.hidden_size, self.output_size], dtype=tf.float64),
                         dtype=tf.float64, name='Output_weight')
            self.b_output = tf.get_variable(name='Bias/Output', shape=[self.output_size],
                                            initializer=tf.random_normal_initializer(dtype=tf.float64),dtype=tf.float64)

            self.hidden_outputs, self.hidden_state = tf.nn.dynamic_rnn(cell=self.cell, time_major=True, inputs=self.input_place,
                                                                       initial_state=self.hidden_state_place)

            self.hidden_outputs_temp = tf.split(0, self.train_batch_size, self.hidden_outputs)
            self.RNN_outputs = [tf.reshape(tf.matmul(tf.reshape(o, shape=[self.batch_size, self.hidden_size]),
                                                     self.W_output), shape=[1, self.batch_size, self.output_size])
                                for o in self.hidden_outputs_temp]
            self.RNN_outputs = tf.concat(0, self.RNN_outputs)
            self.MSE = tf.reduce_mean(tf.square(tf.sub(self.RNN_outputs, self.output_place)), 1)

            # self.output = tf.add(tf.matmul(self.value, self.W_output), self.b_output)

            #   Optimize
            self.global_step = tf.get_variable(name='Optimize/Global_step', shape=[],
                                               initializer=tf.constant_initializer(0))
            # self.MSE = tf.reduce_mean(tf.square(tf.sub(self.output, self.output_place)), axis=0, name='Optimize/MSE')
            self.learning_rate = tf.train.exponential_decay(self.lr, global_step=self.global_step, decay_steps=self.decay_step,
                                                       decay_rate=self.decay_rate, staircase=True, name='Optimize/Learning_rate')
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                               name='Optimize/Optimizer')
            self.minimizer = self.optimizer.minimize(self.MSE, global_step=self.global_step,
                                                     name='Optimize/Minimizer')

    def zero_state(self, batch_size):
        return self.sess.run(self.cell.zero_state(batch_size, dtype=tf.float64))

    def train_time_sequence(self, input_data=None, output_data=None, state=None, input_size=None):
        """
        Train A time series of data
        :param input_data: Input data with the shape of [time_sequence_length, batch_size, input_size]
        :param output_data: Output data with the shape of [time_sequence_length, batch_size, output_size]
        :param state: Hidden state
        :param input_size:
        :return:A series of hidden state, network output, loss value
        """
        if state is None:
            if hasattr(self, 'hidden_state'):
                state = self.hidden_state
            else:
                state = self.sess.run(self.cell.zero_state(batch_size=self.batch_size, dtype=tf.float64))

        [time_sequence_length, batch_size, output_size] = np.shape(output_data)
        value = np.zeros([time_sequence_length, batch_size, self.hidden_size])
        module_output = np.zeros([time_sequence_length, batch_size, output_size])
        MSE = np.zeros([time_sequence_length, output_size])
        for i in np.arange(start=0, stop=time_sequence_length, step=self.train_batch_size):
            if (i+self.train_batch_size) > time_sequence_length-1:
                continue
            inp = input_data[i:i + self.train_batch_size]
            out = output_data[i:i + self.train_batch_size]
            val, state, outp, los, _, lr = \
                self.sess.run(
                    [self.hidden_outputs, self.hidden_state, self.RNN_outputs, self.MSE, self.minimizer, self.learning_rate],
                    {
                        self.input_place: inp,
                        self.output_place: out,
                        self.hidden_state_place: state
                    }
                )
            value[i:i+self.train_batch_size, :, :] =val
            module_output[i:i+self.train_batch_size, :, :] = outp
            if output_size == 1:
                MSE[i:i + self.train_batch_size, :] = np.reshape(los, [self.train_batch_size, 1])
            else:
                MSE[i:i + self.train_batch_size, :] = los
            pass
        return value, state, module_output, np.reshape(MSE, newshape=[time_sequence_length, output_size]), lr

    def test_time_sequence(self, input_data=None, state=None, output_size=None):
        """
        Train A time series of data
        :param input_data: Input data with the shape of [time_sequence_length, batch_size, input_size]
        :param state: Hidden state
        :param is_vq:
        :param input_size:
        :param output_size:
        :return: A series of hidden state, network output, loss value
        """
        if output_size is None:
            output_size = self.output_size

        if state is None:
            if hasattr(self, 'hidden_state'):
                state = self.hidden_state
            else:
                state = self.sess.run(self.cell.zero_state(batch_size=self.batch_size, dtype=tf.float64))

        [time_sequence_length, batch_size, _] = np.shape(input_data)
        value = np.zeros([time_sequence_length, batch_size, self.hidden_size])
        module_output = np.zeros([time_sequence_length, batch_size, output_size])

        for i in np.arange(start=0, stop=len(input_data)-self.train_batch_size, step=self.train_batch_size):
            inp = input_data[i:i + self.train_batch_size]
            val, state, outp = \
                self.sess.run(
                    [self.hidden_outputs, self.hidden_state, self.RNN_outputs, ],
                    {
                        self.input_place: inp,
                        self.hidden_state_place: state
                    }
                )
            value[i:i + self.train_batch_size, :, :] = val
            module_output[i:i + self.train_batch_size, :, :] = outp
            pass
        return value, state, module_output
