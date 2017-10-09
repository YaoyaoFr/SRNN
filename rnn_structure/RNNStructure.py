from tensorflow.python.ops.rnn_cell import BasicLSTMCell, BasicRNNCell
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data_generate.generation_parameters import ExperimentParameters
from all_parameters import AllParameters
import random


class RNNStructureParameters:

    input_size, output_size, hidden_size, batch_size, cell_type, scope, learning_rate, decay_step, decay_rate, \
        train_batch_size, var_scope, active_function = AllParameters().get_rnn_structure_parameters()

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
    active_function = pa.active_function

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
            self.active_function = pa.active_function

    def __init__(self, pa=None, sess=None):
        # Check Arguments
        if pa is not None:
            self.set_parameters(pa=pa)

        if sess is not None:
            self.sess = sess

        parameters = [self.input_size, self.output_size, self.hidden_size]
        parameter_strs = ['input size', 'output size', 'hidden size']
        for i in range(len(parameters)):
            if parameters[i] is None:
                raise TypeError('Argument ' + parameter_strs[i] + ' is None')

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

    def train_time_sequence(self, input_data=None, output_data=None, state=None, input_size=None, is_vq=True):
        """
        Train A time series of data
        :param input_data: Input data with the shape of [time_sequence_length, batch_size, input_size]
        :param output_data: Output data with the shape of [time_sequence_length, batch_size, output_size]
        :param state: Hidden state
        :param input_size:
        :param is_vq:
        :return:A series of hidden state, network output, loss value
        """
        if state is None:
            if hasattr(self, 'hidden_state'):
                state = self.hidden_state
            else:
                state = self.sess.run(self.cell.zero_state(batch_size=self.batch_size, dtype=tf.float64))

        [_, batch_size, output_size] = np.shape(output_data)
        if is_vq:
            [time_sequence_length, _, _] = np.shape(input_data)
            inps= list()
            for i in np.arange(start=input_size-1, stop=time_sequence_length):
                inps.append(np.reshape(input_data[i - input_size + 1:i + 1, :, 0], [batch_size, input_size]))
            time_sequence_length = len(inps)
        else:
            if isinstance(input_data, list):
                time_sequence_length = len(input_data)
            else:
                [time_sequence_length, batch_size, _] = np.shape(input_data)
            inps = input_data
        # Return
        value = np.zeros([time_sequence_length, batch_size, self.hidden_size])
        module_output = np.zeros([time_sequence_length, batch_size, output_size])
        MSE = np.zeros([time_sequence_length, output_size])
        for i in np.arange(start=0, stop=time_sequence_length, step=self.train_batch_size):
            if (i+self.train_batch_size) > time_sequence_length:
                continue
            inp = inps[i:i+self.train_batch_size]
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

    def test_time_sequence(self, input_data=None, state=None, is_vq=True, input_size=None, output_size=None):
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

        if is_vq:
            [time_sequence_length, batch_size, BOLD_size] = np.shape(input_data)
            inps= list()
            for i in np.arange(start=input_size-1, stop=time_sequence_length):
                inps.append(np.reshape(input_data[i - input_size + 1:i + 1, :, 0], [batch_size, input_size]))
            time_sequence_length = len(inps)
        else:
            if isinstance(input_data, list):
                time_sequence_length = len(input_data)
            else:
                [time_sequence_length, batch_size, _] = np.shape(input_data)
            inps = input_data
        Pa = RNNStructureParameters()
        value = np.zeros([time_sequence_length, batch_size, self.hidden_size])
        module_output = np.zeros([time_sequence_length, batch_size, output_size])
        for i in np.arange(start=0, stop=time_sequence_length, step=self.train_batch_size):
            if (i+self.train_batch_size) > time_sequence_length:
                continue
            inp = inps[i:i + Pa.train_batch_size]
            val, state, outp = \
                self.sess.run(
                    [self.hidden_outputs, self.hidden_state, self.RNN_outputs, ],
                    {
                        self.input_place: inp,
                        self.hidden_state_place: state
                    }
                )
            value[i:i + Pa.train_batch_size, :, :] = val
            module_output[i:i + Pa.train_batch_size, :, :] = outp
            pass
        return value, state, module_output


class SRNNStructure:

    def __init__(self, sess):
        input_sizes, output_sizes, hidden_sizes, batch_size, var_scopes = AllParameters().get_srnn_parameters()

        pas = list()
        for i in range(3):
            pa = RNNStructureParameters(input_size=input_sizes[i],
                                        output_size=output_sizes[i],
                                        hidden_size=hidden_sizes[i],
                                        batch_size=batch_size,
                                        var_scope=var_scopes[i])
            if i == 2:
                pa.active_function = 'sigmoid'
            pas.append(pa)
        self.RNNs = [RNNStructure(pa=pa, sess=sess) for pa in pas]
        print('HRNN initialized.')

    def zero_state(self, batch_size):
        return [RNN.zero_state(batch_size=batch_size) for RNN in self.RNNs]

    def train_time_sequence(self, data, train_modules=None):
        input_size = AllParameters().input_size
        train_data_size = AllParameters().train_data_size
        batch_size = AllParameters().batch_size
        index = random.sample(list(np.arange(train_data_size)), batch_size)
        data.train_index = index

        input_data = data.noisy_interpolated_BOLD[:, index, :]
        states = data.hidden_state
        output_data = data.hemodynamic_state[:, index, :]
        [time_sequence_length, batch_size, _] = np.shape(input_data)

        if train_modules is None:
            train_modules = [True for _ in range(3)]

        if states is None:
            self.states = self.zero_state(batch_size=batch_size)
        else:
            self.states = states
        preprocessed_output = data.preprocess_data(output=output_data)
        loss = np.zeros([time_sequence_length-input_size+1, 4])
        prediction = np.zeros([time_sequence_length, batch_size, 4])
        for j in range(3):
            if j == 0:
                if train_modules[j]:
                    value, self.states[j], prediction[input_size-1:time_sequence_length, :, 2:4], loss[:, 2:4], lr = \
                                     self.RNNs[j].train_time_sequence(input_data=input_data,
                                                                      output_data=preprocessed_output[input_size-1:time_sequence_length, :, 2:4],
                                                                      state=self.states[j],
                                                                      input_size=input_size)
                else:
                    value, self.states[j], prediction[input_size - 1:time_sequence_length, :, 2:4]= \
                        self.RNNs[j].test_time_sequence(input_data=input_data,
                                                        state=self.states[j],
                                                        input_size=input_size)
            else:
                if train_modules[j]:
                    value, self.states[j], pre, los, lr = \
                        self.RNNs[j].train_time_sequence(input_data=value,
                                                         output_data=np.reshape(preprocessed_output[input_size-1:time_sequence_length, :, 2-j],
                                                                        newshape=[time_sequence_length-input_size+1, batch_size, 1]),
                                                         state=self.states[j],
                                                         is_vq=False)
                    loss[:, 2-j] = np.reshape(los, [np.shape(los)[0]])
                else:
                    value, self.states[j], pre = \
                        self.RNNs[j].test_time_sequence(input_data=value,
                                                        state=self.states[j],
                                                        is_vq=False)
                prediction[input_size-1:time_sequence_length, :,  2-j] = \
                            np.reshape(pre, [np.shape(pre)[0], batch_size])
        prediction = data.postprocess_data(output=prediction)

        data.set(pre_hemodynamic_state=prediction, los=loss, hidden_state=self.states, lr=lr)

        return data

    def test_time_sequence(self, data, is_noisy=True):
        train_data_size = AllParameters().train_data_size
        test_data_size = AllParameters().test_data_size
        batch_size = AllParameters().batch_size

        if is_noisy:
            input = data.noisy_interpolated_BOLD
        else:
            input = data.raw_BOLD

        if batch_size >= test_data_size:
            index = np.arange(start=train_data_size, stop=train_data_size+test_data_size)
            data.test_index = index
            input = input[:, index, :]
        else:
            data.test_index = np.arange(start=0, stop=batch_size)

        states = data.hidden_state
        input_size = AllParameters().input_size

        [time_sequence_length, batch_size, _] = np.shape(input)

        if test_data_size < batch_size:
            self.states = [RNN.zero_state(batch_size=test_data_size) for RNN in self.RNNs]
        else:
            self.states = states
        prediction = np.zeros([time_sequence_length, batch_size, 4])
        for j in range(3):
            if j == 0:
                value, self.states[j], prediction[input_size-1:time_sequence_length, :, 2:4] = \
                    self.RNNs[j].test_time_sequence(input_data=input,
                                                    state=self.states[j],
                                                    input_size=input_size,
                                                    output_size=2)
            else:
                value, self.states[j], pre= \
                    self.RNNs[j].test_time_sequence(input_data=value,
                                                    state=self.states[j],
                                                    is_vq=False)
                prediction[input_size-1:time_sequence_length, :,  2-j] = np.reshape(pre, [np.shape(pre)[0], batch_size])
        prediction = data.postprocess_data(prediction)

        if batch_size == 1:
            prediction = np.reshape(prediction, [time_sequence_length, 1, 4])

        data.set(pre_hemodynamic_state=prediction, hidden_state=states)
        return data

