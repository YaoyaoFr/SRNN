from math import e
import numpy as np


class AllParameters:

    # Hemodynamic states
    signal = 0
    flow = 0
    volume = 0
    content = 0

    # Block Design Parameters
    cycles = 10
    step_size = 0.1
    stimulus_last = 30
    rest_last = 30
    random_rate = 0.5

    # Gaussian Bump Parameters
    # Cycle Num
    cycle = 4
    # Time Length (s)
    time_length = 60
    # Mu
    mu = [10, 15, 39, 48]
    # Sigma
    sigma = [1, 0.7, 0.9, 0.2]

    # Experiment Parameters
    Vg = 10
    ode_precision = 0.001
    TR = 1
    noise_level = 0
    if noise_level == 0:
        observation_noise = 0
    if noise_level == 1:
        observation_noise = 8 * (10 ** (-5))
    elif noise_level == 10:
        observation_noise = 2 * (10 ** (-3))
    elif noise_level == 20:
        observation_noise = 5 * (10 ** (-3))
    elif noise_level == 30:
        observation_noise = 10 ** (-2)
    input_size = 1
    random_level = 'basic'

    # Biophysical Parameters
    epsilon = 0.5
    kappa = 0.65 * 7 / 8
    gamma = 0.41 * 7 / 8
    tau = 0.98
    alpha = 0.32
    phi = 0.34 * 7 / 8
    V0 = 0.08

    # RNN Structure Parameters
    output_size = 1
    hidden_size = 10
    batch_size = 128
    cell_type = 'LSTM'
    scope = 'structure'
    learning_rate = 0.1
    decay_step = 100
    decay_rate = 0.99
    train_batch_size = 63
    var_scope = 'structure'
    active_function = 'basic'

    # Filter Parameters
    need_init = False
    if need_init:
        restore_epoch = 0
    else:
        restore_epoch = 12000
    is_train = True
    save_path = 'F:/Data/Exp_result/2017.9.11-LSTM/'
    load_path = 'F:/Data/Exp_result/2017.8.22-LSTM/'
    save_epoch = 200
    get_data = 2
    data_path = 'F:/Data/Exp_data/raw_data.mat'
    # Generate Data in Python
    if get_data == 0:
        pass
    # Generate Data in Matlab
    elif get_data == 1:
        Vg = 1
    # Read Data in *.mat File
    elif get_data == 2:
        batch_size = 128
        Vg = 1
    data_type = 'block'
    save_path += data_type
    load_path += data_type
    train_modules = [True, False, False]

    f_size = 1
    s_size = 1
    v_q_size = 2

    vq_hidden_size = 20
    f_hidden_size = 10
    s_hidden_size = 10

    size_str = str(input_size) + '_' + str(vq_hidden_size) + '_' + str(f_hidden_size) + '_' + str(s_hidden_size)
    file_path_train = '\\train-' + size_str + '.txt'
    file_path_test = '\\test-' + size_str + '.txt'
    file_path_train = save_path + file_path_train
    file_path_test = save_path + file_path_test

    # Size of variable vectors
    input_sizes = [input_size, vq_hidden_size, f_hidden_size]
    hidden_sizes = [vq_hidden_size, f_hidden_size, s_hidden_size]
    output_sizes = [v_q_size, f_size, s_size]
    var_scopes = ['v_q', 'f', 's']

    test_data_size = 128
    train_data_size = 1000 - test_data_size

    # Result Parameters
    # train_modules
    batch_nums = [15, 38]

    def get_hemodynamic_states(self):
        return self.signal, self.flow, self.volume, self.content

    def get_block_design_parameters(self):
        return self.cycles, self.step_size, self.stimulus_last, self.rest_last, self.random_rate

    def get_gaussian_bump_parameters(self):
        return self.cycle, self.time_length, self.step_size, self.mu, self.sigma

    def get_experiment_parameters(self):
        return self.Vg, self.ode_precision, self.TR, self.observation_noise, self.input_size

    def get_biophysical_parameters(self):
        return self.epsilon, self.kappa, self.gamma, self.tau, self.alpha, self.phi, self.V0

    def get_rnn_structure_parameters(self):
        return self.input_size, self.output_size, self.hidden_size, self.batch_size, self.cell_type, self.scope, \
               self.learning_rate, self.decay_step, self.decay_rate, self.train_batch_size, self.var_scope, \
               self.active_function

    def get_srnn_parameters(self):
        return self.input_sizes, self.output_sizes, self.hidden_sizes, self.batch_size, self.var_scopes

    def get_log_parameters(self):
        return self.input_size, self.vq_hidden_size, self.f_hidden_size, self.s_hidden_size, self.save_path, \
               self.load_path, self.need_init, self.restore_epoch, self.data_type, self.save_epoch

    def get_generation_parameters(self):
        return self.data_type, self.random_level, self.batch_size

    def get_filter_parameters(self):
        return self.need_init, self.restore_epoch, self.get_data, self.batch_size, self.input_size, self.is_train, \
               self.train_modules, self.get_data

    def get_result_parameters(self):
        subplot_name = []
        subplot = np.zeros(shape=[6, 1])
        if self.train_modules[2]:
            subplot[1] = 1
            subplot_name.append('s')
        if self.train_modules[1]:
            subplot[2] = 1
            subplot_name.append('f')
        if self.train_modules[0]:
            subplot[3] = 1
            subplot_name.append('v')
            subplot[4] = 1
            subplot_name.append('q')

        return subplot, subplot_name, self.batch_nums

    def get_testing_parameters(self):
        self.batch_size = self.batch_size
        return self.batch_size
