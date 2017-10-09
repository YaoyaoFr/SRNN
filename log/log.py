from all_parameters import AllParameters
from data_generate.data import Data
import scipy.io as sio
import numpy as np
import os


class Log:
    input_size, vq_hidden_size, f_hidden_size, s_hidden_size, save_path, load_path, need_init, restore_epoch, data_type, save_epoch = AllParameters().get_log_parameters()

    def __init__(self, sess=None):
        self.sess = sess

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.size_str = str(self.input_size) + '_' + str(self.vq_hidden_size) + '_' + str(self.f_hidden_size) + '_' + str(self.s_hidden_size)

        self.file_path_train = self.save_path + '\\train-' + self.size_str + '.txt'
        self.file_path_test = self.save_path + '\\test-' + self.size_str + '.txt'
        if self.need_init:
            self.file_train_record = open(self.file_path_train, 'w')
            self.file_test_record = open(self.file_path_test, 'w')
            self.file_train_record.close()
            self.file_test_record.close()
        else:
            self.file_train_record = open(self.file_path_train, 'a')
            self.file_test_record = open(self.file_path_test, 'a')
            self.file_train_record.close()
            self.file_test_record.close()

    def write_error(self, los, epoch, lr, is_train):
        if is_train:
            los_type = 'Train'
            file = open(self.file_path_train, 'a')
        else:
            los_type = 'Test '
            file = open(self.file_path_test, 'a')
            lr = 0
        los = np.mean(los[:], 0)
        error_string = los_type + ':  Epoch - {:2d}  MSE:  s - {:e}  f - {:e}  v - {:e} q - {:e}  lr  -  {:f}' \
            .format(epoch + 1, los[0], los[1], los[2], los[3], lr)
        print(error_string)
        file.write(error_string + '\n')
        file.close()

    def save_model(self, epoch, saver):
        if (epoch + 1) % self.save_epoch == 0:
            save_path = self.save_path + '/model' + str(epoch + 1) + '-' + self.size_str + '.ckpt'
            save_path = saver.save(self.sess, save_path)
            print('Model saved in file', save_path)

    def restore(self, saver):
        load_path = self.load_path + '/model' + str(self.restore_epoch) + '-' + self.size_str + '.ckpt'
        saver.restore(self.sess, load_path)
        print('Model restored.')

    def save_mat(self, data, epoch, is_train, save_name=None):
        if (epoch + 1) % self.save_epoch == 0:
            if is_train:
                save_type = '_result-'
            else:
                save_type = '_test_result-'

            if save_name is None:
                save_name = self.data_type + save_type + str(epoch + 1) + '.mat'

            if isinstance(data, Data):
                sio.savemat(self.save_path + '\\' + save_name,
                    {
                        'pre_MVA_neural': data.pre_MA_neural,
                        'pre_state': data.pre_hemodynamic_state,
                        'state': data.hemodynamic_state,
                        'pre_BOLD': data.pre_BOLD,
                        'raw_BOLD': data.raw_BOLD,
                        'noisy_BOLD': data.noisy_interpolated_BOLD,
                        'pre_neural': data.pre_neural,
                        'neural': data.neural
                    })
                print('Data Saved ' + self.save_path + '\\' + save_name)