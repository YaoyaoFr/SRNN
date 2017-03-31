import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from math import e
from data_generate.neural_activity import BasicBlockDesign
from data_generate.hemodynamic_model import StateVariables, HemodynamicModel
from data_generate.BOLD_measurement import BOLDMeasurement
from data_generate.data_generation import DataGeneration
from NARX_structure.NARX_structure import NARXStructure
from class_test import test

print('a'+'b')

hidden_num = 3
input_TDL = 5
output_TDL = 1
batch_size = 1
a = NARXStructure(hidden_num=3, input_TDL=5, output_TDL=1, batch_size=1)

input_value = np.ones(shape=[batch_size, input_TDL+output_TDL])
output_value = np.ones(shape=[batch_size, 1])

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    a(input_value, output_value, sess)



