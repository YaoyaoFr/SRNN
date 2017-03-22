import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from data_generate.neural_activity import BasicBlockDesign
from data_generate.hemodynamic_model import StateVariables, HemodynamicModel
from data_generate.BOLD_measurement import gen_BOLD

HM = HemodynamicModel(batch_size=10)
neural = np.ones(shape=[200, 10])
states = HM.dynamic_hemodynamic_odeint(neural=neural)

BOLD = gen_BOLD(states)
pass


