import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from math import e
from data_generate.neural_activity import BasicBlockDesign
from data_generate.hemodynamic_model import StateVariables, HemodynamicModel
from data_generate.BOLD_measurement import BOLDMeasurement
from data_generate.data_generation import DataGeneration

[neural, state, BOLD] = DataGeneration().data_generation()
pass

