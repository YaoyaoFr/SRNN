from math import e
from all_parameters import AllParameters


class ExperimentParameters:
    Vg, ode_precision, TR, observation_noise, input_size = AllParameters().get_experiment_parameters()

    def __init__(self, exp_pa=None):
        if exp_pa is not None and isinstance(exp_pa, ExperimentParameters):
            self.Vg = exp_pa.Vg
            self.ode_precision = exp_pa.ode_precision
            self.TR = exp_pa.TR
            self.observation_noise = exp_pa.observation_noise


class BiophysicalParameters:
    epsilon, kappa, gamma, tau, alpha, phi, V0 = AllParameters().get_biophysical_parameters()

    def __init__(self, bio_pa=None):
        if isinstance(bio_pa, BiophysicalParameters):
            self.epsilon = bio_pa.epsilon
            self.kappa = bio_pa.kappa
            self.gamma = bio_pa.gamma
            self.tau = bio_pa.tau
            self.alpha = bio_pa.alpha
            self.phi = bio_pa.phi


    def as_array(self):
        bio_pa = [self.epsilon, self.kappa, self.gamma, self.tau, self.alpha, self.phi, self.V0]
        return bio_pa