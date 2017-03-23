from math import e


class ExperimentParameters:
    Vg = 10
    ode_precision = 0.001
    TR = 1
    observation_noise = e ** (-6)

    def __init__(self, exp_pa=None):
        if exp_pa is not None and isinstance(exp_pa, ExperimentParameters):
            self.Vg = exp_pa.Vg
            self.ode_precision = exp_pa.ode_precision
            self.TR = exp_pa.TR
            self.observation_noise = exp_pa.observation_noise


class BiophysicalParameters:
    epsilon = 0.54
    kappa = 0.65
    gamma = 0.38
    tau = 0.98
    alpha = 0.34
    phi = 0.32

    def __init__(self, bio_pa=None):
        if isinstance(bio_pa, BiophysicalParameters):
            self.epsilon = bio_pa.epsilon
            self.kappa = bio_pa.kappa
            self.gamma = bio_pa.gamma
            self.tau = bio_pa.tau
            self.alpha = bio_pa.alpha
            self.phi = bio_pa.phi

    def as_array(self):
        bio_pa = [self.epsilon, self.kappa, self.gamma, self.tau, self.alpha, self.phi]
        return bio_pa