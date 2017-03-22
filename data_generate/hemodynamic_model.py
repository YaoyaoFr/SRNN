import random
import numpy as np
from scipy.integrate import odeint
from data_generate.experiment_parameters import ExperimentParameters


class HemodynamicModel():

    def __init__(self, batch_size=128, bio_pa=None, exp_pa=None):
        """
        Initial Function
        :param batch_size: The batch size of hemodyanmic state
        :param bio_pa: The Biophysical Parameters using in odeint
        :param exp_pa: The parameter of experiment design
        """
        self.batch_size = batch_size

        if bio_pa is None:
            self.bio_pa = BiophysicalParameters

        if exp_pa is None:
            self.exp_pa = ExperimentParameters

    def outflow(self, v, alpha):
        return v ** (1 / alpha)

    def oxygen_extraction(self, f, phy):
        return 1-(1-phy)**(1/f)

    def hemodynamic_differential_equations(self, sta_var=None, t=None, neural=None, extra=None):
        if sta_var is None:
            raise TypeError('sta_var could not be empty!')
        s = sta_var[0]
        f = sta_var[1]
        v = sta_var[2]
        q = sta_var[3]

        if self.bio_pa is None:
            bio_pa = BiophysicalParameters().as_array()
        else:
            bio_pa = self.bio_pa.as_array(self.bio_pa)
        epsilon = bio_pa[0]
        kappa = bio_pa[1]
        gamma = bio_pa[2]
        tau = bio_pa[3]
        alpha = bio_pa[4]
        phi = bio_pa[5]

        dx = np.zeros(4)
        dx[0] = epsilon*neural - kappa*s - gamma*(f-1)
        dx[1] = s
        dx[2] = (f-self.outflow(v, alpha))/tau
        dx[3] = (f*self.oxygen_extraction(f, phi)/phi-self.outflow(v, alpha)*q/v)
        return dx

    def dynamic_hemodynamic_odeint(self, initial_state=None, neural=None):
        if initial_state is None:
            initial_state = StateVariables().initial_state(batch_size=self.batch_size)

        if neural is None:
            raise TypeError('Neural is Empty!')

        state = initial_state
        states = list()

        for n in neural:
            states.append(state)
            state = self.hemodynamic_odeint(sta_var=state, neural=n)

        #   Transfrom List to NdArray
        states = np.array(states)
        return states

    def hemodynamic_odeint(self,sta_var=None, neural=None):
        if neural is None:
            raise TypeError('Neural could not be empty!')

        if self.batch_size is not None:
            batch_size = self.batch_size

        if sta_var is None:
            sta_var = StateVariables().initial_state(batch_size=batch_size)

        if self.bio_pa is None:
            bio_pa = BiophysicalParameters
        else:
            bio_pa = self.bio_pa

        if self.exp_pa is None:
            exp_pa = ExperimentParameters
        else:
            exp_pa = self.exp_pa

        Vg = exp_pa.Vg
        interval = 1/Vg
        ode_precision = exp_pa.ode_precision

        time_span = np.arange(0, interval, ode_precision)

        next_sta_var = np.zeros([batch_size, 4])
        for i in range(batch_size):
            balloon = odeint(self.hemodynamic_differential_equations, sta_var[i, :], time_span,
                             args=tuple((neural[i], None)))
            next_sta_var[i, :] = balloon[len(balloon)-1]

        return next_sta_var


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


class StateVariables:
    signal = 0
    flow = 1
    volume = 1
    content = 1

    def __init__(self, sta_var=None):
        if sta_var is not None:
            if isinstance(sta_var, StateVariables):
                self.signal = sta_var.signal
                self.flow = sta_var.flow
                self.volume = sta_var.volume
                self.content = sta_var.content
            else:
                raise TypeError('The type of sta_var must be StateVariables!')

    def initial_state(self, batch_size=128):
        sta_var = np.zeros([batch_size, 4])
        sta_var[:, :] = [self.signal, self.flow, self.volume, self.content]
        return sta_var


