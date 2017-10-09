import random
import numpy as np
from scipy.integrate import odeint
from data_generate.generation_parameters import ExperimentParameters, BiophysicalParameters
from all_parameters import AllParameters


class HemodynamicModel:
    def __init__(self, batch_size=128, bio_pa=None, exp_pa=None):
        """
        Initial Function
        :param batch_size: The batch size of hemodyanmic state
        :param bio_pa: The Biophysical Parameters using in odeint
        :param exp_pa: The parameter of experiment design
        """
        self.batch_size = batch_size

        if bio_pa is None:
            bio_pa = BiophysicalParameters
        self.bio_pa = bio_pa

        if exp_pa is None:
            exp_pa = ExperimentParameters
        self.exp_pa = exp_pa

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
            self.bio_pa = BiophysicalParameters()
        epsilon = self.bio_pa.epsilon
        kappa = self.bio_pa.kappa
        gamma = self.bio_pa.gamma
        tau = self.bio_pa.tau
        alpha = self.bio_pa.alpha
        phi = self.bio_pa.phi

        dx = np.zeros(4)
        dx[0] = (epsilon*neural - kappa*(s-1) - gamma*(f-1))/s
        dx[1] = (s-1)/f
        dx[2] = (f-self.outflow(v, alpha)) / (tau * v)
        dx[3] = (f*self.oxygen_extraction(f, phi)/phi-self.outflow(v, alpha)*q/v) / (tau * q)
        return dx

    def dynamic_hemodynamic_odeint(self, neural, initial_state=None):
        if initial_state is None:
            initial_state = StateVariables().initial_state(batch_size=self.batch_size)

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


class StateVariables:
    signal, flow, volume, content = AllParameters().get_hemodynamic_states()

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
        sta_var = np.exp(sta_var)
        return sta_var


