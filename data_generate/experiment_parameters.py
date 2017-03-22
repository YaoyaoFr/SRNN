class ExperimentParameters:
    Vg = 10
    ode_precision = 0.001
    TR = 3

    def __init__(self, exp_pa):
        self.Vg = exp_pa.Vg
        self.ode_precision = exp_pa.ode_precision