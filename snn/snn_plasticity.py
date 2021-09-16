import numpy as np

# register of all rules
Rules = {}


class STDP:
    '''
    Spike-time-dependent plasticity rule
    '''

    def __init__(self, lr = 1e-6, tau_pos = 10, tau_neg = 10):
        '''
        Constructor

        INPUTS:
            lr      -   Learning rate
            tau_pos -   Positive tau (ms)
            tau_neg -   Negative tau (ms)
        '''

        self.rule_id = 0
        self.lr = lr
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg

    def dwdt(self, t, w, kwargs):
        '''
        Calculate dwdt as per:
            w'(t) = lr * (A_p(w) * x(t) * s_x - A_n(w) * y(t) * s_y)
        '''

        return kwargs['synapses'][:,6] * (self.A_p(kwargs['synapses'][:,3]) * kwargs['pre'][:,13] * kwargs['s_x'] - self.A_n(kwargs['synapses'][:,3]) * kwargs['post'][:,14] * kwargs['s_y'])

    def A_p(self, w, w_max = 10, n_p = 1):
        '''
        Soft bound weight dependence

        INPUTS:
            w       -   Weights
            w_max   -   Maximum weight
            n_p     -   Scaling
        '''

        return (w_max - w) * n_p

    def A_n(self, w, n_n = 1):
        '''
        Soft bound weight dependence

        INPUTS:
            w   - Weights
            n_n - Scaling
        '''

        return (w * n_n)
# register STDP
Rules[0] = STDP()


class Oja:
    '''
    Oja rule
    '''

    def __init__(self, lr = 1e-6, tau_pos = 10, tau_neg = 10):
        '''
        Constructor

        INPUTS:
            lr      -   Learning rate
            tau_pos -   Positive tau (ms)
            tau_neg -   Negative tau (ms)
        '''

        self.rule_id = 0
        self.lr = lr
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg

    def dwdt(self, t, w, kwargs):
        '''
        Calculate dwdt as per:
            w'(t) = lr * (x * y - y**2 * w)
        '''

        x = kwargs['pre'][:,13] * kwargs['s_x']
        y = kwargs['post'][:,14] * kwargs['s_y']

        return kwargs['synapses'][:,6] * (x * y - (y ** 2) * kwargs['synapses'][:,3])
# register Oja
Rules[1] = Oja()
