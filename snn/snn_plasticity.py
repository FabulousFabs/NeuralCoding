import numpy as np

# register of all rules
Rules = {}


class STDP:
    '''
    Spike-time-dependent plasticity rule
    '''

    def __init__(self, lr = 1e-6, w_max = 1, n_p = 1, n_n = 1):
        '''
        Constructor

        INPUTS:
            lr      -   Learning rate
            w_max   -   Maximum positive weight
            n_p     -   Positive weight scaling
            n_n     -   Negative weight scaling
        '''

        self.rule_id = 0
        self.lr = lr
        self.w_max = w_max
        self.n_p = n_p
        self.n_n = n_n

    def dwdt(self, t, w, kwargs):
        '''
        Calculate dwdt as per:
            w'(t) = lr * (A_p(w) * x(t) * s_x - A_n(w) * y(t) * s_y)
        '''

        return kwargs['synapses'][:,6] * (self.A_p(kwargs['synapses']) * kwargs['pre'][:,13] * kwargs['s_x'] - self.A_n(kwargs['synapses']) * kwargs['post'][:,14] * kwargs['s_y'])

    def A_p(self, s):
        '''
        Soft bound weight dependence

        INPUTS:
            s   -   Synapses
        '''

        return (s[:,7] - s[:,3]) * s[:,8]

    def A_n(self, s):
        '''
        Soft bound weight dependence

        INPUTS:
            s   -   Synapses
        '''

        return (s[:,3] * s[:,9])

    @property
    def free_params(self):
        return np.array([self.w_max, self.n_p, self.n_n, 0.0])
# register STDP
Rules[0] = STDP()


class Oja:
    '''
    Oja rule
    '''

    def __init__(self, lr = 1e-6):
        '''
        Constructor

        INPUTS:
            lr      -   Learning rate
        '''

        self.rule_id = 1
        self.lr = lr

    def dwdt(self, t, w, kwargs):
        '''
        Calculate dwdt as per:
            w'(t) = lr * (x * y - y**2 * w)
        '''

        x = kwargs['pre'][:,13]
        y = kwargs['post'][:,14]
        
        return kwargs['synapses'][:,6] * (x * y - (y ** 2) * kwargs['synapses'][:,3])

    @property
    def free_params(self):
        return np.zeros((4,))
# register Oja
Rules[1] = Oja()
