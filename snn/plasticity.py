import numpy as np

from .neurons_labels import *

# register of all rules
Rules = {}


class STDP:
    '''
    Spike-time-dependent plasticity
    '''

    def __init__(self, lr = 1e-6, w_max = 0.0, n_p = 1.0, n_n = 1.0):
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

        x = kwargs['pre'][:,PARAM_UNI.y.value]
        y = kwargs['post'][:,PARAM_UNI.y.value]
        xy = x - y

        A_p = np.where(xy > 0)[0]
        A_n = np.where(xy < 0)[0]

        dwdt = np.zeros((kwargs['pre'].shape[0],))
        dwdt[A_p] = self.A_p(kwargs['synapses'][A_p,:]) * xy[A_p] * kwargs['s_x'][A_p]
        dwdt[A_n] = self.A_n(kwargs['synapses'][A_n,:]) * xy[A_n] * kwargs['s_y'][A_n]

        return kwargs['synapses'][:,6] * dwdt

    def A_p(self, s):
        '''
        Soft bound weight dependence

        INPUTS:
            s   -   Synapses
        '''

        return (s[:,3] * s[:,8])

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
    Oja
    '''

    def __init__(self, lr = 1e-6, beta = 1e-6):
        '''
        Constructor

        INPUTS:
            lr      -   Learning rate
            beta    -   Forgetting rate
        '''

        self.rule_id = 1
        self.lr = lr
        self.beta = beta

    def dwdt(self, t, w, kwargs):
        '''
        Calculate dwdt as per:
            w'(t) = lr * (x * y - beta * y**2 * w)
        '''

        x = kwargs['pre'][:,PARAM_UNI.y.value]
        y = kwargs['post'][:,PARAM_UNI.y.value]

        return kwargs['synapses'][:,6] * (x * y - kwargs['synapses'][:,7] * (y ** 2) * kwargs['synapses'][:,3])

    @property
    def free_params(self):
        return np.array([self.beta, 0.0, 0.0, 0.0])
# register Oja
Rules[1] = Oja()


class BCM:
    '''
    Bienenstock-Cooper-Munro
    '''

    def __init__(self, lr = 1e-6, epsilon = 1e-3, y_0 = 1):
        '''
        Constructor

        INPUTS:
            lr      -   Learning rate
            epsilon -   Forgetting rate
            y_0     -   Theta scaling factor
        '''

        self.rule_id = 2
        self.lr = lr
        self.epsilon = epsilon
        self.y_0 = y_0

    def dwdt(self, t, w, kwargs):
        '''
        Calculate dwdt as per:
            w'(t) = lr * (y * (y - theta(y)) * x - e * w)
        '''

        x = kwargs['pre'][:,PARAM_UNI.y.value]
        y = kwargs['post'][:,PARAM_UNI.y.value]

        return kwargs['synapses'][:,6] * (y * (y - self.theta(y, kwargs['synapses'])) * x - kwargs['synapses'][:,7] * kwargs['synapses'][:,3])

    def theta(self, y, s):
        '''
        Threshold as per:
            theta(y) = E[y / y_0]
        '''

        return np.mean(y / s[:,8])

    @property
    def free_params(self):
        return np.array([self.epsilon, self.y_0, 0.0, 0.0])
# register BCM
Rules[2] = BCM()
