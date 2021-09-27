import numpy as np

from .units import *
from .neurons_labels import *

FLAG_PARAMS_SIZE = 50 # flag for total parameter size available for neurons

class Prototype:
    '''
    Neuron prototype super class
    '''

    def __init__(self):
        '''
        Constructor creating defaults
        '''

        self.struct = -1
        self.type = -1
        self.E_l = 0.0          # reset potential
        self.V = self.E_l       # membrane potential
        self.V_thr = 1.0        # spiking threshold
        self.I = 0.0            # incoming current
        self.tau_pos = 10.0     # positive spike trace tau constant
        self.tau_neg = 10.0     # negative spike trace tau constant
        self.xp = 1e-6          # presynaptic spike trace (of arrivals)
        self.x = 1e-6           # presynaptic spike trace
        self.y = 1e-6           # postsynaptic spike trace
        self.a = 0.0            # adaptation constant alpha
        self.b = 0.0            # adaptation constant beta
        self.tau_k = 10.0       # adaptation time constant
        self.w = 0.0            # adapation current
        self.A = 1.0            # output current
        self.inhib_ff = 0.0     # lateral inhibition (feedforward)
        self.inhib_fb = 0.0     # lateral inhibition (feedback)
        self.ff0 = 0.1          # feedforward inhibition zero
        self.ff_mva = 0.0       # feedforward max vs average
        self.fb_tau = 1.4       # feedback inhibition time constant
        self.ff = 0.0           # incoming feedforward inhibition
        self.fb = 0.0           # incoming feedback inhibition

        self.__params = np.zeros((1, FLAG_PARAMS_SIZE))
        self.free_params = None

    def set_opts(self, opts = []):
        '''
        Set class-level option

        INPUTS:
            opts    -   K=>V pairs of options
        '''

        if len(opts) < 1:
            return False

        for key in opts:
            setattr(self, key, opts[key])

        return True

    def dxdt(self, t, X, kwargs):
        '''
        Compute presynaptic trace as per:
            x'(t) = 1/tau_pos * (-x + a_p(x) * s)
        '''

        s = np.zeros((kwargs['neurons'].shape[0],))
        s[kwargs['spike_indx']] = 1

        return (1 / kwargs['neurons'][:,PARAM_UNI.tau_pos.value]) * (-kwargs['neurons'][:,PARAM_UNI.x.value] + self.a_p(kwargs['neurons'][:,PARAM_UNI.x.value]) * s)

    def a_p(self, x):
        '''
        a_p modifier for pos presynaptic traces as per:
            a_p(x) = 1 - x
        '''

        return 1 - x

    def dydt(self, t, Y, kwargs):
        '''
        Compute postsynaptic trace as per:
            y'(t) = 1/tau_neg * (-y + a_n(y) * s)
        '''

        s = np.zeros((kwargs['neurons'].shape[0],))
        s[kwargs['spike_indx']] = 1

        return (1 / kwargs['neurons'][:,PARAM_UNI.tau_neg.value]) * (-kwargs['neurons'][:,PARAM_UNI.y.value] + self.a_n(kwargs['neurons'][:,PARAM_UNI.y.value] * s))

    def a_n(self, x):
        '''
        a_n modifier for neg postsynaptic traces as per
            a_n(x) = 1  where x > 0
            a_n(x) = 0  where x <= 0
        '''

        return np.where(x > 0, 1, 0)

    def dffdt(self, t, ff, kwargs):
        '''
        Feedforward inhibition updates as per
            ff'(t) = inhib_ff * (xh - ff0) * xm
        '''

        xh = np.mean(kwargs['neurons'][:,PARAM_UNI.I.value]) + kwargs['neurons'][:,PARAM_UNI.ff_mva.value] * (np.max(kwargs['neurons'][:,PARAM_UNI.I.value]) - np.mean(kwargs['neurons'][:,PARAM_UNI.I.value]))
        xm = np.where(kwargs['neurons'][:,PARAM_UNI.ff0.value] < xh, 1, 0)

        return kwargs['neurons'][:,PARAM_UNI.inhib_ff.value] * (xh - kwargs['neurons'][:,PARAM_UNI.ff0.value]) * xm

    def dfbdt(self, t, fb, kwargs):
        '''
        Feedback inhibition updates as per
            fb'(t) = inhib_fb * (1 / fb_tau * (yh - fb))
        '''

        return kwargs['neurons'][:,PARAM_UNI.inhib_fb.value] * ((1 / kwargs['neurons'][:,PARAM_UNI.fb_tau.value]) * (np.mean(kwargs['neurons'][:,PARAM_UNI.y.value]) - fb))

    def params(self):
        '''
        Get parameters of current neuron model
        '''

        # setup defaults
        for num in PARAM_UNI:
            self.__params[0,num.value] = getattr(self, num.name)

        # setup free parameters
        if self.free_params is not None:
            for n in self.free_params:
                self.__params[0,n.value] = getattr(self, n.name)

        return self.__params


# register of our types
Neurons = {}


class LIF(Prototype):
    '''
    Standard LIF neuromorphics neuron class
    '''

    def __init__(self, opts = []):
        '''
        Standard parameters
        '''

        super().__init__()

        '''
        Free parameters
        '''

        self.type = 0
        self.m = 1
        self.N = 0
        self.rng = np.random.RandomState()
        self.free_params = PARAM_LIF

        self.set_opts(opts)

    def dVdt(self, t, V, kwargs):
        '''
        Compute dVdt as per
            V'(t) = V * m + (I_e - I_ff - I_fb + N)
        '''

        noise = self.rng.normal(scale = self.N, size = (kwargs['neurons'][:,PARAM_LIF.N.value].shape[0],))
        noise_mask = np.where(kwargs['neurons'][:,PARAM_LIF.N.value] != 0, 1, 0).astype(np.int)

        return V * kwargs['neurons'][:,PARAM_LIF.m.value] + (kwargs['neurons'][:,PARAM_UNI.I.value] - kwargs['neurons'][:,PARAM_UNI.ff.value] - kwargs['neurons'][:,PARAM_UNI.fb.value] + (noise_mask * noise))

    def V_apply(self, neurons, dt, dVdt):
        '''
        Apply step
        '''

        return dt * dVdt

    def dwdt(self, t, w, kwargs):
        '''
        Compute dwdt as per
            w'(t) = 1 / tau_k * (a * (V - E_l) - w_k + b_k * tau_k * s)
        '''

        s = np.zeros((kwargs['neurons'].shape[0],))
        s[kwargs['spike_indx']] = 1

        return (1 / kwargs['neurons'][:,PARAM_UNI.tau_k.value]) * (kwargs['neurons'][:,PARAM_UNI.a.value] * (kwargs['neurons'][:,PARAM_UNI.V.value] - kwargs['neurons'][:,PARAM_UNI.E_l.value]) - w + (kwargs['neurons'][:,PARAM_UNI.b.value] * kwargs['neurons'][:,PARAM_UNI.tau_k.value] * s))

    def It(self, neurons):
        '''
        Compute I(t) as per
            I(t) = A
        '''

        return neurons[:,PARAM_UNI.A.value]
# register LIF
Neurons[0] = LIF()
