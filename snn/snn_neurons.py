import numpy as np

class Prototype:
    '''
    Neuron prototype super class
    '''

    def __init__(self):
        pass

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

        return (1 / kwargs['neurons'][:,3]) * (-kwargs['neurons'][:,13] + self.a_p(kwargs['neurons'][:,13] * kwargs['spike_indx']))

    def a_p(self, x):
        '''
        a_p modifier for pos presynaptic traces
        '''

        return np.where(x > 0, 1, 0)

    def dydt(self, t, Y, kwargs):
        '''
        Compute postsynaptic trace as per:
            y'(t) = 1/tau_neg * (-y + a_n(y) * s)
        '''

        s = np.zeros((kwargs['neurons'].shape[0],))
        s[kwargs['spike_indx']] = 1

        return (1 / kwargs['neurons'][:,3]) * (-kwargs['neurons'][:,14] + self.a_n(kwargs['neurons'][:,14] * s))

    def a_n(self, x):
        '''
        a_n modifier for neg postsynaptic traces
        '''

        return np.where(x > 0, 1, 0)


class GLIF(Prototype):
    '''
    Standard neuron class for generalised linear integrate-and-fire as per:
        Teeter, C., Iyer, R., Menon, V., Gouwens, N., Feng, D., Berg, J., Szafer, A., Cain, N., Zeng, H., Hawrylycz, M., Koch, C., & Mihalas, S. (2018). Generalized leaky integrate-and-fire models classify multiple neuron types. Nature Communications, 9, e709. DOI: http://dx.doi.org/10.1038/s41467-017-02717-4
    '''

    def __init__(self, opts = []):
        '''
        Standard parameters (see GLIF4 in paper)
        '''

        self.R = 177.0      # MO
        self.tau = 19.0     # ms
        self.C = 107.0      # pF
        self.E_l = -75.5    # mV
        self.th_i = -47.2   # mV
        self.d_th_i = 27.8  # mV
        self.del_t = 6.55   # ms
        self.g = 40.0       # nS
        self.E_syn = 0.0    # mV
        self.V = self.E_l   # mV
        self.I = 0.0        # pA
        self.trace_pre = 1e-6
        self.trace_post = 1e-6
        self.a = 0.00103
        self.b = 1/0.00225
        self.tau_k = self.tau
        self.w = 0.0

        self.set_opts(opts)

    def dVdt(self, t, V, kwargs):
        '''
        Compute dVdt as per
            V'(t) = 1 / C * (I_e(t) - 1/R * [V(t) - E_L])
        '''

        return (1 / kwargs['neurons'][:,4]) * (kwargs['neurons'][:,12] - (1 / kwargs['neurons'][:,2]) * (V - kwargs['neurons'][:,1]))

    def V_apply(self, neurons, dt, dVdt):
        '''
        Apply step
        '''

        return neurons[:,11] + (dt * dVdt)

    def dwdt(self, t, w, kwargs):
        '''
        Compute dwdt as per
            w'(t) = a_v * (V_t - E_l) - b_v * w
        '''

        return kwargs['neurons'][:,17] * (kwargs['neurons'][:,11] - kwargs['neurons'][:,1]) - kwargs['neurons'][:,18] * w

    def It(self, neurons):
        '''
        Compute It as per
            I(t) = -g * (V - E_syn)
        '''

        return -neurons[:,9] * (neurons[:,11] - neurons[:,1])

    @property
    def params(self):
        '''
        Return parameters
        '''

        return np.array([-1, self.E_l, self.R, self.tau, self.C, self.E_l, self.th_i, self.d_th_i, self.del_t, self.g, self.E_syn, self.V, self.I, self.trace_pre, self.trace_post, 0.0], dtype=np.float)

class LIF(Prototype):
    '''
    Neuromorphics neuron class
    '''

    def __init__(self, opts = []):
        '''
        Standard parameters
        '''

        self.m = 1
        self.A = 1
        self.V = 0
        self.V_reset = 0
        self.V_thr = 1
        self.N = 0
        self.rng = np.random.RandomState()
        self.trace_pre = 1e-6
        self.trace_post = 1e-6
        self.tau = 10
        self.a = 0.0
        self.b = 0.0
        self.tau_k = 1.0
        self.w = 0.0

        self.set_opts(opts)

    def dVdt(self, t, V, kwargs):
        '''
        Compute dVdt as per
            V'(t) = V * m + (I + N)
        '''

        noise = self.rng.normal(scale = self.N, size = (kwargs['neurons'][:,1].shape[0],))
        noise_mask = np.where(kwargs['neurons'][:,9] != 0, 1, 0).astype(np.int)

        return V * kwargs['neurons'][:,4] + (kwargs['neurons'][:,12] + (noise_mask * noise))

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

        return 1 / kwargs['neurons'][:,19] * (kwargs['neurons'][:,17] * (kwargs['neurons'][:,11] - kwargs['neurons'][:,1]) - w + (kwargs['neurons'][:,18] * kwargs['neurons'][:,19] * s))

    def It(self, neurons):
        '''
        Compute I(t) as per
            I(t) = A
        '''

        return neurons[:,10]

    @property
    def params(self):
        '''
        Return parameters
        '''

        return np.array([-1, self.V_reset, 0, self.tau, self.m, self.V_reset, self.V_thr, self.V_thr, 0, self.N, self.A, self.V, 0.0, self.trace_pre, self.trace_post, 0.0, self.w, self.a, self.b, self.tau_k])
