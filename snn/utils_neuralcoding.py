import numpy as np

from . import generators

def rate(homogenous = True, inputs = None, L = None, lam = None):
    '''
    Returns a rate encoding of the stimulus using a Poisson spike train of L as per
        Guo, W., Fouda, M.E., Eltawil, A.M., & Salama, K.N. (2021). Neural coding in spiking neural networks: A comparative study for robust neuromorphic systems. Frontiers in Neuroscience, 15, e638474. DOI: http://dx.doi.org/10.3389/fnins.2021.638474

    INPUTS:
        homogenous  -   Homogenous firing rate? (For time-varying signals, use False)
        inputs      -   Input vector (neuron x input)
        L           -   Length of output vector
        lam         -   Lambda for transforming input values (practical scaling of Hz).

    OUTPUTS:
        y           -   Rate encoded spike train
    '''

    if inputs is None or L is None or lam is None: return False

    ''' Inhomogenous encoding '''
    if homogenous is False:
        # setup outputs and loop
        y = np.array([], dtype = np.int)
        N = inputs.shape[0]

        for n in np.arange(0, N, 1):
            y_n = generators.Poisson(homogenous = False, dim = (1, L), rf = lambda x: ((inputs[n,np.clip(x, 0, L-1).astype(np.int)]/lam) / L / (1000 / L)))
            y = np.vstack((y, y_n)) if y.shape[0] > 0 else np.array(y_n)

        return y

    ''' Homogenous encoding '''
    # setup outputs and loop
    y = np.array([], dtype = np.int)
    N = inputs.shape[0]

    for n in np.arange(0, N, 1):
        y_n = generators.Poisson(dim = (1, L), r = ((inputs[n]/lam) / L / (1000 / L)))
        y = np.vstack((y, y_n)) if y.shape[0] > 0 else np.array(y_n)

    return y

def ttfs(inputs = None, max = None, T = 1, dt = 1e-3, theta_0 = 1.0, tau_theta = 10.0):
    '''
    Returns a time-to-first-spike encoding of the stimulus as per
        Park, S., Kim, S.J., Na, B., & Yoon, S. (2020). T2FSNN: Deep spiking neural networks with time-to-first-spike coding. Proceedings of the 2020 57th ACM/IEEE Design Automation Conference. DOI: http://dx.doi.org/10.1109/DAC18072.2020.9218689

    NOTE that this currently only supports 1d inputs, time-varying signals need yet
    to be implemented for this form of encoding.
    NOTE also that this form of encoding relies on a weight decay function that
    effectively mirrors P_th computed here as per
        P_th(t) = theta_0 * e^(-t / tau_theta)
    Please see snn.synapses.filters.ExponentialDecay.

    INPUTS:
        inputs      -   Input vector (neuron x input)
        max         -   Maximum value of input
        T           -   Desired total time (relative to dt)
        dt          -   Desired time steps
        theta_0     -   Event threshold
        tau_theta   -   Time constant for exponential decay

    OUTPUTS:
        y           -   TTFS-encoded spike train
    '''

    if inputs is None or max is None: return False

    # compute thresholds
    ins = inputs / max
    t = np.arange(0, T, dt)
    P_th = theta_0 * np.exp(-t / tau_theta)

    # compute spikes
    y = np.tile(ins.reshape((ins.shape[0], 1)), (1, t.shape[0]))
    yh = np.tile(P_th, (inputs.shape[0], 1))
    y = np.array(y >= yh).astype(np.int)

    # compute spike train such that only first spike remains
    aw = np.argwhere(y)
    ax, ix = np.unique(aw[:,0], return_index = True)
    yr = np.zeros((ins.shape[0], t.shape[0]))
    yr[ax,aw[ix,1]] = 1

    return yr

def phase(inputs = None, bits = 8, L = None):
    '''
    Returns a phase encoding of the stimulus as per
        Kim, J., Kim, H., Huh, S., Lee, J., & Choi, K. (2018). Deep neural networks with weighted spikes. Neurocomputing, 311, 373-386. DOI: http://dx.doi.org/10.1016/j.neucom.2018.05.087

    NOTE also that this form of encoding requires a weight updating filter by
    current phase of encoding. See snn.synapses.filters.Phase.

    INPUTS:
        inputs      -   Input vector (neuron x input) but ensure that inputs are integers
        bits        -   Number of bits to encode, but ensure that
                            2**bits     >= max(inputs)
                            min(inputs) >= 0
        L           -   Length of output vector

    OUTPUTS:
        y       -   Phase-encoded spike train
    '''

    if inputs is None or L is None or np.max(inputs) > 2 ** bits or np.min(inputs) < 0: return False

    if len(inputs.shape) > 1:
        # time varying signal
        y = np.zeros((inputs.shape[0], bits * inputs.shape[1]))
        N = inputs.shape[0]
        binstr = '{0:' + str(bits) + 'b}'

        for n in np.arange(0, N, 1):
            X = inputs.shape[1]
            for x in np.arange(0, X, 1):
                bn = binstr.format(inputs[n,x]).replace(' ', '0')
                bc = list(bn)
                bi = np.array([int(x) for x in bc]).astype(np.int)
                y[n,x*bits:x*bits+bits] = bi

        return y


    # setup outputs and loop
    y = np.zeros((inputs.shape[0], L))
    N = inputs.shape[0]
    binstr = '{0:' + str(bits) + 'b}'

    for n in np.arange(0, N, 1):
        bn = binstr.format(inputs[n]).replace(' ', '0')
        bc = list(bn)
        bi = np.array([int(x) for x in bc]).astype(np.int)
        y[n,0:bits] = bi

    return y

def burst(inputs = None, L = None, max = None, N_max = 5.0, T_max = None, T_min = 2.0):
    '''
    Returns a burst encoding of the stimulus as per
        Guo, W., Fouda, M.E., Eltawil, A.M., & Salama, K.N. (2021). Neural coding in spiking neural networks: A comparative study for robust neuromorphic systems. Frontiers in Neuroscience, 15, e638474. DOI: http://dx.doi.org/10.3389/fnins.2021.638474

    NOTE that this does not yet support inputs that vary in time. That has yet
    to be implemented.

    INPUTS:
        inputs  -   Input vector (neurons x input)
        L       -   Desired output length
        max     -   Maximum value of input
        N_max   -   Maximum number of spikes in burst
        T_max   -   Maximum ISI
        T_min   -   Minimum ISI

    OUTPUTS:
        y       -   Burst-encoded spike train
    '''

    if inputs is None or L is None or max is None: return False
    if T_max is None: T_max = L

    # calculate spike count in burst and ISI
    P = inputs / max
    NsP = np.ceil(N_max * P)
    IsIP = np.ceil(-(T_max - T_min)*P + T_max)
    IsIP[np.where(NsP <= 1)] = T_max

    # setup spike train
    l = np.max((NsP-1)*IsIP)
    l = L if l < L else l
    y = np.zeros((inputs.shape[0], np.ceil(l).astype(np.int)))
    N = inputs.shape[0]

    for n in np.arange(0, N, 1):
        for s in np.arange(0, NsP[n], 1):
            y[n,int(s*IsIP[n])-1] = 1

    y[:,-1] = 0

    return y[:,0:L]
