import numpy as np

def spike_count(states):
    '''
    Returns spike count over time as per
        r(N, T) = N / T

    INPUTS:
        states  -   Spiking object from monitor
    '''

    return np.sum(states, axis = 1) / states.shape[1]

def linear_filter_and_kernel(states, dt = 1e-3, delta_t = 10):
    '''
    Returns TFR from convolution with linear filter

    INPUTS:
        states      -   Spiking object from monitor
        dt          -   Time step size
        delta_t     -   Time steps per window
    '''

    # kernel
    k = np.ones((states.shape[0], delta_t))

    # signal
    s = np.pad(states, ((0,0), (0, delta_t-(states.shape[1] % delta_t))), 'constant', constant_values = 0)
    s[np.isnan(s)] = 0.0

    # output
    tfr = np.zeros((s.shape[0], np.int(s.shape[1] / delta_t)))

    # slide
    for i in np.arange(0, tfr.shape[1], 1):
        tfr[:,i] = np.sum(s[:,i*delta_t:(i*delta_t+delta_t)] * k, axis = 1) / dt / delta_t

    return tfr

def gaussian_and_kernel(states, dt = 1e-3, delta_t = 10):
    '''
    Returns TFR from convolution with gaussian

    INPUTS:
        states      -   Spiking object from monitor
        dt          -   Time step size
        delta_t     -   Time steps per window
    '''

    # kernel
    std = 1
    n = np.arange(0, delta_t, 1) - (delta_t - 1) / 2
    sig2 = 2 * std * std
    k = np.exp(-n ** 2 / sig2)

    # signal
    s = np.pad(states, ((0,0), (0, delta_t-(states.shape[1] % delta_t))), 'constant', constant_values = 0)
    s[np.isnan(s)] = 0.0

    # output
    tfr = np.zeros((s.shape[0], np.int(s.shape[1] / delta_t)))

    # slide
    for i in np.arange(0, tfr.shape[1], 1):
        tfr[:,i] = np.sum(s[:,i*delta_t:(i*delta_t+delta_t)] * k, axis = 1) / dt / delta_t

    return tfr
