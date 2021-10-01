import numpy as np

def Xavier(dim = 1, efficacy = 1, sd = 0, n = 1, directionality = None):
    '''
    Get Xavier initialisation

    INPUTS:
        dim             -   Shape of dimensons 0
        efficacy        -   unused
        sd              -   unused
        n               -   Number of neurons in previous layer
        directionality  -   Tail of distribution (pos, neg, None for double)
    '''

    if directionality is None:
        return np.random.uniform(low = -1/np.sqrt(n), high = 1/np.sqrt(n), size = (dim,))
    if directionality < 0:
        return np.random.uniform(low = -2/np.sqrt(n), high = -1e-6, size = (dim,))
    if directionality > 0:
        return np.random.uniform(low = 1e-6, high = 2/np.sqrt(n), size = (dim,))

def Gaussian(dim = 1, efficacy = 1, sd = 0, n = 1, directionality = None):
    '''
    Get Gaussian initialisation

    INPUTS:
        dim             -   Shape of dimensons 0
        efficacy        -   Mu of Gaussian
        sd              -   SD of Gaussian
        n               -   unused
        directionality  -   unused
    '''

    return np.random.normal(loc = efficacy, scale = sd, size = (dim,))

def Uniform(dim = 1, efficacy = 1, sd = 0, n = 0, directionality = None):
    '''
    Get uniform initialisation

    INPUTS:
        dim             -   Shape of dimensons 0
        efficacy        -   Lower and upper bound of distribution
        sd              -   unused
        n               -   unused
        directionality  -   Tail of distribution (pos, neg, None for double)
    '''

    if directionality is None:
        return np.random.uniform(low = -efficacy, high = efficacy, size = (dim,))
    if directionality < 0:
        return np.random.uniform(low = -efficacy, high = -1e-6, size = (dim,))
    if directionality > 0:
        return np.random.uniform(low = 1e-6, high = efficacy, size = (dim,))

def Poisson(dim = (1,1), r = 1, homogenous = True, rf = None):
    '''
    Get Poission distribution of size=dim with rate of events = r

    INPUTS:
        dim         -   Shape of outputs
        r           -   Rate of events
        homogenous  -   Homogenous process? (True/False)
        rf          -   Rate function for events for inhomogenous process
    '''

    if homogenous is True:
        return np.clip(np.random.poisson(r, size = dim), a_min = 0, a_max = 1)

    t = np.arange(0, dim[1], 1) * np.ones(dim)
    rb = .5 * (rf(t) + rf(t+1))
    pb = 1 - np.exp(-rb * 1)
    s = np.random.uniform(size = dim)

    return np.array(pb >= s).astype(np.int)
