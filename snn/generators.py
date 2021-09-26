import numpy as np

def Xavier(dim = 1, efficacy = 1, sd = 0, n = 1):
    '''
    Get Xavier initialisation

    INPUTS:
        dim         -   Shape of dimensons 0
        efficacy    -   unused
        sd          -   unused
        n           -   Number of neurons in previous layer
    '''

    return np.random.uniform(low = -1/np.sqrt(n), high = 1/np.sqrt(n), size = (dim,))

def XavierPositive(dim = 1, efficacy = 1, sd = 0, n = 1):
    '''
    Get Xavier initialisation (but positive)

    INPUTS:
        dim         -   Shape of dimensons 0
        efficacy    -   unused
        sd          -   unused
        n           -   Number of neurons in previous layer
    '''

    return np.random.uniform(low = 1e-6, high = 2/np.sqrt(n), size = (dim,))

def Gaussian(dim = 1, efficacy = 1, sd = 0, n = 1):
    '''
    Get Gaussian initialisation

    INPUTS:
        dim         -   Shape of dimensons 0
        efficacy    -   Mu of Gaussian
        sd          -   SD of Gaussian
        n           -   unused
    '''

    return np.random.normal(loc = efficacy, scale = sd, size = (dim,))

def Uniform(dim = 1, efficacy = 1, sd = 0, n = 0):
    '''
    Get uniform initialisation

    INPUTS:
        dim         -   Shape of dimensons 0
        efficacy    -   Lower and upper bound of distribution
        sd          -   unused
        n           -   unused
    '''

    return np.random.uniform(low = -efficacy, high = efficacy, size = (dim,))

def UniformPositive(dim = 1, efficacy = 1, sd = 0, n = 0):
    '''
    Get uniform initialisation (positive only)

    INPUTS:
        dim         -   Shape of dimensons 0
        efficacy    -   Upper bound of distribution
        sd          -   unused
        n           -   unused
    '''

    return np.random.uniform(low = 1e-6, high = efficacy, size = (dim,))
