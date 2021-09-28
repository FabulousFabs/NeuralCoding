import numpy as np

from . import generators

def rate(homogenous = True, inputs = None, L = None, lam = None):
    '''
    Returns a rate encoding of the stimulus using a Poisson spike train of L.

    INPUTS:
        homogenous  -   Homogenous firing rate? (For time-varying signals, use False)
        inputs      -   Input vector (neuron x input)
        L           -   Length of output vector
        lam         -   Lambda for transforming input values (practical scaling of Hz).
    '''

    if inputs is None or L is None or lam is None: return False

    ''' Inhomogenous encoding '''
    if homogenous is False:
        # setup outputs and loop
        y = np.array([], dtype = np.int)
        N = inputs.shape[0]

        for n in np.arange(0, N, 1):
            y_n = generators.Poisson(homogenous = False, dim = (1, L), rf = lambda x: inputs[n,x]/lam)
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
