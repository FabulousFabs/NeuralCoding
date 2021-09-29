import numpy as np

# register of all filters
Filters = {}

FLAG_PARAMS_SIZE = 5 # flag for total parameter size available to filters

class Prototype:
    def __init__(self):
        self.params = np.zeros((FLAG_PARAMS_SIZE,))

    def pre(self):
        pass

    def post(self):
        pass

class ExponentialDecay(Prototype):
    '''
    Exponential weight decay within simulation (required for TTFS, for example).
    '''

    def __init__(self, tau_k = 10.0):
        '''
        Initialise super class

        INPUTS:
            tau_k   -   Time constant for exponential weight decay.
        '''
        super().__init__()

        self.type = 0

        '''
        Initialise free parameters
        '''
        self.params[0] = self.type # rule id
        self.params[1] = tau_k

    def pre(self, t, dt, filters, synapses, synapses_all):
        '''
        Pre-episode filtering that decays the weight exponentially as per
            w(t+1) = w(t) * e^(-t/tau_k)

        NOTE that the state of w(t) is saved here such that it can be reset after
        the current step.

        INPUTS:
            t               -   Current time
            dt              -   Time step size
            filters         -   All filters of this kind
            synapses        -   All synapses corresponding to the filters
            synapses_all    -   All synapses, globally

        OUTPUTS:
            filters     -   Updated filter structure
            synapses    -   Updated synapse structure
        '''

        filters[:,2] = synapses[:,3]
        synapses[:,3] = synapses[:,3] * np.exp(-t / filters[:,1])

        return (filters, synapses)

    def post(self, t, dt, filters, synapses, synapses_all):
        '''
        Resets the state of w(t+1) to w(t) for appropriate weight updating.

        INPUTS:
            t               -   Current time
            dt              -   Time step size
            filters         -   All filters of this kind
            synapses        -   All synapses corresponding to the filters
            synapses_all    -   All synapses, globally

        OUTPUTS:
            filters     -   Updated filter structure
            synapses    -   Updated synapse structure
        '''

        filters[:,3] = synapses[:,3]
        synapses[:,3] = filters[:,2]

        return (filters, synapses)
# register exponential decay
Filters[0] = ExponentialDecay()


class RelativeStrength(Prototype):
    '''
    A LEABRA-style relative strenghtening of weights to allow for some fibres to
    be principally stronger or weaker relative to global fibre strength.
    '''

    def __init__(self, multiplier = 1.0):
        '''
        Initialise super class

        INPUTS:
            multiplier  -   Relative strength of the fibre.
        '''
        super().__init__()

        self.type = 1

        '''
        Initialise free parameters
        '''
        self.params[0] = self.type # rule id
        self.params[1] = multiplier

    def pre(self, t, dt, filters, synapses, synapses_all):
        '''
        Pre-episode filtering that updates the weights by the modifier, relative
        to the global weight evolution.

        NOTE that the state of w(t) is saved here such that it can be reset after
        the current step.

        INPUTS:
            t               -   Current time
            dt              -   Time step size
            filters         -   All filters of this kind
            synapses        -   All synapses corresponding to the filters
            synapses_all    -   All synapses, globally

        OUTPUTS:
            filters     -   Updated filter structure
            synapses    -   Updated synapse structure
        '''

        wh = np.mean(synapses_all[:,3])
        wc = np.mean(synapses[:,3])
        rc = wc / wh
        rn = filters[:,1] / rc

        filters[:,2] = synapses[:,3]
        synapses[:,3] = rn * synapses[:,3]

        return (filters, synapses)

    def post(self, t, dt, filters, synapses, synapses_all):
        '''
        Resets the state of w(t+1) to w(t) for appropriate weight updating.

        INPUTS:
            t               -   Current time
            dt              -   Time step size
            filters         -   All filters of this kind
            synapses        -   All synapses corresponding to the filters
            synapses_all    -   All synapses, globally

        OUTPUTS:
            filters     -   Updated filter structure
            synapses    -   Updated synapse structure
        '''

        filters[:,3] = synapses[:,3]
        synapses[:,3] = filters[:,2]

        return (filters, synapses)
# register exponential decay
Filters[1] = RelativeStrength()


class Phase(Prototype):
    '''
    Phase-related weight filtering (to support, for example, phase encoding).
    '''

    def __init__(self, phases = 8):
        '''
        Initialise super class

        INPUTS:
            phases  -   Number of phases to use.
        '''
        super().__init__()

        self.type = 2

        '''
        Initialise free parameters
        '''
        self.params[0] = self.type # rule id
        self.params[1] = phases

    def pre(self, t, dt, filters, synapses, synapses_all):
        '''
        Pre-episode filtering that decays the weight relative to phases as per
            w(t+1) = w(t) * 2^-[(1 + mod(t, phases))]

        NOTE that the state of w(t) is saved here such that it can be reset after
        the current step.

        INPUTS:
            t               -   Current time
            dt              -   Time step size
            filters         -   All filters of this kind
            synapses        -   All synapses corresponding to the filters
            synapses_all    -   All synapses, globally

        OUTPUTS:
            filters     -   Updated filter structure
            synapses    -   Updated synapse structure
        '''

        filters[:,2] = synapses[:,3]
        synapses[:,3] = synapses[:,3] * 2 ** (-(1 + np.mod(t, filters[:,1])))

        return (filters, synapses)

    def post(self, t, dt, filters, synapses, synapses_all):
        '''
        Resets the state of w(t+1) to w(t) for appropriate weight updating.

        INPUTS:
            t               -   Current time
            dt              -   Time step size
            filters         -   All filters of this kind
            synapses        -   All synapses corresponding to the filters
            synapses_all    -   All synapses, globally

        OUTPUTS:
            filters     -   Updated filter structure
            synapses    -   Updated synapse structure
        '''

        filters[:,3] = synapses[:,3]
        synapses[:,3] = filters[:,2]

        return (filters, synapses)
# register exponential decay
Filters[2] = Phase()
