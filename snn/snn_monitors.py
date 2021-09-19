import numpy as np

Voltage = 11
Ampere = 12

class Prototype:
    '''
    Prototype monitor superclass
    '''

    def __init__(self):
        self.state = np.array([])

class Spikes(Prototype):
    '''
    Spike monitor
    '''

    def __init__(self, targets = None):
        '''
        Constructor

        INPUTS:
            targets     -   Population to monitor
        '''

        super().__init__()

        if targets is None:
            self.neurons = False
        else:
            self.neurons = targets

    def state_change(self, state, events, state2):
        '''
        Step function

        INPUTS:
            state   -   States of object to be observed
            events  -   Events of object to be observed
            state2  -   Synapse states of object
        '''

        if self.neurons is False:
            return False

        events = np.intersect1d(events, self.neurons, assume_unique = True)
        events_full = np.zeros((self.neurons.shape[0],))
        events_full[events] = np.ones((events.shape[0],))
        self.state = np.vstack((self.state, np.array([events_full]))) if self.state.shape[0] > 0 else np.array([events_full])

class States(Prototype):
    '''
    State monitor
    '''

    def __init__(self, targets = None, of = Voltage, is_synapse = False):
        '''
        Constructor

        INPUTS:
            targets     -   Population to be monitored
            of          -   Variable to monitor
            is_synapse  -   Are we monitoring a synapse?
        '''

        super().__init__()

        if targets is None:
            self.targets = False
        else:
            self.targets = targets

        self.of = of
        self.is_synapse = is_synapse

    def state_change(self, state, events, state2):
        '''
        Step function

        INPUTS:
            state   -   States of object to be observed
            events  -   Events of object to be observed
            state2  -   Synapse states of object
        '''

        if self.targets is False:
            return False

        if self.is_synapse is False:
            self.state = np.hstack((self.state, np.array(state[self.targets,self.of]))) if self.state.shape[0] > 0 else np.array(state[self.targets,self.of])
        else:
            self.state = np.hstack((self.state, np.array(state2[self.targets,self.of]))) if self.state.shape[0] > 0 else np.array(state2[self.targets,self.of])
