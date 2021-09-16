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

    def __init__(self, neurons = None):
        '''
        Constructor

        INPUTS:
            neurons     -   Population to monitor
        '''

        super().__init__()

        if neurons is None:
            self.neurons = False
        else:
            self.neurons = neurons

    def state_change(self, state, events):
        '''
        Step function

        INPUTS:
            state   -   States of object to be observed
            events  -   Events of object to be observed
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

    def __init__(self, neurons = None, of = Voltage):
        '''
        Constructor

        INPUTS:
            neurons     -   Population to be monitored
            of          -   Variable to monitor
        '''

        super().__init__()

        if neurons is None:
            self.neurons = False
        else:
            self.neurons = neurons

        self.of = of

    def state_change(self, state, events):
        '''
        Step function

        INPUTS:
            state   -   States of object to be observed
            events  -   Events of object to be observed
        '''

        if self.neurons is False:
            return False

        self.state = np.hstack((self.state, np.array(state[self.neurons,self.of]))) if self.state.shape[0] > 0 else np.array(state[self.neurons,self.of])
