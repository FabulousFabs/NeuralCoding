import numpy as np
from .snn_neurons import *
from .snn_synapses import *

class Network:
    '''
    Network class
    '''

    def __init__(self, neuron_prototype = GLIF, build_from = None):
        '''
        Constructor

        INPUTS:
            neuron_prototype    -   Prototypical neuron to use as per snn.neurons
        '''

        self.structs = np.array([])
        self.fibres = np.array([])
        self.neurons = np.array([], dtype = np.float)
        self.synapses = np.array([], dtype = np.float)
        self.transmission = np.array([], dtype = np.float)

        self.neuron_prototype = neuron_prototype
        self.ready_state = False
        self.build_from = build_from

    def __enter__(self):
        '''
        Entry point
        '''

        return self

    def __exit__(self, t, v, tb):
        '''
        Exit point
        '''

        if self.build_from is not None:
            self.load(model = self.build_from)
        self.ready_state = True

    def neurons_in(self, structure):
        '''
        Get neurons of structure

        INPUTS:
            structure   -   Structure id

        OUTPUTS:
            idx     -   Neuron ids
        '''

        if type(structure) == np.ndarray:
            neurons = np.array([])

            for struct in structure:
                n = np.where(self.neurons[:,0] == struct)[0]
                neurons = np.hstack((neurons, n)) if neurons.shape[0] > 0 else np.array(n)

            return neurons.reshape((neurons.shape[0], 1))

        return np.where(self.neurons[:,0] == structure)[0]

    def synapses_in(self, fibre):
        '''
        Get synapses in fibre

        INPUTS:
            fibre   -   Fibre id

        OUTPUTS:
            idx     -   Synapse ids
        '''

        return np.where(self.synapses[:,0] == fibre)[0]


    def structure(self, n = 1, lateral_inhibition = 0.0, **kwargs):
        '''
        Add new structure

        INPUTS:
            n                   - Number of neurons to create
            lateral_inhibition  - Degree of lateral inhibition (0 to disable)

        OUTPUTS:
            struct  -   Structure id
        '''

        new = self.neuron_prototype(opts = kwargs).params
        struct = self.structs.shape[0]
        new[0] = struct
        new[15] = lateral_inhibition
        new = np.tile(new, (n, 1))
        self.neurons = np.vstack((self.neurons, new)) if self.neurons.shape[0] > 0 else np.array(new)
        self.structs = np.vstack((self.structs, struct)) if self.structs.shape[0] > 0 else np.array([struct])

        return struct

    def fibre(self, pre = None, post = None, type = None, plasticity = None):
        '''
        Add new fibre

        INPUTS:
            pre     - Presynaptic layer
            post    - Postsynaptic layer
            type    - Connection type

        OUTPUTS:
            fibre   - Fibre id
        '''

        if pre is None or post is None or type is None: return False

        pre = self.neurons_in(pre)
        post = self.neurons_in(post)
        fibre = self.fibres.shape[0]
        new = type.synapses(pre=pre, post=post)
        new[:,0] = new[:,0] * fibre

        if plasticity is not None:
            new[:,5] = plasticity.rule_id
            new[:,6] = plasticity.lr
            new[:,7:11] = plasticity.free_params
        else:
            new[:,5] = np.nan

        self.synapses = np.vstack((self.synapses, new)) if self.synapses.shape[0] > 0 else np.array(new)
        self.fibres = np.vstack((self.fibres, np.array([fibre]))) if self.fibres.shape[0] > 0 else np.array([fibre])

        return fibre

    def reset(self):
        '''
        Reset membrane potential, incoming currents, pre-, post-synaptic traces and adaptation currents.
        '''

        self.neurons[:,11] = self.neurons[:,1]
        self.neurons[:,12] = 0
        self.neurons[:,13] = 1e-6
        self.neurons[:,14] = 1e-6
        self.neurons[:,15] = 0

    def save(self, to = None):
        '''
        Save the network to disc.

        INPUTS:
            to  -   File to save to.
        '''

        if to is None: return False

        with open(to, 'wb') as f:
            np.save(f, self.structs)
            np.save(f, self.fibres)
            np.save(f, self.neurons)
            np.save(f, self.synapses)

        return True

    def load(self, model = None):
        '''
        Load the network to disc.

        INPUTS:
            model    - File to load from.
        '''

        if model is None: return False

        with open(model, 'rb') as f:
            self.structs = np.load(f, allow_pickle = True)
            self.fibres = np.load(f, allow_pickle = True)
            self.neurons = np.load(f, allow_pickle = True)
            self.synapses = np.load(f, allow_pickle = True)

        return True
