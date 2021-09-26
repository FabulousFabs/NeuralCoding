import numpy as np

from .neurons import *
from .synapses import *
from .neurons_labels import *

class Network:
    '''
    Network class
    '''

    def __init__(self, neuron_prototype = LIF, build_from = None):
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
                n = np.where(self.neurons[:,PARAM_UNI.struct.value] == struct)[0]
                neurons = np.hstack((neurons, n)) if neurons.shape[0] > 0 else np.array(n)

            return neurons.reshape((neurons.shape[0], 1))

        return np.where(self.neurons[:,PARAM_UNI.struct.value] == structure)[0]

    def synapses_in(self, fibre):
        '''
        Get synapses in fibre

        INPUTS:
            fibre   -   Fibre id

        OUTPUTS:
            idx     -   Synapse ids
        '''

        if type(fibre) == np.ndarray:
            synapses = np.array([])

            for f in fibre:
                n = np.where(self.synapses[:,0] == f)[0]
                synapses = np.hstack((synapses, n)) if synapses.shape[0] > 0 else np.array(n)

            return synapses.reshape((synapses.shape[0], 1))

        return np.where(self.synapses[:,0] == fibre)[0]


    def structure(self, n = 1, t = LIF, inhib_ff = 0.0, inhib_fb = 0.0, **kwargs):
        '''
        Add new structure

        INPUTS:
            n                   - Number of neurons to create
            t                   - Neuron type to use for this structure
            inhib_ff            - Degree of lateral feedforward inhibition (0 to disable)
            inhib_fb            - Degree of lateral feedback inhibition (0 to disable)
            **kwargs            - Arguments to pass to neuron constructor (opts)

        OUTPUTS:
            struct  -   Structure id
        '''

        tt = t(opts = kwargs)
        new = tt.params()
        struct = self.structs.shape[0]
        new[0,PARAM_UNI.struct.value] = struct
        new[0,PARAM_UNI.type.value] = tt.type
        new[0,PARAM_UNI.inhib_ff.value] = inhib_ff
        new[0,PARAM_UNI.inhib_fb.value] = inhib_fb
        new = np.tile(new, (n, 1))
        self.neurons = np.vstack((self.neurons, new)) if self.neurons.shape[0] > 0 else np.array(new)
        self.structs = np.vstack((self.structs, struct)) if self.structs.shape[0] > 0 else np.array([struct])

        return struct

    def fibre(self, pre = None, post = None, type = None, plasticity = None):
        '''
        Add new fibre

        INPUTS:
            pre         - Presynaptic layer
            post        - Postsynaptic layer
            type        - Connection type
            plasticity  - Plasticity rule to use on this fibre

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

        self.neurons[:,PARAM_UNI.V.value] = self.neurons[:,PARAM_UNI.E_l.value]
        self.neurons[:,PARAM_UNI.I.value] = 0
        self.neurons[:,PARAM_UNI.x.value] = 1e-6
        self.neurons[:,PARAM_UNI.y.value] = 1e-6
        self.neurons[:,PARAM_UNI.w.value] = 0

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
