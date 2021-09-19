import numpy as np
from .snn_solvers import *
from .snn_monitors import *
from . import snn_plasticity as plasticity

class Simulator:
    '''
    Simulator class
    '''

    def __init__(self, dt = 0.001, network = None, solver = RungeKutta,
                       verbose = False, plasticity = True):
        '''
        Constructor

        INPUTS:
            dt      -   Time step size
            network -   Network class to simulate
            solver  -   Solver function to use for diffeqs
            verbose -   Logging state (true/false)
        '''

        if network is None: return False

        self.dt = dt
        self.network = network
        self.solver = solver
        self.verbose = verbose
        self.monitors = []
        self.plasticity = plasticity

    def monitor(self, monitor_type = None, structures = None, of = None, is_synapse = False):
        '''
        Add a monitor

        INPUTS:
            monitor_type    -   Monitor type as per snn.monitors
            structures      -   Structures to be monitored (None = all)
            of              -   If continuous monitor, what target?
            is_synapse      -   Are we monitoring a synapse?
        '''

        if self.network.ready_state is False: return False
        if is_synapse is True and of is None: return False

        if is_synapse is False:
            if structures is None: structures = self.network.structs
            targets = self.network.neurons_in(structures)
        else:
            if structures is None: structures = self.network.fibres
            targets = self.network.synapses_in(structures)

        if of is None:
            self.monitors.append(monitor_type(targets = targets))
        else:
            self.monitors.append(monitor_type(targets = targets, of = of, is_synapse = is_synapse))

        return len(self.monitors)-1


    def run(self, T = 1, inputs = None, reset = True):
        '''
        Run a simulation

        INPUTS:
            T       -   Duration of simulation
            inputs  -   Input structure for the simulation that is an array of dictionaries of inject:pattern into:structure.
            reset   -   Reset network state before running current simulation (True/False).
        '''

        # check network is ready
        if self.network.ready_state is False:
            print('Network is not ready yet. Run simulations after building the network.')
            return False


        # logging
        if self.verbose is True:
            print('Starting simulation, dt = {:f}, T = {:f}.'.format(self.dt, T))
            print('To turn off logging, use `verbose = False`.')
            print('\n')


        # reset neural states
        if reset is True:
            self.network.reset()


        # setup prototypes and transmission
        proto = self.network.neuron_prototype()
        self.network.transmission = np.zeros((self.network.neurons.shape[0], np.ceil(T / self.dt).astype(np.int)))


        # setup 'clamping' for input pattern
        if inputs is not None:
            for input in inputs:
                neurons = self.network.neurons_in(input['into'])
                ins = np.array(input['inject'], dtype=np.int)

                assert(ins.shape[1] <= self.network.transmission.shape[1])
                assert(ins.shape[0] == neurons.shape[0])

                L = self.network.transmission.shape[1]
                pattern = np.pad(ins, ((0,0), (0, L - ins.shape[1])), 'constant', constant_values = 0)
                self.network.transmission[neurons] += self.network.neurons[neurons,6].reshape((neurons.shape[0], 1)) * pattern


        # main integration loop
        for t in np.arange(0, T, self.dt):
            # logging
            if self.verbose is True:
                progress = t / T
                print('[\t\t\t\t\t]', end='\r')
                print('[' + ''.join(['-' for i in range(int(progress * 40))]), end='\r')
                print('\t\t{:2.2f}%.'.format(np.round(progress*100, 2)), end='\r')


            # update lateral inhibition (feedforward)
            for struct in self.network.structs:
                neurons = self.network.neurons_in(struct)
                avg = np.mean(self.network.neurons[neurons,13])
                self.network.transmission[neurons,int(t/self.dt)] += (self.network.neurons[neurons,15] / 2) * (-avg)


            # setup current
            self.network.neurons[:,12] = self.network.transmission[:,int(t/self.dt)]


            # integrate membrane potential
            dVdt = self.solver(self.network.neurons[:,11], t, self.dt, proto.dVdt, neurons = self.network.neurons)
            self.network.neurons[:,11] = proto.V_apply(self.network.neurons, self.dt, dVdt)


            # detect spikes (in spite of floating point issues)
            spike_def = self.network.neurons[:,11] > (self.network.neurons[:,6] + self.network.neurons[:,16])
            spike_may = np.isclose(self.network.neurons[:,11], (self.network.neurons[:,6] + self.network.neurons[:,16]), rtol=1e-3, atol=1e-3)
            spike_indx = np.where(spike_def | spike_may)[0]


            # update adaptation of neurons
            dwdt = self.solver(self.network.neurons[:,16], t, self.dt, proto.dwdt, neurons = self.network.neurons, spike_indx = spike_indx)
            self.network.neurons[:,16] = self.network.neurons[:,16] + (self.dt * dwdt)


            # update pre-syn traces
            dxdt = self.solver(self.network.neurons[:,13], t, self.dt, proto.dxdt, neurons = self.network.neurons, spike_indx = spike_indx)
            self.network.neurons[:,13] = self.network.neurons[:,13] + (self.dt * dxdt)


            # update post-syn traces
            dydt = self.solver(self.network.neurons[:,14], t, self.dt, proto.dydt, neurons = self.network.neurons, spike_indx = spike_indx)
            self.network.neurons[:,14] = self.network.neurons[:,14] + (self.dt * dydt)


            # push to monitors
            [m.state_change(self.network.neurons, spike_indx, self.network.synapses) for m in self.monitors]


            # reset currents
            self.network.neurons[:,12] = np.zeros((self.network.neurons.shape[0],))


            # propagate spikes
            for pre in spike_indx:
                if self.network.synapses.shape[0] < 1:
                    break

                conns = np.where(self.network.synapses[:,1] == pre)[0].astype(np.int)

                # check it has outputs
                if conns.shape[0] < 1:
                    continue

                # check outputs are within T
                if np.any(self.network.synapses[conns,4].astype(np.int)+int(t/self.dt) > self.network.transmission.shape[1]-1):
                    continue

                # get post and transmit
                post = self.network.synapses[conns,2].astype(np.int)
                outs = self.network.neurons[pre,:].reshape((1, proto.params.shape[0]))
                outs = np.tile(outs, (conns.shape[0], 1))
                self.network.transmission[post,self.network.synapses[conns,4].astype(np.int)+int(t/self.dt)] += self.network.synapses[conns,3] * (np.ones((conns.shape[0],)) * proto.It(outs))


            '''
            # update pre-syn traces
            received_spike = np.zeros((self.network.neurons.shape[0],))

            for pre in spike_indx:
                if self.network.synapses.shape[0] < 1:
                    break

                conns = np.where(self.network.synapses[:,1] == pre)[0].astype(np.int)

                # check it has outputs
                if conns.shape[0] < 1:
                    continue

                received_spike[self.network.synapses[conns,2].astype(np.int)] = 1

            dxdt = self.solver(self.network.neurons[:,13], t, self.dt, proto.dxdt, neurons = self.network.neurons, spike_indx = received_spike)
            self.network.neurons[:,13] = self.network.neurons[:,13] + (self.dt * dxdt)



            # update pre-syn traces for inputs
            if inputs is not None:
                for input in inputs:
                    neurons = self.network.neurons_in(input['into'])
                    ins = np.array(input['inject'], dtype=np.int)

                    L = self.network.transmission.shape[1]
                    pattern = np.pad(ins, ((0,0), (0, L - ins.shape[1])), 'constant', constant_values = 0)

                    dxdt = self.solver(self.network.neurons[neurons,13], t, self.dt, proto.dxdt, neurons = self.network.neurons[neurons,:], spike_indx = pattern[:,int(t/self.dt)])
                    self.network.neurons[neurons,13] = self.network.neurons[neurons,13] + (self.dt * dxdt)
            '''


            # update lateral inhibition (feedback)
            for struct in self.network.structs:
                if t < T-self.dt:
                    neurons = self.network.neurons_in(struct)
                    avg = np.mean(self.network.neurons[neurons,14])
                    self.network.transmission[neurons,int(t/self.dt)+1] += (self.network.neurons[neurons,15] / 2) * (-avg)


            # plasticity rules
            if self.plasticity is True:
                for fibre in self.network.fibres:
                    synapses = self.network.synapses_in(fibre).astype(np.int)

                    if np.all(np.isnan(self.network.synapses[synapses,5]) == False):
                        pre = self.network.neurons[self.network.synapses[synapses,1].astype(np.int),:]
                        post = self.network.neurons[self.network.synapses[synapses,2].astype(np.int),:]

                        # floating point work-around
                        rule = np.round(np.mean(self.network.synapses[synapses,5])).astype(np.int)

                        # updates
                        s_x = np.isin(self.network.synapses[synapses,1], spike_indx)
                        s_y = np.isin(self.network.synapses[synapses,2], spike_indx)
                        dwdt = self.solver(self.network.synapses[synapses,3], t, self.dt, plasticity.Rules[rule].dwdt, pre = pre, post = post, synapses = self.network.synapses[synapses], s_x = s_x, s_y = s_y)
                        dwdt[np.where(np.isnan(dwdt))] = 0.0
                        self.network.synapses[synapses,3] += self.dt * dwdt


            # reset potentials
            self.network.neurons[spike_indx,11] = self.network.neurons[spike_indx,1]


            # check for diverging values
            div = np.where(self.network.neurons[:,11] <= (-3 * np.abs(self.network.neurons[:,6])))[0].astype(np.int)
            self.network.neurons[div,11] = self.network.neurons[div,1]
