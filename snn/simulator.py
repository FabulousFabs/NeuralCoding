import numpy as np
import time

from .solvers import *
from .monitors import *
from . import plasticity
from . import neurons as Neurons
from . import synapses as Synapses
from .neurons_labels import *

class Simulator:
    '''
    Simulator class
    '''

    def __init__(self, dt = 1, network = None, solver = Clean,
                       verbose = False, plasticity = True):
        '''
        Constructor

        INPUTS:
            dt          - Time step size
            network     - Network class to simulate
            solver      - Solver function to use for diffeqs
            verbose     - Logging state (True/False)
            plasticity  - Allow plasticity in this simulator? (True/False)
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
            self.monitors.append(monitor_type(targets = targets, is_synapse = is_synapse))

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
        proto = Neurons.Prototype()
        self.network.transmission = np.zeros((self.network.neurons.shape[0], np.ceil(T / self.dt).astype(np.int32)))


        # setup 'clamping' for input pattern
        if inputs is not None:
            for input in inputs:
                neurons = self.network.neurons_in(input['into'])
                ins = np.array(input['inject'], dtype=np.int32)

                assert(ins.shape[1] <= self.network.transmission.shape[1])
                assert(ins.shape[0] == neurons.shape[0])

                L = self.network.transmission.shape[1]
                pattern = np.pad(ins, ((0,0), (0, L - ins.shape[1])), 'constant', constant_values = 0)
                self.network.transmission[neurons] += (self.network.neurons[neurons,PARAM_UNI.V_thr.value] - self.network.neurons[neurons,PARAM_UNI.V.value]).reshape((neurons.shape[0], 1)) * pattern


        # main integration loop
        for t in np.arange(0, T, self.dt):
            # logging
            if self.verbose is True:
                progress = t / T
                print('[\t\t\t\t\t]', end='\r')
                print('[' + ''.join(['-' for i in range(int(progress * 40))]), end='\r')
                print('\t\t{:2.2f}%.'.format(np.round(progress*100, 2)), end='\r')


            # setup current
            self.network.neurons[:,PARAM_UNI.I.value] = self.network.transmission[:,int(t/self.dt)]


            # update synaptic filters (pre step)
            for f_t in Synapses.filters.Filters:
                f_i_t = np.where(np.round(self.network.synapses_filters[:,0]) == Synapses.filters.Filters[f_t].type)[0]

                # check for filters of this type
                if f_i_t.shape[0] < 1:
                    continue

                self.network.synapses_filters[f_i_t,:], self.network.synapses[f_i_t,:] = Synapses.filters.Filters[f_t].pre(t, self.dt, self.network.synapses_filters[f_i_t,:], self.network.synapses[f_i_t,:], self.network.synapses)


            # update lateral inhibition (feedforward)
            for struct in self.network.structs:
                neurons = self.network.neurons_in(struct)
                no = neurons.shape[0]
                neurons = np.squeeze(neurons).reshape((no,))
                dffdt = self.solver(self.network.neurons[neurons,PARAM_UNI.ff.value], t, self.dt, proto.dffdt, neurons = self.network.neurons[neurons,:])
                self.network.neurons[neurons,PARAM_UNI.ff.value] = self.dt * dffdt


            # integrate membrane potential per neuron type
            for n_t in Neurons.Neurons:
                n_i_t = np.where(np.round(self.network.neurons[:,PARAM_UNI.type.value]) == Neurons.Neurons[n_t].type)[0]

                # check for neurons of this type
                if n_i_t.shape[0] < 1:
                    continue

                dVdt = self.solver(self.network.neurons[:,PARAM_UNI.V.value], t, self.dt, Neurons.Neurons[n_t].dVdt, neurons = self.network.neurons[n_i_t,:])
                self.network.neurons[n_i_t,PARAM_UNI.V.value] = Neurons.Neurons[n_t].V_apply(self.network.neurons[n_i_t,:], self.dt, dVdt)


            # detect spikes (in spite of floating point issues)
            spike_def = self.network.neurons[:,PARAM_UNI.V.value] > (self.network.neurons[:,PARAM_UNI.V_thr.value] + self.network.neurons[:,PARAM_UNI.w.value])
            spike_may = np.isclose(self.network.neurons[:,PARAM_UNI.V.value], (self.network.neurons[:,PARAM_UNI.V_thr.value] + self.network.neurons[:,PARAM_UNI.w.value]), rtol=1e-3, atol=1e-3)
            spike_indx = np.where(spike_def | spike_may)[0]


            # update adapatation by neuron type
            for n_t in Neurons.Neurons:
                n_i_t = np.where(np.round(self.network.neurons[:,PARAM_UNI.type.value]) == Neurons.Neurons[n_t].type)[0]

                # check for neurons of this type
                if n_i_t.shape[0] < 1:
                    continue

                s = np.isin(n_i_t, spike_indx)
                dwdt = self.solver(self.network.neurons[:,PARAM_UNI.w.value], t, self.dt, Neurons.Neurons[n_t].dwdt, neurons = self.network.neurons[n_i_t,:], spike_indx = s)
                self.network.neurons[n_i_t,PARAM_UNI.w.value] = self.network.neurons[n_i_t,PARAM_UNI.w.value] + (self.dt * dwdt)


            # update pre-syn traces
            dxdt = self.solver(self.network.neurons[:,PARAM_UNI.x.value], t, self.dt, proto.dxdt, neurons = self.network.neurons, spike_indx = spike_indx)
            self.network.neurons[:,PARAM_UNI.x.value] = self.network.neurons[:,PARAM_UNI.x.value] + (self.dt * dxdt)


            # update post-syn traces
            dydt = self.solver(self.network.neurons[:,PARAM_UNI.y.value], t, self.dt, proto.dydt, neurons = self.network.neurons, spike_indx = spike_indx)
            self.network.neurons[:,PARAM_UNI.y.value] = self.network.neurons[:,PARAM_UNI.y.value] + (self.dt * dydt)


            # update synaptic filters (post step)
            for f_t in Synapses.filters.Filters:
                f_i_t = np.where(np.round(self.network.synapses_filters[:,0]) == Synapses.filters.Filters[f_t].type)[0]

                # check for filters of this type
                if f_i_t.shape[0] < 1:
                    continue

                self.network.synapses_filters[f_i_t,:], self.network.synapses[f_i_t,:] = Synapses.filters.Filters[f_t].post(t, self.dt, self.network.synapses_filters[f_i_t,:], self.network.synapses[f_i_t,:], self.network.synapses)


            # push to monitors
            [m.state_change(self.network.neurons, spike_indx, self.network.synapses) for m in self.monitors]


            # reset currents
            self.network.neurons[:,PARAM_UNI.I.value] = np.zeros((self.network.neurons.shape[0],))


            # propagate spikes
            for n_t in Neurons.Neurons:
                # get neurons of this type
                n_i_t = np.where(np.round(self.network.neurons[:,PARAM_UNI.type.value]) == Neurons.Neurons[n_t].type)[0]

                # check for neurons of this type
                if n_i_t.shape[0] < 1:
                    continue

                # get spiking neurons of this type
                s = n_i_t[np.isin(n_i_t, spike_indx)]

                # check for spikes
                if s.shape[0] < 1:
                    continue


                # get active synapses
                a_s = np.isin(self.network.synapses[:,1], s)

                # control activity
                if self.network.synapses[a_s,:].shape[0] < 1:
                    continue

                # get outputs within T
                v_s = np.where(self.network.synapses[a_s,4].astype(np.int32) + int(t / self.dt) <= self.network.transmission.shape[1]-1)[0]

                # control outputs within T
                if v_s.shape[0] < 1:
                    continue

                # combine masks
                a_s[np.where(a_s == True)[0][v_s]] = True
                a_s = np.where(a_s == True)[0].astype(np.int32)

                # get pre, post and transmit
                pre = self.network.neurons[self.network.synapses[a_s,1].astype(np.int32),:]
                post = self.network.synapses[a_s,2].astype(np.int32)
                self.network.transmission[post, self.network.synapses[a_s,4].astype(np.int32) + int(t / self.dt)] += self.network.synapses[a_s,3] * (np.ones((pre.shape[0],)) * Neurons.Neurons[n_t].It(pre))


            # update pre-syn traces (arrival)
            received_spike = np.isin(np.arange(0, self.network.neurons.shape[0], 1), self.network.synapses[np.isin(self.network.synapses[:,1], spike_indx), 2])
            received_spike = np.where(received_spike == True)[0]

            dxpdt = self.solver(self.network.neurons[:,PARAM_UNI.xp.value], t, self.dt, proto.dxdt, neurons = self.network.neurons, spike_indx = received_spike)
            self.network.neurons[:,PARAM_UNI.xp.value] = self.network.neurons[:,PARAM_UNI.xp.value] + (self.dt * dxpdt)


            # update lateral inhibition (feedback)
            for struct in self.network.structs:
                neurons = self.network.neurons_in(struct)
                no = neurons.shape[0]
                neurons = np.squeeze(neurons).reshape((no,))
                dfbdt = self.solver(self.network.neurons[neurons,PARAM_UNI.fb.value], t, self.dt, proto.dfbdt, neurons = self.network.neurons[neurons,:])
                self.network.neurons[neurons,PARAM_UNI.fb.value] = self.network.neurons[neurons,PARAM_UNI.fb.value] + self.dt * dfbdt


            # plasticity rules
            if self.plasticity is True:
                for fibre in self.network.fibres:
                    synapses = self.network.synapses_in(fibre).astype(np.int32)

                    if np.all(np.isnan(self.network.synapses[synapses,5]) == False):
                        pre = self.network.neurons[self.network.synapses[synapses,1].astype(np.int32),:]
                        pre = np.squeeze(pre).reshape((pre.shape[0], self.network.neurons.shape[1]))
                        post = self.network.neurons[self.network.synapses[synapses,2].astype(np.int32),:]
                        post = np.squeeze(post).reshape((post.shape[0], self.network.neurons.shape[1]))
                        syn = self.network.synapses[synapses,:]
                        syn = np.squeeze(syn).reshape((synapses.shape[0], self.network.synapses.shape[1]))

                        # floating point work-around
                        rule = np.round(np.mean(syn[:,5])).astype(np.int32)

                        # updates
                        s_x = np.isin(syn[:,1], spike_indx)
                        s_y = np.isin(syn[:,2], spike_indx)
                        dwdt = self.solver(syn[:,3], t, self.dt, plasticity.Rules[rule].dwdt, pre = pre, post = post, synapses = syn, s_x = s_x, s_y = s_y).reshape((syn.shape[0], 1))
                        dwdt[np.where(np.isnan(dwdt))] = 0.0
                        self.network.synapses[synapses,3] += self.dt * dwdt


            # reset potentials
            self.network.neurons[spike_indx,PARAM_UNI.V.value] = self.network.neurons[spike_indx,PARAM_UNI.E_l.value]


            # check for diverging values
            div = np.where(self.network.neurons[:,PARAM_UNI.V.value] <= (-3 * np.abs(self.network.neurons[:,PARAM_UNI.V_thr.value])))[0].astype(np.int32)
            self.network.neurons[div,PARAM_UNI.V.value] = self.network.neurons[div,1]
