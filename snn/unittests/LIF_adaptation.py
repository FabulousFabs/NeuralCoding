'''
Unit test for adaptation currents.

Creates two neurons, neuron A spiking at B at every time step. Creates plots of
the spike train, the firing rate evolution in time and adaptation current w in
time.
'''

import numpy as np
import sys
sys.path.append('./../..')
import snn
import matplotlib.pyplot as plt

with snn.Network(neuron_prototype = snn.neurons.LIF) as net:
    pre = net.structure(n = 1)
    post = net.structure(n = 1, a = 0.0, b = 0.15, tau_k = 25.0)
    conn = net.fibre(pre = pre, post = post,
                     type = snn.synapses.One_To_One(efficacy = 1))

sim = snn.Simulator(dt = 1, network = net, solver = snn.solvers.Clean, verbose = False)
monitor_spiking = sim.monitor(monitor_type = snn.monitors.Spikes)
monitor_w = sim.monitor(monitor_type = snn.monitors.States, of = snn.neurons.PARAM_UNI.w.value)

inject = np.zeros((1, 200))
inject[0,:] = 1
inputs = [{'inject': inject, 'into': pre}]

sim.run(T = inject[0].shape[0], inputs = inputs)

snn.plots.spike_train(monitor = monitor_spiking, simulator = sim)
snn.plots.rate_in_time(monitor = monitor_spiking, simulator = sim, L = 10)
snn.plots.continuous(monitor = monitor_w, simulator = sim)
snn.plots.show()
