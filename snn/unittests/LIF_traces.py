'''
Unit test for pre- and post-synaptic spike traces.

Creates two neurons, with neuron A firing at B once at t = 5, then gives a raster
plot of the spiking behaviour of the neurons as well as continuous measurements of
the pre- and postsynaptic spike traces monitored.
'''

import numpy as np
import sys
sys.path.append('./../..')
import snn
import matplotlib.pyplot as plt

with snn.Network(neuron_prototype = snn.neurons.LIF) as net:
    pre = net.structure(n = 1)
    post = net.structure(n = 1)
    conn = net.fibre(pre = pre, post = post,
                     type = snn.synapses.One_To_One(efficacy = 1))

sim = snn.Simulator(dt = 1, network = net, solver = snn.solvers.Clean, verbose = False)
monitor_spiking = sim.monitor(monitor_type = snn.monitors.Spikes)
monitor_pre = sim.monitor(monitor_type = snn.monitors.States, of = snn.neurons.PARAM_UNI.x.value)
monitor_post = sim.monitor(monitor_type = snn.monitors.States, of = snn.neurons.PARAM_UNI.y.value)

inject = np.zeros((1, 15))
inject[0,5] = 1
inputs = [{'inject': inject, 'into': pre}]

sim.run(T = inject[0].shape[0], inputs = inputs)

snn.plots.raster(monitor = monitor_spiking, simulator = sim)
snn.plots.continuous(monitor = monitor_pre, simulator = sim)
snn.plots.continuous(monitor = monitor_post, simulator = sim)
snn.plots.show()
