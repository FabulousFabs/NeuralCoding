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
monitor_pre = sim.monitor(monitor_type = snn.monitors.States, of = 13)
monitor_post = sim.monitor(monitor_type = snn.monitors.States, of = 14)

inject = np.zeros((1, 15))
inject[0,5] = 1
inputs = [{'inject': inject, 'into': pre}]

sim.run(T = inject[0].shape[0], inputs = inputs)

snn.plots.raster(monitor = monitor_spiking, simulator = sim)
snn.plots.continuous(monitor = monitor_pre, simulator = sim)
snn.plots.continuous(monitor = monitor_post, simulator = sim)
snn.plots.show()
