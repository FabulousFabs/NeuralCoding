'''
Unit test for lateral inhibition.

Creates two populations (A = 100, B = 25) of neurons, with A firing at B for a
sparse representation of A. This showcases the inhibition vs no-inhibition
cases for ff and fb.
'''

import numpy as np
import sys
sys.path.append('./../..')
import snn
import matplotlib.pyplot as plt

# setup degree of inhibition to test
G_i = 1.8

# feedforward activated
with snn.Network() as net:
    pre = net.structure(n = 100)
    post = net.structure(n = 25, inhib_ff = G_i, inhib_fb = 0.0)
    conn = net.fibre(pre = pre, post = post,
                     type = snn.synapses.Full(n = 100, generator = snn.generators.Xavier, directionality = 1))

sim = snn.Simulator(dt = 1, network = net, solver = snn.solvers.Clean, verbose = False)
monitor_spiking = sim.monitor(monitor_type = snn.monitors.Spikes)
monitor_ff = sim.monitor(monitor_type = snn.monitors.States, of = snn.neurons.PARAM_UNI.ff.value)

inject = snn.generators.Poisson(homogenous = False, rf = lambda x: 25 * np.abs(np.sin(4 * np.pi * x * 1e-3)), dim = (100, 300))
inputs = [{'inject': inject, 'into': pre}]

sim.run(T = inject[0].shape[0], inputs = inputs)

snn.plots.spike_train(monitor = monitor_spiking, simulator = sim, title = 'Spike train (feedforward = True)')
snn.plots.continuous(monitor = monitor_ff, simulator = sim, title = 'Feedforward inhibition (feedforward = True)')


# feedback activated
with snn.Network() as net:
    pre = net.structure(n = 100)
    post = net.structure(n = 25, inhib_ff = 0.0, inhib_fb = G_i)
    conn = net.fibre(pre = pre, post = post,
                     type = snn.synapses.Full(n = 100, generator = snn.generators.Xavier, directionality = 1))

sim = snn.Simulator(dt = 1, network = net, solver = snn.solvers.Clean, verbose = False)
monitor_spiking = sim.monitor(monitor_type = snn.monitors.Spikes)
monitor_fb = sim.monitor(monitor_type = snn.monitors.States, of = snn.neurons.PARAM_UNI.fb.value)

sim.run(T = inject[0].shape[0], inputs = inputs)

snn.plots.spike_train(monitor = monitor_spiking, simulator = sim, title = 'Spike train (feedback = True)')
snn.plots.continuous(monitor = monitor_fb, simulator = sim, title = 'Feedback inhibition (feedback = True)')


# feedforward and feedback activated
with snn.Network() as net:
    pre = net.structure(n = 100)
    post = net.structure(n = 25, inhib_ff = G_i, inhib_fb = G_i)
    conn = net.fibre(pre = pre, post = post,
                     type = snn.synapses.Full(n = 100, generator = snn.generators.Xavier, directionality = 1))

sim = snn.Simulator(dt = 1, network = net, solver = snn.solvers.Clean, verbose = False)
monitor_spiking = sim.monitor(monitor_type = snn.monitors.Spikes)
monitor_ff = sim.monitor(monitor_type = snn.monitors.States, of = snn.neurons.PARAM_UNI.ff.value)
monitor_fb = sim.monitor(monitor_type = snn.monitors.States, of = snn.neurons.PARAM_UNI.fb.value)

sim.run(T = inject[0].shape[0], inputs = inputs)

snn.plots.spike_train(monitor = monitor_spiking, simulator = sim, title = 'Spike train (Feedback and -forward = True)')
snn.plots.continuous(monitor = monitor_ff, simulator = sim, title = 'Feedforward inhibition (ff = True, fb = True)')
snn.plots.continuous(monitor = monitor_fb, simulator = sim, title = 'Feedback inhibition (ff = True, fb = True)')


# fully deactivated
with snn.Network() as net:
    pre = net.structure(n = 100)
    post = net.structure(n = 25, inhib_ff = 0.0, inhib_fb = 0.0)
    conn = net.fibre(pre = pre, post = post,
                     type = snn.synapses.Full(n = 100, generator = snn.generators.Xavier, directionality = 1))

sim = snn.Simulator(dt = 1, network = net, solver = snn.solvers.Clean, verbose = False)
monitor_spiking = sim.monitor(monitor_type = snn.monitors.Spikes)

sim.run(T = inject[0].shape[0], inputs = inputs)

snn.plots.spike_train(monitor = monitor_spiking, simulator = sim, title = 'Spike train (Feedback and -forward = False)')
snn.plots.show()
