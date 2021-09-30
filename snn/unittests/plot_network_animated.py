'''
Unit test for animating the network through an epoch.

Creates a simple network of A (rate coding) and D (phase coding) that fire at
B (classifier) that is inhibited by the feedback population C. The network is
simulated for arbitrary inputs and animated. See plot_network_animated.gif for
outputs.
'''

import numpy as np
import sys
sys.path.append('./../..')
import snn
import matplotlib.pyplot as plt

with snn.Network() as net:
    A = net.structure(n = 10)
    B = net.structure(n = 25)
    C = net.structure(n = 25)
    D = net.structure(n = 10)

    conn1 = net.fibre(pre = A, post = B,
                      type = snn.synapses.Full(n = 10, generator = snn.generators.Xavier, directionality = 1))
    conn2 = net.fibre(pre = B, post = C,
                      type = snn.synapses.One_To_One(efficacy = 1))
    conn3 = net.fibre(pre = C, post = B,
                      type = snn.synapses.kWTA(n = 25, generator = snn.generators.Xavier, directionality = -1))
    conn4 = net.fibre(pre = D, post = B,
                      type = snn.synapses.Full(n = 10, generator = snn.generators.Xavier, directionality = 1))

T = 24
input_pattern = np.array([250, 150, 180, 125, 114, 198, 116, 127, 115, 114])
input_rate = snn.utils.neuralcoding.rate(inputs = input_pattern.reshape((10, 1)), L = T, lam = 2).astype(np.float)
input_phase = snn.utils.neuralcoding.phase(inputs = np.tile(input_pattern.reshape(10, 1), (1, 3)), L = T, bits = 8).astype(np.float)

ins = [{'inject': input_rate, 'into': A}, {'inject': input_phase, 'into': D}]

sim = snn.Simulator(dt = 1, network = net, solver = snn.solvers.Clean, verbose = True, plasticity = False)
spikes = sim.monitor(monitor_type = snn.monitors.Spikes)
sim.run(T = T, inputs = ins)

snn.plots.network(net = net, cmap_neurons_by_state = True, cmap_neurons_sim = sim, cmap_neurons_monitor = spikes, cmap_anim_save_to = './plot_network_animated.gif')
