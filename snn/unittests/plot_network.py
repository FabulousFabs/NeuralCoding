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

with snn.Network() as net:
    A = net.structure(n = 10)
    B = net.structure(n = 25)
    C = net.structure(n = 25)

    conn1 = net.fibre(pre = A, post = B,
                      type = snn.synapses.Full(n = 10, generator = snn.generators.Xavier, directionality = 1))
    conn2 = net.fibre(pre = B, post = C,
                      type = snn.synapses.One_To_One(efficacy = 1))
    conn3 = net.fibre(pre = C, post = B,
                      type = snn.synapses.kWTA(n = 25, generator = snn.generators.Xavier, directionality = -1))

snn.plots.network(net = net)
snn.plots.show()
