'''
Unit test for neuromorphic NOR-gate.
'''

import numpy as np
import sys
sys.path.append('./../..')
import snn
import matplotlib.pyplot as plt

# time
T = 5

# setup network
with snn.Network() as net:
    NOR = snn.utils.gates.NOR(net = net)

# setup simulation
sim = snn.Simulator(dt = 1, network = net, solver = snn.solvers.Clean, verbose = False, plasticity = False)
spikes = sim.monitor(monitor_type = snn.monitors.Spikes)

# setup inputs
I = np.zeros((3, T))
I[2,0] = 1
inj_0_0 = [{'inject': I[0,:].reshape((1, T)), 'into': NOR['I0']}, {'inject': I[1,:].reshape((1, T)), 'into': NOR['I1']}, {'inject': I[2,:].reshape((1, T)), 'into': NOR['A']}]

I = np.zeros((3, T))
I[1,0] = 1
I[2,0] = 1
inj_0_1 = [{'inject': I[0,:].reshape((1, T)), 'into': NOR['I0']}, {'inject': I[1,:].reshape((1, T)), 'into': NOR['I1']}, {'inject': I[2,:].reshape((1, T)), 'into': NOR['A']}]

I = np.zeros((3, T))
I[0,0] = 1
I[2,0] = 1
inj_1_0 = [{'inject': I[0,:].reshape((1, T)), 'into': NOR['I0']}, {'inject': I[1,:].reshape((1, T)), 'into': NOR['I1']}, {'inject': I[2,:].reshape((1, T)), 'into': NOR['A']}]

I = np.zeros((3, T))
I[:,0] = 1
inj_1_1 = [{'inject': I[0,:].reshape((1, T)), 'into': NOR['I0']}, {'inject': I[1,:].reshape((1, T)), 'into': NOR['I1']}, {'inject': I[2,:].reshape((1, T)), 'into': NOR['A']}]

# simulate and plot
sim.run(T = T, inputs = inj_0_0)
snn.plots.raster(monitor = spikes, simulator = sim, title = 'Input: 0 | 0')
sim.monitors[spikes].reset()
snn.plots.show()

sim.run(T = T, inputs = inj_0_1)
snn.plots.raster(monitor = spikes, simulator = sim, title = 'Input: 0 | 1')
sim.monitors[spikes].reset()
snn.plots.show()

sim.run(T = T, inputs = inj_1_0)
snn.plots.raster(monitor = spikes, simulator = sim, title = 'Input: 1 | 0')
sim.monitors[spikes].reset()
snn.plots.show()

sim.run(T = T, inputs = inj_1_1)
snn.plots.raster(monitor = spikes, simulator = sim, title = 'Input: 1 | 1')
sim.monitors[spikes].reset()
snn.plots.show()
