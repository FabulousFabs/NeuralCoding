'''
Unit test for Bienenstock-Cooper-Munro rule.

Creates 50 neurons for A and B each, with A firing at B. The relative
postsynaptic traces are manipulated such that sometimes the neuron of
interest fires under or over the mean V_th. Plots are created that
show the degree of learning that takes place given V - V_th. This should
show that LTD occurs if V < V_th but that a switches to LTP occurs as
soon as V > V_th.
'''

import numpy as np
import sys
sys.path.append('./../..')
import snn
import matplotlib.pyplot as plt

vtl = np.array([])
dwl = np.array([])

T = 50

with snn.Network() as net:
    ins = net.structure(n = 50)
    outs = net.structure(n = 50)
    conn = net.fibre(pre = ins, post = outs,
                     type = snn.synapses.One_To_One(),
                     plasticity = snn.plasticity.BCM())

for i in np.arange(0, T, 1):
    sim = snn.Simulator(dt = 1, network = net, solver = snn.solvers.Clean, verbose = False)
    monitor = sim.monitor(monitor_type = snn.monitors.States, of = 3, is_synapse = True)
    monitor2 = sim.monitor(monitor_type = snn.monitors.States, of = snn.neurons.PARAM_UNI.y.value, structures = np.array([ins]))

    a = T-i
    inject = np.zeros((T,T))
    inject[0:-2,:] = 1
    inject[-1,0:-1:a] = 1
    inputs = [{'inject': inject, 'into': ins}]

    sim.run(T = inputs[0]['inject'].shape[1], inputs = inputs)

    vtl = np.append(vtl, sim.monitors[monitor2].state[-1,-1] - np.mean(sim.monitors[monitor2].state[:,-1]))
    dwl = np.append(dwl, (sim.monitors[monitor].state[-1,-1] - sim.monitors[monitor].state[0,0]))

vth = np.array([])
dwh = np.array([])

with snn.Network() as net:
    ins = net.structure(n = 50)
    outs = net.structure(n = 50)
    conn = net.fibre(pre = ins, post = outs,
                     type = snn.synapses.One_To_One(),
                     plasticity = snn.plasticity.BCM())

for i in np.arange(0, T, 1):
    sim = snn.Simulator(dt = 1, network = net, solver = snn.solvers.Clean, verbose = False)
    monitor = sim.monitor(monitor_type = snn.monitors.States, of = 3, is_synapse = True)
    monitor2 = sim.monitor(monitor_type = snn.monitors.States, of = snn.neurons.PARAM_UNI.y.value, structures = np.array([outs]))

    a = T-i
    inject = np.zeros((T,T))
    inject[0:-2,0:-1:i+1] = 1
    inject[-1,:] = 1
    inputs = [{'inject': inject, 'into': ins}]

    sim.run(T = inputs[0]['inject'].shape[1], inputs = inputs)

    vth = np.append(vth, sim.monitors[monitor2].state[-1,-1] - np.mean(sim.monitors[monitor2].state[:,-1]))
    dwh = np.append(dwh, (sim.monitors[monitor].state[-1,-1] - sim.monitors[monitor].state[0,0]))

plt.plot(vtl, dwl)
plt.plot(vth, dwh)
plt.xlabel('V - V_th')
plt.ylabel('dwdt')
plt.show()
