import numpy as np
import sys
sys.path.append('./../..')
import snn
import matplotlib.pyplot as plt

T = 50

dt = np.array([])
dw = np.array([])
dt2 = np.array([])
dw2 = np.array([])

for i in np.arange(1, T+1, 1):
    with snn.Network(neuron_prototype = snn.neurons.LIF) as net:
        pre = net.structure(n = 1)
        post = net.structure(n = 1)
        conn = net.fibre(pre = pre, post = post,
                         type = snn.synapses.One_To_One(efficacy = 1, delay = i),
                         plasticity = snn.plasticity.Oja())

    sim = snn.Simulator(dt = 1, network = net, solver = snn.solvers.Clean, verbose = False)
    monitor = sim.monitor(monitor_type = snn.monitors.States, of = 3, is_synapse = True)
    monitor2 = sim.monitor(monitor_type = snn.monitors.Spikes)

    inject = np.zeros((1, i+2))
    inject[0,0] = 1
    inputs = [{'inject': inject, 'into': pre}]

    sim.run(T = inject[0].shape[0], inputs = inputs)

    dt = np.append(dt, -i)
    dw = np.append(dw, sim.monitors[monitor].state[0,-1] - sim.monitors[monitor].state[0,0])

for i in np.arange(1, T+1, 1):
    with snn.Network(neuron_prototype = snn.neurons.LIF) as net:
        pre = net.structure(n = 1)
        post = net.structure(n = 1)
        conn = net.fibre(pre = pre, post = post,
                         type = snn.synapses.One_To_One(efficacy = 1, delay = 1),
                         plasticity = snn.plasticity.Oja())

    sim = snn.Simulator(dt = 1, network = net, solver = snn.solvers.Clean, verbose = False)
    monitor = sim.monitor(monitor_type = snn.monitors.States, of = 3, is_synapse = True)
    monitor2 = sim.monitor(monitor_type = snn.monitors.Spikes)

    inject = np.zeros((1, i+2))
    inject[0,-2] = 1
    inject2 = np.zeros((1, i+2))
    inject2[0,0] = 1
    inputs = [{'inject': inject, 'into': pre}, {'inject': inject2, 'into': post}]

    sim.run(T = inject[0].shape[0], inputs = inputs)

    dt2 = np.append(dt2, i)
    dw2 = np.append(dw2, sim.monitors[monitor].state[0,-1] - sim.monitors[monitor].state[0,0])


plt.plot(dt, dw)
plt.plot(dt2, dw2)
plt.xlabel('dt_pre - dt_post')
plt.ylabel('dw')
plt.show()
