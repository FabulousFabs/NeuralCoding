import numpy as np
import snn
import os
import pmane_helper

pwd = '/project/3018012.23/sim_test/'

# what pass are we on?
with open(os.path.join(pwd, 'models', 'pmane.npy'), 'rb') as f:
    last_pass = np.load(f)


# load current model
with snn.Network(build_from = os.path.join(pwd, 'models', 'pmane_pass{:d}.npy'.format(last_pass))) as network:
    input_layer = network.structure(n = 20)
    excitatory_layer = network.structure(n = 100, inhib_ff = 1.0, inhib_fb = 1.0, a = 0.0, b = 5.0, tau_k = 10.0)

    in_ex = network.fibre(pre = input_layer, post = excitatory_layer,
                          type = snn.synapses.Full(n = 20, generator = snn.generators.Xavier, directionality = 1),
                          plasticity = snn.plasticity.STDP(lr = 1e-6))


# setup stimuli
pmane_dir = '/project/3018012.23/stimuli/simplified_spikes/'
stimuli = pmane_helper.find_files(pmane_dir, '12_8_1.npy')


print(np.mean(network.synapses[:,3]))

# iterate and simulate
simulator = snn.Simulator(dt = 1, network = network, solver = snn.solvers.Clean, verbose = False, plasticity = False)
monitor = simulator.monitor(monitor_type = snn.monitors.Spikes)

ins = [{'inject': pmane_helper.load(os.path.join(pmane_dir, stimuli[0])), 'into': input_layer}]
simulator.run(T = ins[0]['inject'].shape[1], inputs = ins)

snn.plots.spike_train(monitor = monitor, simulator = simulator)
snn.plots.rate_in_time(monitor = monitor, simulator = simulator, L = 10)
snn.plots.show()
