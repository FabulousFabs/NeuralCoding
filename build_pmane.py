import numpy as np
import snn
import os
import pmane_helper

pwd = '/project/3018012.23/sim_test/'

# build simple model
with snn.Network() as network:
    input_layer = network.structure(n = 20)
    excitatory_layer = network.structure(n = 100, inhib_ff = 1.0, inhib_fb = 1.0, a = 0.0, b = 5.0, tau_k = 10.0)

    in_ex = network.fibre(pre = input_layer, post = excitatory_layer,
                          type = snn.synapses.Full(n = 20, generator = snn.generators.Xavier, directionality = 1),
                          plasticity = snn.plasticity.STDP(lr = 1e-6))
network.save(os.path.join(pwd, 'models', 'pmane_pass1.npy'))


# save current pass
with open(os.path.join(pwd, 'models', 'pmane.npy'), 'wb') as f:
    last_pass = 1
    np.save(f, last_pass)
