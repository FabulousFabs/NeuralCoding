import numpy as np
import mnist_reader
import snn

pwd = '/project/3018012.23/sim_test/'

# setup network
with snn.Network() as net:
    ins = net.structure(n = 784)
    exc = net.structure(n = 100, a = 0.0, b = 0.25, tau_k = 10.0, inhib_ff = 0.8, inhib_fb = 0.8)

    fib = net.fibre(pre = ins, post = exc, type = snn.synapses.Full(n = 784, generator = snn.generators.Xavier, directionality = 1), plasticity = snn.plasticity.STDP(lr = 1e-6))
network.save(os.path.join(pwd, 'models', 'fashion_phase_pass1.npy'))


# save current pass
with open(os.path.join(pwd, 'models', 'fashion_phase.npy'), 'wb') as f:
    last_pass = 1
    np.save(f, last_pass)
