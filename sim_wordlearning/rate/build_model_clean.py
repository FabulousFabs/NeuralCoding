import numpy as np
import os, sys
sys.path.append('./../..')
import snn

# setup paths
pwd = '/project/3018012.23/sim_test/sim_wordlearning/rate/'
mod = os.path.join(pwd, 'models')

# setup population parameters
pop_in_n = 50
pop_ro_n = 60
pop_ro_inhib_ff = 1.0
pop_ro_a = 0.0
pop_ro_b = 0.05
pop_ro_tau_k = 25.0
pop_fb_n = 60

# setup model
with snn.Network() as net:
    pop_in = net.structure(n = pop_in_n)
    pop_ro = net.structure(n = pop_ro_n, inhib_ff = pop_ro_inhib_ff, a = pop_ro_a, b = pop_ro_b, tau_k = pop_ro_tau_k)
    pop_fb = net.structure(n = pop_fb_n)

    fib_excite = net.fibre(pre = pop_in, post = pop_ro, type = snn.synapses.Full(generator = snn.generators.Xavier, n = 50, directionality = 1), plasticity = snn.plasticity.STDP(lr = 1e-6))
    fib_feedforward = net.fibre(pre = pop_ro, post = pop_fb, type = snn.synapses.One_To_One(efficacy = 0.5))
    fib_feedback = net.fibre(pre = pop_fb, post = pop_ro, type = snn.synapses.kWTA(generator = snn.generators.Xavier, n = 1, directionality = -1))
net.save(to = os.path.join(mod, 'rate_clean_p0.npy'))

# save pass
last_pass = 0

with open(os.path.join(mod, 'rate_clean_i.npy'), 'wb') as f:
    np.save(f, last_pass)
