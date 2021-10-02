import numpy as np
import os, sys
sys.path.append('./../..')
import snn

# setup paths
pwd = '/project/3018012.23/sim_test/sim_wordlearning/rate/'
sod = os.path.join(pwd, 'stimuli')
mod = os.path.join(pwd, 'models')
bod = os.path.join(pwd, 'behaviour')

# setup population parameters
pop_in_n = 50
pop_ro_n = 60
pop_ro_inhib_ff = 1.0
pop_ro_a = 0.0
pop_ro_b = 0.05
pop_ro_tau_k = 25.0
pop_fb_n = 60

# setup training parameters
EPOCHS = 5

# define custom functions
def find_files(f, t):
    '''
    Grab all files of type t from f
    '''

    af = os.listdir(f)
    at = []
    for f in af:
        if f.endswith(t):
            at.append(f)
    return at

# load stimuli
stimuli = find_files(sod, '.npy')

# load current pass
with open(os.path.join(mod, 'rate_clean_i.npy'), 'rb') as f:
    last_pass = np.load(f)

# load model
with snn.Network(build_from = os.path.join(mod, 'rate_clean_p{:d}.npy'.format(last_pass))) as net:
    pop_in = net.structure(n = pop_in_n)
    pop_ro = net.structure(n = pop_ro_n, inhib_ff = pop_ro_inhib_ff, a = pop_ro_a, b = pop_ro_b, tau_k = pop_ro_tau_k)
    pop_fb = net.structure(n = pop_fb_n)

    fib_excite = net.fibre(pre = pop_in, post = pop_ro, type = snn.synapses.Full(generator = snn.generators.Xavier, n = 50, directionality = 1), plasticity = snn.plasticity.STDP(lr = 1e-6))
    fib_feedforward = net.fibre(pre = pop_ro, post = pop_fb, type = snn.synapses.One_To_One(efficacy = 0.5))
    fib_feedback = net.fibre(pre = pop_fb, post = pop_ro, type = snn.synapses.kWTA(generator = snn.generators.Xavier, n = 1, directionality = -1))

# setup simulation
sim = snn.Simulator(network = net, verbose = False, plasticity = True)
mon = sim.monitor(monitor_type = snn.monitors.Spikes)

# enter training loop
for EPOCH in np.arange(0, EPOCHS, 1):
    for i, stimulus in zip(np.arange(0, len(stimuli), 1), stimuli):
        # logging
        progress = i / len(stimuli)
        print('[\t\t\t\t\t]', end='\r')
        print('[' + ''.join(['-' for n in range(int(progress * 40))]), end='\r')
        print('\t\t{:2.2f}% (E{:d})'.format(np.round(progress*100, 2), last_pass), end='\r')

        # read current episode
        with open(os.path.join(sod, stimulus), 'rb') as f:
            episode = np.load(f)

        # simulate episode
        sim.monitors[mon].reset()
        sim.run(T = episode.shape[1], inputs = [{'inject': episode, 'into': pop_in}], reset = True)

        # save behaviour during episode
        s = stimulus.split('.')[0]
        with open(os.path.join(bod, 'rate_clean_e{:d}_{:s}.npy'.format(last_pass, s)), 'wb') as f:
            np.save(f, sim.monitors[mon].state)

    # save model trained as per current epoch
    last_pass = last_pass + 1
    net.save(to = os.path.join(mod, 'rate_clean_p{:d}.npy'.format(last_pass)))

    # save pass
    with open(os.path.join(mod, 'rate_clean_i.npy'), 'wb') as f:
        np.save(f, last_pass)

print('Completed training. Check model outputs and behaviours.')
