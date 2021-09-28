import numpy as np
import snn
import os
import pmane_helper

pwd = '/project/3018012.23/sim_test/'

EPOCHS = 5

for i in np.arange(0, EPOCHS, 1):
  print('________________________________________')
  print('TRAINING EPOCH {:d}.'.format(i))

  # what pass are we on?
  with open(os.path.join(pwd, 'models', 'fashion_phase.npy'), 'rb') as f:
      last_pass = np.load(f)


  # load current model
  with snn.Network(build_from = os.path.join(pwd, 'models', 'fashion_phase_pass{:d}.npy'.format(last_pass))) as network:
      ins = net.structure(n = 784)
      exc = net.structure(n = 100, a = 0.0, b = 0.25, tau_k = 10.0, inhib_ff = 0.8, inhib_fb = 0.8)

      fib = net.fibre(pre = ins, post = exc, type = snn.synapses.Full(n = 784, generator = snn.generators.Xavier, directionality = 1), plasticity = snn.plasticity.STDP(lr = 1e-6))


  # setup stimuli
  dir = '/project/3018012.23/sim_test/data/fashion/'
  state_dir = '/project/3018012.23/sim_test/models/eval/'
  X_train, Y_train = mnist_reader.load_mnist(dir, kind = 'train')
  L = 200
  phase_lam = 4


  # iterate and simulate
  simulator = snn.Simulator(dt = 1, network = network, solver = snn.solvers.Clean, verbose = False, plasticity = True)
  spikes = simulator.monitor(monitor_type = snn.monitors.Spikes)

  weights_1 = network.synapses[network.synapses_in(0),3]

  for f in range(X_train.shape[0]):
      progress = f / len(stimuli)
      print('[\t\t\t\t\t]', end='\r')
      print('[' + ''.join(['-' for i in range(int(progress * 40))]), end='\r')
      print('\t\t{:2.2f}%'.format(np.round(progress*100, 2)), end='\r')

      # simulate current stimulus
      input = [{'inject': snn.utils.neuralcoding.rate(inputs = X_train[f], L = L, lam = phase_lam), 'into': ins}]
      simulator.run(T = L, inputs = input)

      current_behaviour = simulator.monitors[spikes].state
      with open('fashion_phase_pass{:d}_stim{:d}.npy'.format(last_pass, f), 'wb') as current_file:
          np.save(current_file, current_behaviour)
      simulator.monitors[spikes].reset()

  weights_2 = network.synapses[network.synapses_in(0),3]

  print('\n\n')
  print('mean weight change:')
  print(np.mean(np.abs(weights_1 - weights_2)))

  network.save(os.path.join(pwd, 'models', 'pmane_pass{:d}.npy'.format(last_pass+1)))


  # save current pass
  with open(os.path.join(pwd, 'models', 'pmane.npy'), 'wb') as f:
      last_pass = last_pass + 1
      np.save(f, last_pass)

  print('EPOCH {:d} completed.'.format(i))
  print('________________________________________')
  print('\n')
