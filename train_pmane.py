import numpy as np
import snn
import os
import pmane_helper

pwd = '/project/3018012.23/sim_test/'

EPOCHS = 15

for i in np.arange(0, EPOCHS, 1):
  print('-----------------')
  print('TRAINING EPOCH {:d}.'.format(i))
  
  # what pass are we on?
  with open(os.path.join(pwd, 'models', 'pmane.npy'), 'rb') as f:
      last_pass = np.load(f)


  # load current model
  with snn.Network(neuron_prototype = snn.neurons.LIF,
		  build_from = os.path.join(pwd, 'models', 'pmane_pass{:d}.npy'.format(last_pass))) as network:
      input_layer = network.structure(n = 20)
      excitatory_layer = network.structure(n = 100, lateral_inhibition = 2.4, a = 0.0, b = 5.0, tau_k = 10.0)

      in_ex = network.fibre(pre = input_layer, post = excitatory_layer,
			    type = snn.synapses.Full(n = 20, generator = snn.generators.XavierPositive),
			    plasticity = snn.plasticity.Oja(lr = 1e-2))


  # setup stimuli
  pmane_dir = '/project/3018012.23/stimuli/simplified_spikes/'
  stimuli = pmane_helper.find_files(pmane_dir, '.npy')


  # iterate and simulate
  simulator = snn.Simulator(dt = 1, network = network, solver = snn.solvers.Clean, verbose = False)

  weights_1 = network.synapses[network.synapses_in(0),3]
  for f in range(len(stimuli)):
      progress = f / len(stimuli)
      print('[\t\t\t\t\t]', end='\r')
      print('[' + ''.join(['-' for i in range(int(progress * 40))]), end='\r')
      print('\t\t{:2.2f}%.'.format(np.round(progress*100, 2)), end='\r')

      ins = [{'inject': pmane_helper.load(os.path.join(pmane_dir, stimuli[f])), 'into': 0}]
      simulator.run(T = ins[0]['inject'].shape[1], inputs = ins)
  weights_2 = network.synapses[network.synapses_in(0),3]

  print('mean weight change:')
  print(np.mean(np.abs(weights_1 - weights_2)))

  network.save(os.path.join(pwd, 'models', 'pmane_pass{:d}.npy'.format(last_pass+1)))


  # save current pass
  with open(os.path.join(pwd, 'models', 'pmane.npy'), 'wb') as f:
      last_pass = last_pass + 1
      np.save(f, last_pass)
  
  print('EPOCH {:d} completed.'.format(i))
  print('-----------------')
  print('\n')
  