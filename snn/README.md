@author: Fabian Schneider <fabian.schneider@donders.ru.nl>, GitHub: FabulousFabs

# Custom spiking neural network package
This is a custom package for different kinds of spiking neural network simulations, written explicitly for minimal dependencies (with the package requiring only standard libraries, `numpy` and `matplotlib`). Find references below.

## Basic Usage
The package was written to be relatively easy to use. The general structure will always follow:
```python
import numpy as np
import snn

'''
setup the network
'''
with snn.Network() as net:
  population_A = net.structure(n = 1)
  population_B = net.structure(n = 1)

  net.fibre(pre = population_A, post = population_B, type = snn.synapses.Full())

'''
setup an arbitrary 1D-signal and rate-encode it as inputs
'''
max_Hz = 50
inputs = np.random.random(size = (1, 1)) * max_Hz
inputs = snn.utils.neuralcoding.rate(inputs = inputs, L = 1e3, lam = 1.0)

'''
setup a simulation and monitors for spikes
'''
sim = snn.Simulator(network = net)
spikes = sim.monitors(monitor_type = snn.monitors.Spikes)

'''
run simulation
'''
sim.run(T = 1e3, inputs = [{'inject': inputs, 'into': population_A}])

'''
show spiking activity
'''
snn.plots.spike_train(monitor = spikes, simulator = sim)
snn.plots.show()
```
This will create a network consisting of two populations (A and B, with one neuron each). A fires at B. Input is rate-coded noise at a maximum of 50Hz. One epoch is simulated and the resulting spiking activity is plotted.

Note that, by default, networks will use `snn.neurons.LIF_NMC` as the neuron type and, correspondingly, simulator will use `dt = 1`, `solver = snn.solvers.Clean`. If you want to use a 'real' LIF neuron (or other type), you can specify the type when creating the neural structure. For example,
```python
  ...
  population_A = net.structure(n = 1, type = snn.neurons.LIF)
  ...
```
will do the job. If you do this, please note that this may change the requirements on the simulator, in turn. Particularly, this will require you to specify a solver and time step size for the simulator, like so:
```python
...
sim = snn.Simulator(network = net, dt = 1e-3, solver = snn.solvers.RungeKutta)
...
```
Also note that, for any neuron type, you can always change default parameters by simply supplying them as kwargs to the `.structure()` call. Say, for example, I want to use a standard neuron (i.e., `snn.neurons.LIF_NMC`), but I would like it to have a firing threshold of two rather than one, this could be done like so:
```python
  ...
  population_A = net.structure(n = 1, V_thr = 2)
  ...
```
which will set `V_thr = 2` for all neurons of population A. For a list of the available universal and class-specific free parameters, please see the documentation below.

Similarly, synapses parameters can be tweaked at creation by altering the call to `snn.synapses.*`. Say, for example, I want population A to fire at population B at only half efficacy and with a delay of an additional time step, this can be setup like so:
```python
  ...
  net.fibre(pre = population_A,
            post = population_B,
            type = snn.synapses.Full(efficacy = 0.5, delay = 2))
  ...
```

Of course, in most ML-oriented scenarios, sharing the same weight between two structures is not going to be ideal at all. Say, for example, we want to build a slightly more complicated network. We have an input population (i.e., `pop_in`), a read out population whose activity we want to later classify (i.e., `pop_ro`) and an inhibitory population that feeds back into our read out population to make spiking patterns more sparse (i.e., `pop_fb`) like so:
```python
  ...
  pop_in = net.structure(n = 100)
  pop_ro = net.structure(n = 25)
  pop_fb = net.structure(n = 25)

  fib_excite = net.fibre(pre = pop_in,
                         post = pop_ro,
                         type = snn.synapses.Full(generator = snn.generators.Xavier,
                                                  n = 100,
                                                  directionality = 1))
  fib_feedforward = net.fibre(pre = pop_ro,
                              post = pop_fb,
                              type = snn.synapses.One_To_One(efficacy = 0.5))

  fib_feedback = net.fibre(pre = pop_fb,
                           post = pop_ro,
                           type = snn.synapses.kWTA(generator = snn.generators.Xavier,
                                                    n = 25,
                                                    directionality = -1))
  ...
```
For `fib_excite` and `fib_feedback` we have now specified generators for weight initialisation. `snn.generatos.Xavier` gets parameters `n` and `directionality` each, with `n` referring to the pre-synpatic population size for normalisation and `directionality` indicating excitatory, inhibitory or mixed weighting (1, -1, 0, respectively).

## Learning
To enable learning, simply specify a plasticity rule when initialising a fibre. This can be achieved by modifying the basic usage example as follows:
```python
...
  net.fibre(pre = population_A,
            post = population_B,
            type = snn.synapses.Full(),
            plasticity = snn.plasticity.STDP(lr = 1e-6))
...
```

## Neuromorphic logic gates
If you want to build specific computing algorithms that require logic gates, you can find implementations of the most common types in `snn.utils.gates.*`. Please note that these gates are guaranteed proper functionality only for `snn.neurons.LIF_NMC` and are untested for other neuron types and simulator settings. For a very naive example, if I want to know the outcome of `AND` for `I0 = 1`, `I1 = 1`, I can compute this like so:
```python
import numpy as np
import snn

'''
setup network
'''
with snn.Network() as net:
  AND = snn.utils.gates.AND(net = net)

'''
setup inputs
'''
inputs = np.zeros((2, 5))
inputs[0,0] = 1
inputs[1,0] = 1

'''
setup simulation
'''
sim = snn.Simulator(network = net)
spikes = sim.monitors(monitor_type = snn.monitors.Spikes)

'''
run simulation
'''
sim.run(T = inputs.shape[1], inputs = [{'inject': inputs[0,:], 'into': AND['I0']},
                                       {'inject': inputs[1,:], 'into': AND['I1']}])

'''
show spiking activity in raster
'''
snn.plots.raster(monitor = spikes, simulator = sim)
snn.plots.show()
```

Now, if I want to connect several gates, this can be achieved by using `O` from the returned structure. For example,
```python
  ...
  AND1 = snn.utils.gates.AND(net = net)
  AND2 = snn.utils.gates.AND(net = net)
  OR = snn.utils.gates.OR(net = net)

  net.fibre(pre = AND1['O'], post = OR['I0'], type = snn.synapses.One_To_One())
  net.fibre(pre = AND2['O'], post = OR['I1'], type = snn.synapses.One_To_One())
  ...
```

Note that some gates will require more than one time step to compute or may require to be explicitly activated (by spiking at `A` within the network). For details on this, please refer to the documentation below. In these cases, however, gates can easily be chained by using appropriate synaptic delays between gates (or a more complex clocking architecture around gates).

## Neural Coding
As seen in 'Basic Usage', several schemes for neural coding are supplied with this package. It should be noted, however, that some schemes may have further requirements to function properly.

For example, `snn.utils.neuralcoding.phase` creates a bit-wise encoding of inputs. To illustrate this, consider that the raw value `255` would be encoded into a spike train of `1 1 1 1 1 1 1 1`. This, in turn, means that this bit-wise structure must necessarily also be present in synaptic weights that need to have phases corresponding to the spike time. This can be achieved by using synaptic filtering, like so:
```python
  ...
  net.fibre(pre = phase_encoded_ins, post = subsequent_population,
            filter = snn.synapses.filters.Phase(phases = 8))
  ...
```
whereby weights are now updated in 8-phasic cycles as per the corresponding function <img src="https://render.githubusercontent.com/render/math?math=w(t%2b 1) = w(t) \times 2^{-[(1 %2b mod(t, phases))]}">. This phasic updating does not, however, interfere with plasticity rules, for example, because it is reset within every time step and applied only for spike propagation.

Similarly, `snn.utils.neuralcoding.ttfs` implements a time-to-first-spike encoding which requires that weights decay exponentially over time, in correspondence with the encoding function. Again, this can be achieved by supplying the appropriate filtering option to the synapse constructor, like so:
```python
  ...
  net.fibre(pre = ttfs_encoded_ins, post = subsequent_population,
            filter = snn.synapses.filters.ExponentialDecay())
  ...
```
whereby weights are now going to decay exponentially within an epoch (but, again, leaving plasticity unaffected and serving only encoding purposes), as per the corresponding function <img src="https://render.githubusercontent.com/render/math?math=w(t%2b 1) = w(t) \times e^{-\frac{t}{\tau_k}}">.


# Documentation
...
