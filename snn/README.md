@author: Fabian Schneider <fabian.schneider@donders.ru.nl>, GitHub: @FabulousFabs

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
spikes = sim.monitor(monitor_type = snn.monitors.Spikes)

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
  population_A = net.structure(n = 1, t = snn.neurons.LIF)
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
For `fib_excite` and `fib_feedback` we have now specified generators for weight initialisation. `snn.generatos.Xavier` gets parameters `n` and `directionality` each, with `n` referring to the pre-synpatic population size for normalisation and `directionality` indicating excitatory, inhibitory or mixed weighting (1, -1, None, respectively).

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
spikes = sim.monitor(monitor_type = snn.monitors.Spikes)

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
## 1 - `snn.Network`: Class
### 1.1 - `snn.Network.__init__`: Method
- Returns a network object.
- __INPUTS__
  - `build_from = None`:
    - If network should be built from previously saved state, specify file name here.

### 1.2 - `snn.Network.neurons_in`: Method
- Returns neuron indices of a structure.
- __INPUTS__
  - `structure`:
    - Structure identifier. To obtain neuron indices of multiple structures, pass structure identifiers as `np.ndarray`.
- __OUTPUTS__
  - `idx`:
    - Indices of neurons.

### 1.3 - `snn.Network.synapses_in`: Method
- Returns synapse indices of a fibre.
- __INPUTS__
  - `fibre`:
    - Fibre identifier. To obtain synapse indices of multiple fibres, pass fibre identifiers as `np.ndarray`.
- __OUTPUTS__
  - `idx`:
    - Indices of synapses.

### 1.4 - `snn.Network.structure`: Method
- Creates a new structure of neurons.
- __INPUTS__
  - `n = 1`:
    - Number of neurons to create
  - `t = LIF_NMC`:
    - Neuron type to use for this structure
  - `inhib_ff = 0.0`:
    - Degree of lateral feedforward inhibition within structure.
  - `inhib_fb = 0.0`:
    - Degree of lateral feedback inhibitin within structure.
  - `**kwargs`:
    - Additional named arguments to pass to the constructor of the neuron type that will set custom parameters.
- __OUTPUTS__
  - `struct`:
    - Structure identifier

### 1.5 - `snn.Network.fibre`: Method
- Creates a new fibre between structures.
- __INPUTS__
  - `pre`:
    - Presynaptic structure identifier
  - `post`:
    - Postsynaptic structure identifier
  - `type`:
    - Fibre type (see `snn.synapses.*`)
  - `plasticity = None`:
    - Plasticity rule to apply to fibre (see `snn.plasticity.*`)
  - `filter = None`:
    - Filter to apply to fibre (see `snn.synapses.filters.*`)
- __OUTPUTS__
  - `fibre`:
    - Fibre identifier

### 1.6 - `snn.Network.reset`: Method
- Resets all non-constant properties of a neuron (before running a new simulation).

### 1.7 - `snn.Network.save`: Method
- Save the network to disc.
- __INPUTS__
  - `to`:
    - File to save network to

### 1.8 - `snn.Network.load`: Method
- Load the network from disc.
- __NOTE__ This function should not technically be used. Please use the `build_from` option in `snn.Network.__init__` instead.
- __INPUTS__
  - `model`:
    - File to load the model from.

## 2 - `snn.Simulator`: Class
### 2.1 - `snn.Simulator.__init__`:
- Returns a simulator object.
- __INPUTS__
  - `network`:
    - Network object to simulate
  - `dt = 1`:
    - Size of time steps
  - `solver = snn.solvers.Clean`:
    - Solver to use for diffential equations
  - `verbose = False`:
    - Logging state
  - `plasticity = True`:
    - Allow plasticity in this simulator?

### 2.2 - `snn.Simulator.monitor`: Method
- Add a monitor to the simulator.
- __INPUTS__
  - `monitor_type`:
    - Type of monitor (see `snn.monitors.*`)
  - `is_synapse = False`:
    - Are we targetting synapses?
  - `targets = None`:
    - Structure or fibre targets of this monitor
    - If `None`, all will be included
  - `of = None`:
    - If `monitor_type = snn.monitors.Continuous`, what measurement should be taken (see `snn.neurons.labels.*`)

### 2.3 - `snn.Simulator.run`: Method
- Run a simulation
- __INPUTS__
  - `T = 1`:
    - Duration of the simulation
  - `inputs = None`:
    - Inputs to use for the simultion
    - Must be `array` of `dictionary`s, for example `[{'inject': signal1, 'into': structure1}, ...]`
  - `reset = True`:
    - Before running the simulation, should neurons be reset to ground state?

## 3 - `snn.plots`: Utility methods
### 3.1 - `snn.plots.raster`: Method
- Create a raster plot of spiking activity
- __INPUTS__
  - `monitor`:
    - Monitor identifier
  - `simulator`:
    - Simulator object
  - `title = None`:
    - Title of plot

### 3.2 - `snn.plots.spike_train`: Method
- Create a spike train plot of activity
- __INPUTS__
  - `monitor`:
    - Monitor identifier
  - `simulator`:
    - Simulator object
  - `title = None`:
    - Title of plot

### 3.3 - `snn.plots.continuous`: Method
- Create a spaced plot of continuous measurements across units
- __INPUTS__
  - `monitor`:
    - Monitor identifier
  - `simulator`:
    - Simulator object
  - `title = None`:
    - Title of plot

### 3.4 - `snn.plots.rate_in_time`: Method
- Creates a grid plot of firing rates of neurons across time
- __INPUTS__
  - `monitor`:
    - Monitor identifier
  - `simulator`:
    - Simulator object
  - `title = None`:
    - Title of plot
  - `grid = False`:
    - Show grid lines?
  - `rf = snn.utils.ratefunctions.linear_filter_and_kernel`:
    - Rate function to use for creating the time-frequency representation (see `snn.utils.ratefunctions.*`)
  - `L = 1`:
    - Window length for kernel function of RF

### 3.5 - `snn.plots.network`: Method
- Creates a neat plot of the network structure (using a mini force-directed physics simulation for the layout of each structure)
- This function can also be used to show activity of the network across a simulated episode if the simulator and corresponding monitor are passed
- __NOTE__ that, if activity animation is desired, this is very slow and, hence, not available as a live view, but saved to disc instead (set `cmap_anim_save_to`)
- __INPUTS__
  - `net`:
    - Network object
  - `struct_spacing = 0.2`:
    - Spacing between structures
  - `fibre_spacing = 0.012`:
    - Spacing between fibres
  - `cmap_neurons = 'inferno'`:
    - Name of the colourmap for neurons
  - `cmap_neurons_by_struct = True`:
    - Colour neurons by structure (or randomly)?
  - `cmap_synapses = 'binary'`:
    - Name of the colourmap for synapses
  - `cmap_synapses_by_fibre = True`:
    - Colour synapses by fibre (or randomly)?
  - `synapses_alpha = 0.1`:
    - Alpha level to display synapses at
  - `labels_struct = None`:
    - Labels of the structures
    - If `None`, labels will be auto-generated.
  - `labels_fibre = None`:
    - Labels of the fibres
    - If `None`, labels will be auto-generated.
  - `show_labels_struct = True`:
    - Show structure labels?
  - `show_labels_fibre = False`:
    - Show fibre labels?
  - `cmap_neurons_by_state = False`:
    - Colour neurons by state
    - __NOTE__ that this enables the animation and requires that you provide `cmap_neurons_sim`, `cmap_neurons_monitor`.
  - `cmap_neurons_sim = None`:
    - Simulator object (for animation)
  - `cmap_neurons_monitor = None`:
    - Monitor identifier (for animation)
  - `cmap_neurons_use_kernel = True`:
    - Use a kernel to smooth out events in monitor?
  - `cmap_neurons_kernel_L = 10`:
    - Length of the smoothing kernel.
  - `cmap_anim_interval = 50`:
    - Interval between frames of animation
  - `cmap_anim_save_to = 'network_animated.gif'`:
    - File to export animation to

### 3.6 - `snn.plots.show`: Method
- Show plots (short-hand to avoid imports in main)

## 4 - `snn.solvers`: Utility methods
### 4.1 - `snn.solvers.Heuns`: Method
- Heun's method for solving differential equations
- __INPUTS__
  - `x0`:
    - State of x at t
  - `t0`:
    - Time t
  - `dt`:
    - Time step size
  - `dxdt`:
    - Function to solve
  - `**kwargs`:
    - Additional named parameters to pass to `dxdt`

### 4.2 - `snn.solvers.Eulers`: Method
- Euler's method for solving differential equations
- __INPUTS__
  - `x0`:
    - State of x at t
  - `t0`:
    - Time t
  - `dt`:
    - Time step size
  - `dxdt`:
    - Function to solve
  - `h = None`:
    - Effective dt for iterations
    - If `None`, then `h = dt / n`
  - `n = 10`:
    - Number of iterations to compute
  - `**kwargs`:
    - Additional named parameters to pass to `dxdt`

### 4.3 - `snn.solvers.RungeKutta`: Method
- Runge-Kutta's method for solving differential equations
- __INPUTS__
  - `x0`:
    - State of x at t
  - `t0`:
    - Time t
  - `dt`:
    - Time step size
  - `dxdt`:
    - Function to solve
  - `h = None`:
    - Effective dt for iterations
    - If `None`, then `h = dt / n`
  - `n = 10`:
    - Number of iterations to compute
  - `**kwargs`:
    - Additional named parameters to pass to `dxdt`

### 4.4 - `snn.solvers.Clean`: Method
- Does not solve but uses a simple step (calls `dxdt` directly)
- __INPUTS__
  - `x0`:
    - State of x at t
  - `t0`:
    - Time t
  - `dt`:
    - Time step size
  - `dxdt`:
    - Function to solve
  - `**kwargs`:
    - Additional named parameters to pass to `dxdt`

## 5 - `snn.monitors`: Collection of classes
- Please see documentation of `snn.Simulator.monitor`

### 5.1 - `snn.monitors.Spikes`: Class
- Use to capture spikes
- Not applicable to fibres

### 5.2 - `snn.monitors.States`: Class
- Use to capture states
- Applicable to structures of fibres

## 6 - `snn.generators`: Utility methods
### 6.1 - `snn.generators.Xavier`: Method
- Get Xavier initialisation
- __INPUTS__
  - `dim = 1`:
    - Shape of dimension zero
  - `n = 1`:
    - Number of neurons in previous layer
  - `directionality = None`:
    - Tail of distribution (1, -1, or None for double)
- __OUTPUTS__
  - `y`:
    - Xavier samples

### 6.2 - `snn.generators.Gaussian`: Method
- Get Gaussian initialisation
- __INPUTS__
  - `dim = 1`:
    - Shape of dimension zero
  - `efficacy = 1`:
    - Mu of Gaussian
  - `sd = 0`:
    - SD of Gaussian
- __OUTPUTS__
  - `y`:
    - Gaussian samples

### 6.3 - `snn.generators.Uniform`: Method
- Get uniform initialisation
- __INPUTS__
  - `dim = 1`:
    - Shape of dimension zero
  - `efficacy = 1`:
    - Lower and upper bound of distribution
  - `directionality`:
    - Tail of distribution (1, -1, or None for double)
- __OUTPUTS__
  - `y`:
    - Uniform samples

### 6.4 - `snn.generators.Poisson`: Method
- Get Poisson distribution of size = dim with rate of events = r
- __INPUTS__
  - `dim = (1, 1)`:
    - Shape of outputs
  - `r = 1`:
    - Rate of events (if homogenous)
  - `homogenous = True`:
    - Sample homogenous Poisson?
    - If `False`, `rf` is required
  - `rf = None`:
    - Rate function for events for inhomogenous process
- __OUTPUTS__
  - `y`:
    - Poisson samples

## 7 - `snn.utils.ratefunctions`: Utility methods
### 7.1 - `snn.utils.ratefunctions.spike_count`: Method
- Returns spike count over time as per <img src="https://render.githubusercontent.com/render/math?math=r(N,%20T)%20=%20N%20/%20T">
- __INPUTS__
  - `states`:
    - Spiking object from monitor
- __OUTPUT__
  - `r`:
    - Rate value

### 7.2 - `snn.utils.ratefunctions.linear_filter_and_kernel`: Method
- Returns TFR from convolution with linear filter
- __INPUTS__
  - `states`:
    - Spiking object from monitor
  - `dt = 1e-3`:
    - Time step size
  - `delta_t = 10`:
    - Time steps per window
- __OUTPUTS__
  - `tfr`:
    - Time-frequency representation

### 7.3 - `snn.utils.ratefunction.gaussian_and_kernel`: Method
- Returns TFR from convolution with Gaussian
- __INPUTS__
  - `states`:
    - Spiking object from monitor
  - `dt = 1e-3`:
    - Time step size
  - `delta_t = 10`:
    - Time steps per window
- __OUTPUTS__
  - `tfr`:
    - Time-frequency representation

## 8 - `snn.utils.neuralcoding`: Utility methods
### 8.1 - `snn.utils.neuralcoding.rate`: Method
- Returns a rate encoding of the stimulus using a Poisson spike train as per
  - Guo, W., Fouda, M.E., Eltawil, A.M., & Salama, K.N. (2021). Neural coding in spiking neural networks: A comparative study for robust neuromorphic systems. Frontiers in Neuroscience, 15, e638474. DOI: http://dx.doi.org/10.3389/fnins.2021.638474
- __INPUTS__
  - `inputs`:
    - Input vector (neuron x input)
  - `L`:
    - Length of output vector
  - `lam`:
    - Lambda for transforming input values (practical inverse scaling of Hz)
  - `homogenous = True`:
    - If `False`, indicates `inputs` is time-varying signal and uses inhomogenous Poisson distribution
- __OUTPUTS__
  - `y`:
    - Rate-encoded spike train

### 8.2 - `snn.utils.neuralcoding.ttfs`: Method
- Returns a time-to-first-spike encoding of the stimulus as per
  - Park, S., Kim, S.J., Na, B., & Yoon, S. (2020). T2FSNN: Deep spiking neural networks with time-to-first-spike coding. Proceedings of the 2020 57th ACM/IEEE Design Automation Conference. DOI: http://dx.doi.org/10.1109/DAC18072.2020.9218689
- __NOTE__ that this currently only supports 1d inputs
- __NOTE__ that this form of encoding relies on a weight decay function that effectively mirrors P_th computed (see `snn.synapses.filters.ExponentialDecay`)
- __INPUTS__
  - `inputs`:
    - Input vector (neuron x input)
  - `max`:
    - Maximum value of input
  - `T = 1`:
    - Desired total time (relative to dt)
  - `dt = 1e-3`:
    - Time step size
  - `theta_0 = 1.0`:
    - Event threshold
  - `tau_theta = 10.0`:
    - Time constant for exponential decay
- __OUTPUTS__
  - `y`
    - TTFS-encoded spike train

### 8.3 - `snn.utils.neuralcoding.phase`: Method
- Returns a phase encoding of the stimulus as per
  - Kim, J., Kim, H., Huh, S., Lee, J., & Choi, K. (2018). Deep neural networks with weighted spikes. Neurocomputing, 311, 373-386. DOI: http://dx.doi.org/10.1016/j.neucom.2018.05.087
- __NOTE__ that this form of encoding requires a weight updating filter by current phase of encoding (see `snn.synapses.filters.Phase`)
- __INPUTS__
  - `inputs`:
    - Input vector (neuron x input)
  - `L`:
    - Length of output vector
  - `bits = 8`:
    - Number of bits to encode but ensure that `2**bits >= max(inputs)` and `min(inputs) >= 0`
- __OUTPUTS__
  - `y`:
    - Phase-encoded spike train

### 8.4 - `snn.utils.neuralcoding.burst`: Method
- Returns a burst encoding of the stimulus as per
  - Guo, W., Fouda, M.E., Eltawil, A.M., & Salama, K.N. (2021). Neural coding in spiking neural networks: A comparative study for robust neuromorphic systems. Frontiers in Neuroscience, 15, e638474. DOI: http://dx.doi.org/10.3389/fnins.2021.638474
- __NOTE__ that this does not yet support inputs that vary in time.
- __INPUTS__
  - `inputs`:
    - Input vector (neurons x input)
  - `L`:
    - Desired output length
  - `max`:
    - Maximum value of input
  - `N_max = 5.0`:
    - Maximum number of spikes in burst
  - `T_min = 2.0`:
    - Minimum ISI
  - `T_max = None`:
    - Maximum ISI
    - If `None`, `T_max = L`
- __OUTPUTS__
  - `y`:
    - Burst-encoded spike train

## 9 - `snn.utils.gates`: Utility methods
### 9.1 - `snn.utils.gates.CLOCK`: Method
- Implements a clock in the supplied network.
- __INPUTS__
  - `net`:
    - Network object
  - `tau = 1`:
    - Time constant (i.e., one beat every tau steps)
  - `start_ticking = True`:
    - Start clock now?
- __OUTPUTS__
  - `CLOCK`:
    - Network identifier for CLOCK

### 9.2 - `snn.utils.gates.AND`: Method
- Implements an AND-gate in the supplied network
- __DT__ 1
- __TRUTHTABLE__
```
| I0 | I1 | O  |
| -- | -- | -- |
| 0  | 0  | 0  |
| 0  | 1  | 0  |
| 1  | 0  | 0  |
| 1  | 1  | 1  |
```
- __INPUTS__
  - `net`:
    - Network object
- __OUTPUTS__
  - `AND`:
    - Network identifiers for `I0`, `I1`, `O` in a dictionary

### 9.3 - `snn.utils.gates.OR`: Method
- Implements an OR-gate in the supplied network
- __DT__ 1
- __TRUTHTABLE__
```
| I0 | I1 | O  |
| -- | -- | -- |
| 0  | 0  | 0  |
| 0  | 1  | 1  |
| 1  | 0  | 1  |
| 1  | 1  | 1  |
```
- __INPUTS__
  - `net`:
    - Network object
- __OUTPUTS__
  - `OR`:
    - Network identifiers for `I0`, `I1`, `O` in a dictionary

### 9.4 - `snn.utils.gates.NOR`: Method
- Implements a NOR-gate in the supplied network
- __NOTE__ that this requires `A` to be activated by a spike in the same time step as `I0` / `I1`
- __DT__ 1
- __TRUTHTABLE__
```
| I0 | I1 | O  |
| -- | -- | -- |
| 0  | 0  | 1  |
| 0  | 1  | 0  |
| 1  | 0  | 0  |
| 1  | 1  | 0  |
```
- __INPUTS__
  - `net`:
    - Network object
- __OUTPUTS__
  - `NOR`:
    - Network identifiers for `A` (activate), `I0`, `I1`, `O`, `D` (done) in a dictionary

### 9.5 - `snn.utils.gate.NAND`: Method
- Implements a NAND-gate in the supplied network
- __NOTE__ that this requires `A` to be activated by a spike in the same time step as `I0` / `I1`
- __DT__ 1
- __TRUTHTABLE__
```
| I0 | I1 | O  |
| -- | -- | -- |
| 0  | 0  | 1  |
| 0  | 1  | 1  |
| 1  | 0  | 1  |
| 1  | 1  | 0  |
```
- __INPUTS__
  - `net`:
    - Network object
- __OUTPUTS__
  - `NAND`:
    - Network identifiers for `A` (activate), `I0`, `I1`, `O`, `D` (done) in a dictionary

### 9.6 - `snn.utils.gate.XOR`: Method
- Implements an XOR-gate in the supplied network.
- __NOTE__ that this requires `A` to be activated by a spike in the same time step as `I0` / `I1`
- __DT__ 4
- __TRUTHTABLE__
```
| I0 | I1 | O  |
| -- | -- | -- |
| 0  | 0  | 0  |
| 0  | 1  | 1  |
| 1  | 0  | 1  |
| 1  | 1  | 0  |
```
- __INPUTS__
  - `net`:
    - Network object
- __OUTPUTS__
  - `XOR`:
    - Network identifiers for `A` (activate), `I0`, `I1`, `O`, `D` (done) in a dictionary

### 9.7 - `snn.utils.gate.XNOR`: Method
- Implements an XNOR-gate in the supplied network.
- __NOTE__ that this requires `A` to be activated by a spike in the same time step as `I0` / `I1`
- __DT__ 4
- __TRUTHTABLE__
```
| I0 | I1 | O  |
| -- | -- | -- |
| 0  | 0  | 1  |
| 0  | 1  | 0  |
| 1  | 0  | 0  |
| 1  | 1  | 1  |
```
- __INPUTS__
  - `net`:
    - Network object
- __OUTPUTS__
  - `XNOR`:
    - Network identifiers for `A` (activate), `I0`, `I1`, `O`, `D` (done) in a dictionary

### 9.8 - `snn.utils.gate.MUX`: Method
- Implements a MUX-gate in the supplied network
- __NOTE__ that this requires `A` to be activated by a spike in the same time step as `I0` / `I1`
- __DT__ 4
- __TRUTHTABLE__
```
| S0 | I0 | I1 | O  |
| -- | -- | -- | -- |
| 0  | 0  | 0  | 0  |
| 0  | 1  | 0  | 1  |
| 0  | 0  | 1  | 0  |
| 0  | 1  | 1  | 1  |
| 1  | 0  | 0  | 0  |
| 1  | 1  | 0  | 0  |
| 1  | 0  | 1  | 1  |
| 1  | 1  | 1  | 1  |
```
- __INPUTS__
  - `net`:
    - Network object
- __OUTPUTS__
  - `MUX`:
    - Network identifiers for `A` (activate), `I0`, `I1`, `S`, `O`, `D` (done) in a dictionary
