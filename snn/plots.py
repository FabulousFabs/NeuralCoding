import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

from . import utils

def raster(monitor = None, simulator = None, title = None):
    '''
    Create a raster plot of spiking activity

    INPUTS:
        monitor     -   Monitor that provides data
        simulator   -   Simulator object for the monitor
        title       -   Title of this plot (optional)
    '''

    if monitor is None: return False
    if title is None: title = 'Raster plot of activity measured in monitor {:d}.'.format(monitor)

    dt = simulator.dt
    states = simulator.monitors[monitor].state.T

    plt.figure()
    plt.matshow(states, cmap = 'gray', fignum = 1)

    axes = plt.gca()
    axes.set_title(title)
    axes.set_ylabel('Neurons')
    axes.set_xlabel('Time steps')
    axes.set_xticks(np.arange(0, states.shape[1], 1))
    axes.set_yticks(np.arange(0, states.shape[0], 1))
    axes.set_xticklabels(np.arange(0, states.shape[1], 1))
    axes.set_yticklabels(np.arange(0, states.shape[0], 1))
    axes.set_xticks(np.arange(-.5, states.shape[1], 1), minor = True)
    axes.set_yticks(np.arange(-.5, states.shape[0], 1), minor = True)
    axes.grid(which = 'minor', color = 'gray', linestyle = '-', linewidth = 1)

def spike_train(monitor = None, simulator = None, title = None, marker = '|', linewidth = 0.5):
    '''
    Create a spike train plot of activity

    INPUTS:
        monitor     -   Monitor that provides data
        simulator   -   Simulator object for the monitor
        title       -   Title of this plot (optional)
    '''

    if monitor is None or simulator is None: return False
    if title is None: title = 'Spike train measured in monitor {:d}.'.format(monitor)

    plt.figure()
    plt.title(title)
    plt.ylabel('Neurons')
    plt.xlabel('Time')

    dt = simulator.dt
    states = simulator.monitors[monitor].state.T
    states[states[:,:] == 0] = np.nan

    for i in range(states.shape[0]):
        plt.scatter(np.arange(states.shape[1]) * dt, states[i,:] * i, marker=marker, linewidth=linewidth)

    axes = plt.gca()
    axes.set_xlim([0-dt, states.shape[1] * dt])
    axes.set_ylim([-0.1, states.shape[0]+0.1])
    axes.set_yticks(np.arange(0, states.shape[0]))

def continuous(monitor = None, simulator = None, title = None):
    '''
    Create a spaced plot of continuous measurements across units

    INPUTS:
        monitor     -   Monitor that provides data
        simulator   -   Simulator object for the monitor
        title       -   Title of this plot (optional)
    '''

    if monitor is None or simulator is None: return False
    if title is None: title = 'Continuous measurement unit {:d} in monitor {:d}.'.format(simulator.monitors[monitor].of, monitor)

    plt.figure()
    plt.title(title)
    plt.ylabel('Unit {:d}'.format(simulator.monitors[monitor].of))
    plt.xlabel('Time')

    dt = simulator.dt
    states = simulator.monitors[monitor].state
    spacing = 1.5 * np.max(np.abs(states))
    if spacing.astype(np.int) == 0: spacing = 1

    for i in range(states.shape[0]):
        plt.plot(np.arange(states.shape[1]) * dt, states[i,:] + (i * spacing))

    axes = plt.gca()
    axes.set_xlim([0-dt, states.shape[1] * dt])
    axes.set_ylim(-np.max(np.abs(states)), (states.shape[0] * spacing))
    axes.set_yticks(np.arange(0, states.shape[0] * spacing, spacing)[0:states.shape[0]])
    axes.set_yticklabels(np.arange(0, states.shape[0]))

def rate_in_time(monitor = None, simulator = None, L = 1, grid = False, title = None, rf = utils.ratefunctions.linear_filter_and_kernel):
    '''
    Show grid plot of firing rates of neurons across time.

    INPUTS:
        monitor     -   Monitor that provides data
        simulator   -   Simulator object for the monitor
        L           -   Length of window over which rates are calculated
        grid        -   Show grid lines? (True/False)
        title       -   Title of this plot (optional)
        rf          -   Rate function to use (optional)
    '''

    if title is None: title = 'Firing rate measured in monitor {:d}.'.format(monitor)

    dt = simulator.dt
    states = simulator.monitors[monitor].state.T
    tfr = rf(states, dt, delta_t = L)

    plt.figure()
    plt.title(title)
    plt.ylabel('Neurons')
    plt.xlabel('Time')
    plt.imshow(tfr, cmap = 'hot', interpolation = 'nearest')
    plt.colorbar()

    axes = plt.gca()
    axes.set_xticks(np.arange(0, tfr.shape[1], 1))
    axes.set_yticks(np.arange(0, tfr.shape[0], 1))
    axes.set_xticklabels(np.arange(0, tfr.shape[1], 1))
    axes.set_yticklabels(np.arange(0, tfr.shape[0], 1))
    axes.set_xticks(np.arange(-.5, tfr.shape[1], 1), minor = True)
    axes.set_yticks(np.arange(-.5, tfr.shape[0], 1), minor = True)
    if grid is True:
        axes.grid(which = 'minor', color = 'gray', linestyle = '-', linewidth = 1)

def show():
    '''
    Show plots (short-hand to avoid importing in main file)
    '''

    plt.show()
