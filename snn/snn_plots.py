import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

def raster(monitor = None, simulator = None):
    '''
    Create a raster plot of spiking activity
    '''

    if monitor is None: return False

    dt = simulator.dt
    states = simulator.monitors[monitor].state.T

    plt.figure()
    plt.matshow(states, cmap = 'gray', fignum = 1)

    axes = plt.gca()
    axes.set_title('Raster plot of activity measured in monitor {:d}.'.format(monitor))
    axes.set_ylabel('Neurons')
    axes.set_xlabel('Time steps')
    axes.set_xticks(np.arange(0, states.shape[1], 1))
    axes.set_yticks(np.arange(0, states.shape[0], 1))
    axes.set_xticklabels(np.arange(0, states.shape[1], 1))
    axes.set_yticklabels(np.arange(0, states.shape[0], 1))
    axes.set_xticks(np.arange(-.5, states.shape[1], 1), minor = True)
    axes.set_yticks(np.arange(-.5, states.shape[0], 1), minor = True)
    axes.grid(which = 'minor', color = 'gray', linestyle = '-', linewidth = 1)

def spike_train(monitor = None, simulator = None):
    '''
    Create a spike train plot of activity
    '''

    if monitor is None or simulator is None: return False

    plt.figure()
    plt.title('Spike train measured in monitor {:d}.'.format(monitor))
    plt.ylabel('Neurons')
    plt.xlabel('Time')

    dt = simulator.dt
    states = simulator.monitors[monitor].state.T
    states[states[:,:] == 0] = np.nan

    for i in range(states.shape[0]):
        plt.scatter(np.arange(states.shape[1]) * dt, states[i,:] * i, marker='|', linewidths=0.5)

    axes = plt.gca()
    axes.set_xlim([0-dt, states.shape[1] * dt])
    axes.set_ylim([-0.1, states.shape[0]+0.1])
    axes.set_yticks(np.arange(0, states.shape[0]))

def continuous(monitor = None, simulator = None):
    '''
    Create a spaced plot of continuous measurements across units
    '''

    if monitor is None or simulator is None: return False

    plt.figure()
    plt.title('Continuous measurement unit {:d} in monitor {:d}.'.format(simulator.monitors[monitor].of, monitor))
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
    axes.set_yticks(np.arange(0, states.shape[0] * spacing, spacing))
    axes.set_yticklabels(np.arange(0, states.shape[0]))

def rate_in_time(monitor = None, simulator = None, L = 1, grid = False):
    dt = simulator.dt
    states = simulator.monitors[monitor].state.T
    states = np.pad(states, ((0,0), (0, L-(states.shape[1] % L))), 'constant', constant_values = 0)
    states[np.isnan(states)] = 0.0
    tfr = np.zeros((states.shape[0], np.int(states.shape[1] / L)))

    plt.figure()
    plt.title('Firing rate measured in monitor {:d}.'.format(monitor))
    plt.ylabel('Neurons')
    plt.xlabel('Time')

    for i in np.arange(0, tfr.shape[1], 1):
        tfr[:,i] = np.mean(states[:,i*L:(i*L+L)], axis = 1) / dt

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
