import numpy as np
import numpy.matlib
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

from . import utils
from . import neurons_labels as nl

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
    if grid is True: axes.grid(which = 'minor', color = 'gray', linestyle = '-', linewidth = 1)

def network(net = None, struct_spacing = 0.2, fibre_spacing = 0.012, cmap_neurons = 'inferno', cmap_neurons_by_struct = False,
            cmap_neurons_by_state = False, cmap_neurons_use_kernel = True, cmap_neurons_kernel_L = 10, cmap_neurons_monitor = None,
            cmap_neurons_sim = None, cmap_anim_interval = 50, cmap_anim_save_to = 'network_animated.gif',
            cmap_synapses = 'binary', cmap_synapses_by_fibre = True, synapse_alpha = 0.1, labels_struct = None,
            labels_fibre = None, show_labels_struct = True, show_labels_fibre = False):
    '''
    Create a neat plot of the network structure (using a mini force-directed physics
    simulation for the layout of each structure). This function can also be used to
    show activity of the network across a simulated episode if the simulator and the
    corresponding monitor are passed.

    NOTE: Animation is very slow and, hence, not available as a live view. Animations
    are automatically saved to disc as such (set cmap_anim_save_to).

    INPUTS:
        net                         -   Network object
        struct_spacing              -   Spacing between structures (default = 0.2)
        fibre_spacing               -   Spacing between fibres (default = 0.012)
        cmap_neurons                -   Name of the colourmap for neurons
        cmap_neurons_by_struct      -   Colour neurons by structure? (True/False)
        cmap_neurons_by_state       -   Colour neurons by state (True/False)
                                        NOTE: This enables the animation and requires
                                        that you provide cmap_neurons_sim as well as
                                        cmap_neurons_monitor
        cmap_neurons_use_kernel     -   Use a kernel for animation plots that spaces
                                        states in a gaussian mix (True/False). Note
                                        that this is useful only for spiking behaviour.
        cmap_neurons_kernel_L       -   Length of the kernel (how much smoothing to
                                        apply in T)
        cmap_neurons_monitor        -   Monitor identifier (for animation)
        cmap_neurons_sim            -   Simulator object (for animation)
        cmap_anim_interval          -   Time between frames (for animation)
        cmap_anim_save_to           -   File name (for animation)
        cmap_synapses               -   Name of colourmap for synapses
        cmap_synapses_by_fibre      -   Colour synapses by fibre? (True/False)
        synapses_alpha              -   Alpha level of plotted synapses (default = 0.1)
        labels_struct               -   Labels of the structures.
        labels_fibre                -   Labels of the fibres.
        show_labels_struct          -   Show structure labels? (True/False)
        show_labels_fibre           -   Show fibre labels? (True/False)
    '''

    if net is None or (cmap_neurons_by_state is True and (cmap_neurons_monitor is None or cmap_neurons_sim is None)): return False

    # setup custom helper functions repel() and force_position()
    def repel(el = None, r = 1.0, tau = 1.0, dm = 1e-3):
        '''
        Computes one iteration of force-direction among dots on a 2D-surface.

        INPUTS:
            el      -   Elements as Elements x Pos (X, Y)
            r       -   Repulsion constant
            tau     -   Repulsion decay constant
            dm      -   Step size

        OUTPUTS:
            new_el  -   Updated elements matrix
        '''

        if el is None: return False

        new_el = np.zeros((el.shape[0], el.shape[1]))

        for i in np.arange(el.shape[0]):
            e = el[i,:]
            xy = np.copy(el)
            xy = np.delete(xy, i, 0)

            delta_x = xy[:,0] - e[0]
            delta_y = xy[:,1] - e[1]
            theta = np.arctan2(delta_y, delta_x)

            rxy = np.array([delta_x, delta_y]).T
            rd = r * np.exp(-1/tau_r * np.sqrt(rxy[:,0] ** 2 + rxy[:,1] ** 2))
            rtheta = theta - np.pi

            rme = np.sum(np.array([e[0] + np.cos(rtheta) * rd, e[1] + np.sin(rtheta) * rd]) * dm, axis = 1)
            new_el[i,:] = rme

        return new_el

    def force_position(el = None, r = 1.0, tau = 1.0, dm = 1e-3, it = 1e3):
        '''
        Computes a force-directed positioning of the elements.

        INPUTS:
            el      -   Elements as Elements x Pos (X, Y)
            r       -   Repulsion constant
            tau     -   Repulsion decay constant
            dm      -   Step size
            it      -   Number of iterations

        OUTPUTS:
            new_el  -   Updated elements matrix
        '''

        if el is None: return False

        for i in np.arange(0, it, 1): el = repel(el = el, r = r, tau = tau, dm = dm)

        return el

    # setup ordered neuronal structures
    cmap1 = plt.get_cmap(cmap_neurons)
    neurons_pos = np.zeros((net.neurons.shape[0], 2))
    neurons_col = np.zeros((net.neurons.shape[0], 4))
    struct_axes = np.ceil(np.sqrt(net.structs.shape[0]-1)).astype(np.int)
    struct_axes = struct_axes if struct_axes > 0 else 1
    struct_labels = [''] * net.structs.shape[0] if labels_struct is None else labels_struct
    struct_labels_pos = np.zeros((len(struct_labels), 2))

    # setup force-direction parameters
    struct_spacing = struct_spacing * struct_axes
    repulsion = 2.5 * net.structs.shape[0]
    tau_r = 0.25

    # setup neuron structs
    for struct in net.structs:
        struct_x = np.ceil(struct / struct_axes)
        struct_y = struct % struct_axes

        neurons = net.neurons_in(struct)
        neurons = np.squeeze(neurons).reshape((neurons.shape[0],))
        elements = np.random.normal(loc = 0.0, scale = 0.05, size=(neurons.shape[0], 2)) # sample (x, y) from gaussian
        neurons_pos[neurons,:] = force_position(el = elements, r = repulsion, tau = tau_r) + np.tile(np.array([struct_x, struct_y]).reshape((1, 2)) * struct_spacing, (elements.shape[0], 1)) # compute force-directed (x, y) for elements
        neurons_col[neurons,:] = cmap1(np.random.random(size=(1,))) if cmap_neurons_by_struct is True else cmap1(np.random.random(size=(neurons.shape[0],)))
        if cmap_neurons_by_state is True: neurons_col[neurons,:] = cmap1(cmap_neurons_sim.network.neurons[neurons,nl.PARAM_UNI.E_l.value] / cmap_neurons_sim.network.neurons[neurons,nl.PARAM_UNI.V_thr.value])
        if labels_struct is None: struct_labels[struct[0].astype(np.int)] = 'Population no. {:d}'.format(struct[0].astype(np.int))
        struct_labels_pos[struct,:] = np.mean(neurons_pos[neurons,:], axis = 0)


    # setup synapse structs
    synapses_pos = np.zeros((net.synapses.shape[0], 5))
    synapses_pos[:,0:2] = neurons_pos[net.synapses[:,1].astype(np.int),:]
    synapses_pos[:,2:4] = neurons_pos[net.synapses[:,2].astype(np.int),:]
    synapses_pos[:,4] = net.synapses[:,3] / np.max(net.synapses[:,3])

    cmap2 = plt.get_cmap(cmap_synapses)
    synapses_col = np.zeros((net.synapses.shape[0], 4))
    fibre_labels = [''] * net.fibres.shape[0] if labels_fibre is None else labels_fibre
    fibre_labels_pos = np.zeros((len(fibre_labels), 2))

    for fibre in net.fibres:
        synapses = net.synapses_in(fibre)
        synapses = np.squeeze(synapses).reshape((synapses.shape[0]))
        synapses_col[synapses,:] = cmap2(np.random.random(size=(1,))) if cmap_synapses_by_fibre is True else cmap2(np.random.random(size=(synapses.shape[0],)))
        if labels_fibre is None: fibre_labels[fibre[0].astype(np.int)] = 'Fibre no. {:d}'.format(fibre[0].astype(np.int))
        fibre_labels_pos[fibre,0] = np.mean(synapses_pos[synapses,0:4:2])
        fibre_labels_pos[fibre,1] = np.mean(synapses_pos[synapses,1:4:2]) + fibre[0] * fibre_spacing
    synapses_col[:,3] = synapse_alpha

    # setup plot
    fig, ax = plt.subplots(1)
    fig.set_tight_layout(True)
    ax.axis('off')

    # animation
    if cmap_neurons_by_state is True:
        states = cmap_neurons_sim.monitors[cmap_neurons_monitor].state.T
        states = states / np.max(states)

        if cmap_neurons_use_kernel is True:
            # setup gaussian
            std = 5
            n = np.arange(0, cmap_neurons_kernel_L, 1) - (cmap_neurons_kernel_L - 1) / 2
            n = np.tile(n, (states.shape[0], 1))
            sig2 = 2 * std * std
            k = np.exp(-n ** 2 / sig2)

            # setup smoothed states
            smoothed_anim_states = np.zeros((states.shape[0], states.shape[1] * cmap_neurons_kernel_L))

            # apply smoothing
            for i in np.arange(0, states.shape[1], 1):
                for l in np.arange(0, cmap_neurons_kernel_L, 1):
                    smoothed_anim_states[:,i*cmap_neurons_kernel_L+l] = k[:,l] * states[:,i]

            # set states
            states = smoothed_anim_states

        # animation function
        def animate(i):
            print('Animating frame {:d}. Hold on.'.format(i), end = '\r')

            neurons_col[:,:] = cmap1(states[:,i])

            ax.clear()
            ax.axis('off')

            # plot synapses
            ax.plot(np.array([synapses_pos[:,0], synapses_pos[:,2]]), np.array([synapses_pos[:,1], synapses_pos[:,3]]), zorder = 0)
            for n, j in enumerate(ax.lines):
                j.set_color(synapses_col[n,:])
                j.set_linewidth(synapses_pos[n,4])

            # plot neurons
            ax.scatter(neurons_pos[:,0], neurons_pos[:,1], c = neurons_col, zorder = 1)

            # label structs
            if show_labels_struct is True:
                for n in np.arange(0, struct_labels_pos.shape[0], 1):
                    ax.text(struct_labels_pos[n,0], struct_labels_pos[n,1], struct_labels[n], horizontalalignment = 'center', verticalalignment = 'center')

            # label fibres
            if show_labels_fibre is True:
                for n in np.arange(0, fibre_labels_pos.shape[0], 1):
                    ax.text(fibre_labels_pos[n,0], fibre_labels_pos[n,1], fibre_labels[n], horizontalalignment = 'center', verticalalignment = 'center')

            ax.text(0, -0.1, 't = {:d}'.format(i))

            return ()


        anim = FuncAnimation(fig, animate, frames = np.arange(0, states.shape[1], 1), interval = cmap_anim_interval, blit = True, repeat = True)
        anim.save(cmap_anim_save_to, dpi = 60, writer = 'Pillow')
    else:
        # plot synapses
        ax.plot(np.array([synapses_pos[:,0], synapses_pos[:,2]]), np.array([synapses_pos[:,1], synapses_pos[:,3]]), zorder = 0)
        for i, j in enumerate(ax.lines):
            j.set_color(synapses_col[i,:])
            j.set_linewidth(synapses_pos[i,4])

        # plot neurons
        ax.scatter(neurons_pos[:,0], neurons_pos[:,1], c = neurons_col, zorder = 1)

        # label structs
        if show_labels_struct is True:
            for i in np.arange(0, struct_labels_pos.shape[0], 1):
                ax.text(struct_labels_pos[i,0], struct_labels_pos[i,1], struct_labels[i], horizontalalignment = 'center', verticalalignment = 'center')

        # label fibres
        if show_labels_fibre is True:
            for i in np.arange(0, fibre_labels_pos.shape[0], 1):
                ax.text(fibre_labels_pos[i,0], fibre_labels_pos[i,1], fibre_labels[i], horizontalalignment = 'center', verticalalignment = 'center')



def show():
    '''
    Show plots (short-hand to avoid importing in main file)
    '''

    plt.show()
