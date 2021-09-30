import numpy as np

from . import synapses

def AND(net = None):
    '''
    Implements an AND-gate in the supplied network.

    DT:
        1

    TRUTHTABLE:
        I0  I1  O
        0   0   0
        0   1   0
        1   0   0
        1   1   1

    INPUTS:
        net     -   Network object

    OUTPUTS:
        AND     -   Network identifiers for I0, I1, O (dictionary).
    '''

    if net is None: return False

    # input zero
    I0 = net.structure(A = 1, m = -1)
    # input one
    I1 = net.structure(A = 1, m = -1)
    # output (with threshold two, requiring both to spike or immediate decay)
    O = net.structure(V_thr = 2, m = -1)

    # connect both inputs to output
    I0_O = net.fibre(pre = I0, post = O, type = synapses.One_To_One(efficacy = 1, delay = 1))
    I1_O = net.fibre(pre = I1, post = O, type = synapses.One_To_One(efficacy = 1, delay = 1))

    return {'I0': I0, 'I1': I1, 'O': O}

def OR(net = None):
    '''
    Implements an OR-gate in the supplied network.

    DT:
        1

    TRUTHTABLE:
        I0  I1  O
        0   0   0
        0   1   1
        1   0   1
        1   1   1

    INPUTS:
        net     -   Network object

    OUTPUTS:
        OR      -   Network identifiers for I0, I1, O (dictionary).
    '''

    if net is None: return False

    # input zero
    I0 = net.structure(A = 1, m = -1)
    # input one
    I1 = net.structure(A = 1, m = -1)
    # output (with threshold one, requiring only one spike)
    O = net.structure(V_thr = 1, m = -1)

    # connect both inputs to output
    I0_O = net.fibre(pre = I0, post = O, type = synapses.One_To_One(efficacy = 1, delay = 1))
    I1_O = net.fibre(pre = I1, post = O, type = synapses.One_To_One(efficacy = 1, delay = 1))

    return {'I0': I0, 'I1': I1, 'O': O}

def NOR(net = None):
    '''
    Implements a NOR-gate in the supplied network.

    NOTE requires A to be activated by a spike in same time step as I0/I1.

    DT:
        1

    TRUTHTABLE:
        I0  I1  O
        0   0   1
        0   1   0
        1   0   0
        1   1   0

    INPUTS:
        net     -   Network object

    OUTPUTS:
        NOR     -   Network identifiers for A, I0, I1, O, D (dictionary).
    '''

    if net is None: return False

    # activation neuron
    A = net.structure(A = 1, m = -1)
    # input zero
    I0 = net.structure(A = 1, m = -1)
    # input one
    I1 = net.structure(A = 1, m = -1)
    # output (with threshold two, requiring both neurons not to spike)
    O = net.structure(V_thr = 2, m = -1)
    # done flag
    D = net.structure(A = 1, m = -1)

    # connect activation neuron to output with w=2 to cause a spike
    A_O = net.fibre(pre = A, post = O, type = synapses.One_To_One(efficacy = 2, delay = 1))

    # connect both inputs to output with w=-1 to cancel spike
    I0_O = net.fibre(pre = I0, post = O, type = synapses.One_To_One(efficacy = -1, delay = 1))
    I1_O = net.fibre(pre = I1, post = O, type = synapses.One_To_One(efficacy = -1, delay = 1))

    # connect activation neuron to done flag
    A_D = net.fibre(pre = A, post = D, type = synapses.One_To_One(efficacy = 1, delay = 1))

    return {'A': A, 'I0': I0, 'I1': I1, 'O': O, 'D': D}

def NAND(net = None):
    '''
    Implements a NAND-gate in the supplied network.

    NOTE requires A to be activated by a spike in same time step as I0/I1.

    DT:
        1

    TRUTHTABLE:
        I0  I1  O
        0   0   1
        0   1   1
        1   0   1
        1   1   0

    INPUTS:
        net     -   Network object

    OUTPUTS:
        NAND    -   Network identifiers for A, I0, I1, O, D (dictionary).
    '''

    if net is None: return False

    # activation neuron
    A = net.structure(A = 1, m = -1)
    # input zero
    I0 = net.structure(A = 1, m = -1)
    # input one
    I1 = net.structure(A = 1, m = -1)
    # output (with threshold 1 requiring only one spike)
    O = net.structure(V_thr = 1, m = -1)
    # done flag
    D = net.structure(A = 1, m = -1)

    # connect activation neuron to output with w = 2, twice the threshold
    A_O = net.fibre(pre = A, post = O, type = synapses.One_To_One(efficacy = 2, delay = 1))
    # connect inputs to output with w = -1 such that outputs are inhibited only if both spike
    I0_O = net.fibre(pre = I0, post = O, type = synapses.One_To_One(efficacy = -1, delay = 1))
    I1_O = net.fibre(pre = I1, post = O, type = synapses.One_To_One(efficacy = -1, delay = 1))
    # connect activation neuron to done flag
    A_D = net.fibre(pre = A, post = D, type = synapses.One_To_One(efficacy = 1, delay = 1))

    return {'A': A, 'I0': I0, 'I1': I1, 'O': O, 'D': D}

def XOR(net = None):
    '''
    Implements an XOR-gate in the supplied network.

    NOTE requires A to be activated by a spike in same time step as I0/I1.

    DT:
        4

    TRUTHTABLE:
        I0  I1  O
        0   0   0
        0   1   1
        1   0   1
        1   1   0

    INPUTS:
        net     -   Network object

    OUTPUTS:
        XOR     -   Network identifiers for A, I0, I1, O, D (dictionary).
    '''

    if net is None: return False

    # activation neuron
    A = net.structure(A = 1, m = -1)
    # input zero
    I0 = net.structure(A = 1, m = -1)
    # input one
    I1 = net.structure(A = 1, m = -1)
    # done flag
    D = net.structure(A = 1, m = -1)
    # OR-gate for parallelisation at step one
    OR1 = OR(net = net)
    # NAND-gate for parallelisation at step one
    NAND1 = NAND(net = net)
    # AND-gate for step two
    AND2 = AND(net = net)

    # connect done flag
    net.fibre(pre = A, post = D, type = synapses.One_To_One(efficacy = 1, delay = 4))

    # connect inputs to OR-gate (OR spikes _if there is one-to-two active unit_)
    net.fibre(pre = I0, post = OR1['I0'], type = synapses.One_To_One(efficacy = 1, delay = 1))
    net.fibre(pre = I1, post = OR1['I1'], type = synapses.One_To_One(efficacy = 1, delay = 1))

    # connect inputs to NAND-gate (NAND spikes _if there is none-to-one active unit_)
    net.fibre(pre = I0, post = NAND1['I0'], type = synapses.One_To_One(efficacy = 1, delay = 1))
    net.fibre(pre = I1, post = NAND1['I1'], type = synapses.One_To_One(efficacy = 1, delay = 1))
    net.fibre(pre = A, post = NAND1['A'], type = synapses.One_To_One(efficacy = 1, delay = 1))

    # connect OR and NAND to AND to combine activity - AND can only be true for the overlapping truth table entries
    # which is only the case for [0 1] or [1 0], yielding the XOR result
    net.fibre(pre = OR1['O'], post = AND2['I0'], type = synapses.One_To_One(efficacy = 1, delay = 1))
    net.fibre(pre = NAND1['O'], post = AND2['I1'], type = synapses.One_To_One(efficacy = 1, delay = 1))

    return {'I0': I0, 'I1': I1, 'A': A, 'O': AND2['O'], 'D': D}

def XNOR(net = None):
    '''
    Implements an XNOR-gate in the supplied network.

    NOTE requires A to be activated by a spike in same time step as I0/I1.

    DT:
        4

    TRUTHTABLE:
        I0  I1  O
        0   0   1
        0   1   0
        1   0   0
        1   1   1

    INPUTS:
        net     -   Network object

    OUTPUTS:
        XOR     -   Network identifiers for A, I0, I1, O, D (dictionary).
    '''

    if net is None: return False

    # activation neuron
    A = net.structure(A = 1, m = -1)
    # input zero
    I0 = net.structure(A = 1, m = -1)
    # input one
    I1 = net.structure(A = 1, m = -1)
    # done flag
    D = net.structure(A = 1, m = -1)
    # AND gate for parallelisation in step one
    AND1 = AND(net = net)
    # NOR gate for parallelisation in step one
    NOR1 = NOR(net = net)
    # OR gate for step two
    OR2 = OR(net = net)

    # connect done flag
    net.fibre(pre = A, post = D, type = synapses.One_To_One(efficacy = 1, delay = 4))

    #connect inputs to AND-gate (spiking only at full activity)
    net.fibre(pre = I0, post = AND1['I0'], type = synapses.One_To_One(efficacy = 1, delay = 1))
    net.fibre(pre = I1, post = AND1['I1'], type = synapses.One_To_One(efficacy = 1, delay = 1))

    # connect inputs to NOR-gate (spiking only at zero activity)
    net.fibre(pre = I0, post = NOR1['I0'], type = synapses.One_To_One(efficacy = 1, delay = 1))
    net.fibre(pre = I1, post = NOR1['I1'], type = synapses.One_To_One(efficacy = 1, delay = 1))
    net.fibre(pre = A, post = NOR1['A'], type = synapses.One_To_One(efficacy = 1, delay = 1))

    # pool outputs in OR-gate, spiking only if there is activity (so, either full or no activity)
    net.fibre(pre = AND1['O'], post = OR2['I0'], type = synapses.One_To_One(efficacy = 1, delay = 1))
    net.fibre(pre = NOR1['O'], post = OR2['I1'], type = synapses.One_To_One(efficacy = 1, delay = 1))

    return {'I0': I0, 'I1': I1, 'A': A, 'O': OR2['O'], 'D': D}

def MUX(net = None):
    '''
    Implements a MUX-gate in the supplied network.

    NOTE requires A to be activated by a spike in same time step as I0/I1.

    DT:
        4

    TRUTHTABLE:
        S0  I0  I1  O
        0   0   0   0
        0   1   0   1
        0   0   1   0
        0   1   1   1
        1   0   0   0
        1   1   0   0
        1   0   1   1
        1   1   1   1

    INPUTS:
        net     -   Network object

    OUTPUTS:
        MUX     -   Network identifiers for A, I0, I1, S, O, D (dictionary).
    '''

    if net is None: return False

    # activation neuron
    A = net.structure(A = 1, m = -1)
    # input zero
    I0 = net.structure(A = 1, m = -1)
    # input one
    I1 = net.structure(A = 1, m = -1)
    # selector neuron
    S = net.structure(A = 1, m = -1)
    # done flag
    D = net.structure(A = 1, m = -1)
    # NAND-gated NOT selector
    NOTS = NAND(net = net)
    # NOT_selector * I0 neuron (threshold two to require two simultaneous input spikes)
    I00 = net.structure(V_thr = 2, A = 1, m = -1)
    # selector * I1 neuron (threshold two to require two simultaneous inptu spikes)
    I11 = net.structure(V_thr = 2, A = 1, m = -1)
    # output
    O = net.structure(V_thr = 1, m = -1)

    # connect done flag
    net.fibre(pre = A, post = D, type = synapses.One_To_One(efficacy = 1, delay = 4))

    # connect (S, S) to NAND-gate to invert S
    net.fibre(pre = S, post = NOTS['I0'], type = synapses.One_To_One(efficacy = 1, delay = 1))
    net.fibre(pre = S, post = NOTS['I1'], type = synapses.One_To_One(efficacy = 1, delay = 1))
    net.fibre(pre = A, post = NOTS['A'], type = synapses.One_To_One(efficacy = 1, delay = 1))

    # connect NOT selector and I0 to I00, firing only if both have spiked
    net.fibre(pre = NOTS['O'], post = I00, type = synapses.One_To_One(efficacy = 1, delay = 1))
    net.fibre(pre = I0, post = I00, type = synapses.One_To_One(efficacy = 1, delay = 3))

    # connect S + I1 to I11, firing only if both have spiked
    net.fibre(pre = S, post = I11, type = synapses.One_To_One(efficacy = 1, delay = 3))
    net.fibre(pre = I1, post = I11, type = synapses.One_To_One(efficacy = 1, delay = 3))

    # connect the activity-selected neurons to output, producing a spike if activity remains
    net.fibre(pre = I00, post = O, type = synapses.One_To_One(efficacy = 1, delay = 1))
    net.fibre(pre = I11, post = O, type = synapses.One_To_One(efficacy = 1, delay = 1))

    return {'I0': I0, 'I1': I1, 'S': S, 'A': A, 'D': D, 'O': O}
