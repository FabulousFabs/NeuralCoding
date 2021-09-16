import numpy as np

class Full:
    '''
    Full NxM synapses class
    '''

    def __init__(self, efficacy = 1, delay = 1, n = 1, sd = 0, generator = None):
        '''
        Constructor

        INPUTS:
            efficacy    -   Synapse weight
            delay       -   Transmission delay
            n           -   Previous layer size (only applicable if generator = snn.generators.Xavier)
            generators  -   Generator function to use for sampling weights
        '''

        self.efficacy = efficacy
        self.delay = delay
        self.n = n
        self.sd = sd
        self.generator = generator

    def synapses(self, pre = None, post = None):
        '''
        Full connection matrix

        INPUTS:
            pre     -   Presynaptic neurons
            post    -   Postsynaptic neurons
        '''

        if pre is None or post is None: return False

        outs = np.array([])

        for pre_syn in pre:
            pre_syn_post = np.ones((post.shape[0], 9))
            pre_syn_post[:,1] = pre_syn
            pre_syn_post[:,2] = post
            pre_syn_post[:,3] = self.efficacy
            pre_syn_post[:,4] = self.delay

            outs = np.vstack((outs, pre_syn_post)) if outs.shape[0] > 0 else np.array(pre_syn_post)

        if self.generator is not None:
            outs[:,3] = self.generator(dim = outs.shape[0], efficacy = self.efficacy, n = self.n, sd = self.sd)

        return outs

class kWTA:
    '''
    kWTA class (an everybody-but-me implementation of NxM full connection)
    '''

    def __init__(self, efficacy = 1, delay = 1, n = 1, sd = 0, generator = None):
        '''
        Constructor

        INPUTS:
            efficacy    -   Synapse weight
            delay       -   Transmission delay
            n           -   Previous layer size (only applicable if generator = snn.generators.Xavier)
            generators  -   Generator function to use for sampling weights
        '''

        self.efficacy = efficacy
        self.delay = delay
        self.n = n
        self.sd = sd
        self.generator = generator

    def synapses(self, pre = None, post = None):
        '''
        kWTA connection matrix (for backpass)

        INPUTS:
            pre     -   Presynaptic neurons
            post    -   Postsynaptic neurons
        '''

        if pre is None or post is None: return False

        outs = np.array([])

        for i in np.arange(0, pre.shape[0]):
            c_pre = pre[i]
            c_post = np.delete(post, i)
            pre_syn_post = np.ones((post.shape[0]-1, 9))
            pre_syn_post[:,1] = c_pre
            pre_syn_post[:,2] = c_post
            pre_syn_post[:,3] = self.efficacy
            pre_syn_post[:,4] = self.delay

            outs = np.vstack((outs, pre_syn_post)) if outs.shape[0] > 0 else np.array(pre_syn_post)

        if self.generator is not None:
            outs[:,3] = self.generator(dim = outs.shape[0], efficacy = self.efficacy, n = self.n, sd = self.sd)

        return outs

class One_To_One:
    '''
    One to One connection class
    '''

    def __init__(self, efficacy = 1, delay = 1, n = 1, sd = 0, generator = None):
        '''
        Constructor

        INPUTS:
            efficacy    -   Synapse weight
            delay       -   Transmission delay
            n           -   Previous layer size (only applicable if generator = snn.generators.Xavier)
            generators  -   Generator function to use for sampling weights
        '''

        self.efficacy = efficacy
        self.delay = delay
        self.n = n
        self.sd = sd
        self.generator = generator

    def synapses(self, pre = None, post = None):
        '''
        Sparse one to one connection matrix

        INPUTS:
            pre     -   Presynaptic neurons
            post    -   Postsynaptic neurons
        '''

        if pre is None or post is None: return False

        assert(pre.shape == post.shape)

        outs = np.ones((pre.shape[0], 9))
        outs[:,1] = pre
        outs[:,2] = post
        outs[:,3] = self.efficacy
        outs[:,4] = self.delay

        if self.generator is not None:
            outs[:,3] = self.generator(dim = outs.shape[0], efficacy = self.efficacy, n = self.n, sd = self.sd)

        return outs

class Percent_To_One:
    '''
    Percentage of pre to one in post class
    '''

    def __init__(self, p = .25, efficacy = 1, delay = 1, n = 1, sd = 0, generator = None):
        '''
        Constructor

        INPUTS:
            efficacy    -   Synapse weight
            delay       -   Transmission delay
            n           -   Previous layer size (only applicable if generator = snn.generators.Xavier)
            generators  -   Generator function to use for sampling weights
        '''

        self.p = p
        self.efficacy = efficacy
        self.delay = delay
        self.n = n
        self.sd = sd
        self.generator = generator

    def synapses(self, pre = None, post = None):
        '''
        Distributed P% to 1 synapses

        INPUTS:
            pre     -   Presynaptic neurons
            post    -   Postsynaptic neurons
        '''

        if pre is None or post is None: return False

        p = np.round(pre.shape[0] * self.p).astype(np.int)
        outs = np.empty((0, 9), dtype=np.float)

        for i in np.arange(post.shape[0]):
            i_syn = np.ones((p, 9))
            i_syn[:,1] = np.random.choice(pre, size=p, replace=False)
            i_syn[:,2] = post[i]
            i_syn[:,3] = self.efficacy
            i_syn[:,4] = self.delay

            if self.generator is not None:
                i_syn[:,3] = self.generator(dim = i_syn.shape[0], efficacy = self.efficacy, n = self.n, sd = self.sd)

            outs = np.vstack((outs, i_syn))

        return outs
