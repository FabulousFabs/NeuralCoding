import numpy as np
import os, sys, librosa
sys.path.append('./../..')
import snn
import matplotlib.pyplot as plt

# setup paths
pwd = '/project/3018012.23/sim_test/sim_wordlearning/phase/'
swd = '/project/3018012.23/stimuli/audio/'
sod = os.path.join(pwd, 'stimuli')

# setup opts
F0 = 8
FMax = 8000
K = 50
L_window = 256
n_cycles = 5
bits = 8

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

def load_all(dir):
    '''
    Returns a tensor of .npy matrices
    '''

    files = find_files(dir, '.npy')
    tensor = np.array([])

    for file in files:
        npy = load(os.path.join(dir, file))
        tensor = np.append(tensor, np.array([npy]), axis=0) if tensor.shape[0] > 0 else np.array([npy])

    return tensor

def make_wavelet(F, fs, n_cycles = 5):
    '''
    Returns length, time vector and complex morlet wavelet at F with n_cycles
    '''

    L = (1 / F) * n_cycles
    t = np.arange(-L/2, L/2+(1/fs), 1/fs)

    csw = np.exp(1j * 2 * np.pi * F * t)
    win = np.hanning(t.shape[0])
    csw = csw * win

    return (len(csw), t, csw)

def get_CQT_freq(F0 = 15, Fm = 8000, K = 20):
    '''
    Returns centre frequencies and bandwidths of CQT as per Brown et al. 91/92 papers
    Note that the equation is slightly modified for fixed K. Ergo, we compute b not K.
    '''

    b = np.abs(K / np.log2(F0 / Fm))
    Q = (2 ** (1 / b) - 1) ** (-1)

    k = np.arange(K)
    Fk = F0 * (2 ** (k / b))
    Bk = Fk / Q

    return (Fk, Bk)

def do_convolve(signal, kernel, L = None, pad_left = 0, pad_right = 0):
    '''
    Convolve signal with kernel in time and return
    '''

    if L is None: L = kernel.shape[0]

    padding = np.floor(L / 2).astype(np.int)
    padded = np.pad(signal, (padding, padding), 'constant', constant_values = (pad_left, pad_right))
    convolved = np.zeros(signal.shape[0]).astype(np.complex)

    for k in np.arange(signal.shape[0]):
        convolved[k] = np.dot(padded[k:k+L], kernel)

    return convolved

def do_frame_tfs(convol, l = 64, pad = 0.0+0j):
    '''
    Compute spectral energy in striding windows
    '''

    s = int(l / 2)
    L = convol.shape[0]
    padding = int(L % s)
    padded = np.pad(convol, (0, padding), 'constant', constant_values = (0, pad))
    K = int(padded.shape[0] / s)
    E = np.zeros(K)

    for k in np.arange(K):
        E[k] = 10*np.log(np.sum(np.abs(convol[k*s:(k*s+l)])) + 1e-27)

    return E, l

# find targets
targets = find_files(swd, '.wav')

# enter loop
print('Starting phase encoding of stimuli.')

for i in np.arange(0, len(targets), 1):
    # log progress
    progress = i / len(targets)
    print('[\t\t\t\t\t]', end='\r')
    print('[' + ''.join(['-' for i in range(int(progress * 40))]), end='\r')
    print('\t\t{:2.2f}%.'.format(np.round(progress*100, 2)), end='\r')

    # load audio
    y, fs = librosa.load(os.path.join(swd, targets[i]))

    # setup spike matrix
    M = np.zeros((50, 236*bits))

    # compute centre frequencies
    Fk = get_CQT_freq(F0=F0, Fm=FMax, K=K)[0]
    Ek = np.array([])

    # compute spectral energy
    for fk in Fk:
        L, t, kernel = make_wavelet(fk, fs, n_cycles = n_cycles)
        filtered = do_convolve(y, kernel, L = L)
        E, l = do_frame_tfs(filtered, l = L_window)
        Ek = np.append(Ek, np.array([E]), axis = 0) if Ek.shape[0] > 0 else np.array([E])

    Ek = Ek + np.abs(np.min(Ek))
    Ek = Ek / np.max(Ek) * (2 ** bits - 1)

    # compute the phase response
    M[:,0:Ek.shape[1]*bits] = snn.utils.neuralcoding.phase(inputs = Ek.astype(np.int), L = Ek.shape[1] * bits, bits = bits)

    # save stimulus
    fn = targets[i].split('.')[0] + '.npy'
    with open(os.path.join(sod, fn), 'wb') as f:
        np.save(f, M)

print('Phase encoding completed.')
