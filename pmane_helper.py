'''
Quick helper functions. This is a separate file because I really don't want
to load the pmane (particularly because it loads librosa) every time which
would be really quite inefficient.
'''

import numpy as np
import os

def load(f):
    ''' Returns the .npy matrix '''

    return np.load(f)

def find_files(f, t):
    ''' Grab all files of type t from f '''

    af = os.listdir(f)
    at = []
    for f in af:
        if f.endswith(t):
            at.append(f)
    return at

def load_all(dir):
    ''' Returns a tensor of .npy matrices '''

    files = find_files(dir, '.npy')
    tensor = np.array([])

    for file in files:
        npy = load(os.path.join(dir, file))
        tensor = np.append(tensor, np.array([npy]), axis=0) if tensor.shape[0] > 0 else np.array([npy])

    return tensor
