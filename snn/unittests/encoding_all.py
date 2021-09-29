'''
Unit test for neural encoding utilities.

Creates an input pattern for 250 neurons (gradually getting weaker from in250-1)
and plots the encoded spike patterns produced.
'''

import numpy as np
import sys
sys.path.append('./../..')
import snn
import matplotlib.pyplot as plt

L = 250
input_pattern = np.arange(250, 0, -1).reshape((250, 1))

input_rate = snn.utils.neuralcoding.rate(inputs = input_pattern, L = L, lam = 4).astype(np.float)
input_ttfs = snn.utils.neuralcoding.ttfs(inputs = input_pattern, max = 250, T = L, dt = 1).astype(np.float)
input_phase = snn.utils.neuralcoding.phase(inputs = np.tile(input_pattern, (1, 15)), L = L, bits = 8).astype(np.float)
input_burst = snn.utils.neuralcoding.burst(inputs = input_pattern, L = L, max = 250, T_max = 10).astype(np.float)

fig, (ax1, ax2) = plt.subplots(2, 2)

spikes = input_rate
spikes[np.where(spikes == 0)] = np.nan
for i in range(spikes.shape[0]):
    ax1[0].scatter(np.arange(0, spikes.shape[1], 1), spikes[i,:] * (100 - i), marker = '|', linewidth = 0.5)
ax1[0].set_title('Rate encoding')


spikes = input_ttfs
spikes[np.where(spikes == 0)] = np.nan
for i in range(spikes.shape[0]):
    ax1[1].scatter(np.arange(0, spikes.shape[1], 1), spikes[i,:] * (100 - i), marker = '|', linewidth = 0.5)
ax1[1].set_title('Time-to-first-spike encoding')


spikes = input_phase
spikes[np.where(spikes == 0)] = np.nan
for i in range(spikes.shape[0]):
    ax2[0].scatter(np.arange(0, spikes.shape[1], 1), spikes[i,:] * (100 - i), marker = '|', linewidth = 0.5)
ax2[0].set_title('Phase encoding')


spikes = input_burst
spikes[np.where(spikes == 0)] = np.nan
for i in range(spikes.shape[0]):
    ax2[1].scatter(np.arange(0, spikes.shape[1], 1), spikes[i,:] * (100 - i), marker = '|', linewidth = 0.5)
ax2[1].set_title('Burst encoding')

plt.show()
