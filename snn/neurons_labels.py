'''
Parameter labels for neurons
'''

from enum import Enum

# Universal labels
PARAM_UNI = Enum('UNI', 'struct type E_l V V_thr I tau_pos tau_neg x y a b tau_k w A inhib_ff inhib_fb', start = 0)

# LIF labels
PARAM_LIF = Enum('LIF', 'm N', start = len(PARAM_UNI))
