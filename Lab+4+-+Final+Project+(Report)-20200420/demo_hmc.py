# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 10:04:09 2020

@author: Tisana
"""

import numpy as np
import matplotlib.pyplot as plt
import lab4_hmc as hmc


def e_func(x, f):
    # Simulating some unknown log-probability
    p0 = -x[0]**2/2
    p1 = -x[1]**2/2 + np.log(2+np.cos(f*x[1]))
    lgp = p0 + p1
    return -lgp

def e_grad(x, f):
    g = np.empty(2)
    g[0] = x[0]
    g[1] = x[1] + f*np.sin(f*x[1]) / (2+np.cos(f*x[1]))
    return g


f = 5  # The "frequency" argument for the energy, used here to demonstrate use of "args"
# Plotting parameters
fsz = (10,8)
gsz = 100
lim = 3
#
gx = np.linspace(-lim, lim, gsz)
GX, GY = np.meshgrid(gx, gx)
Gsz = GX.size
G = np.hstack((GX.reshape((Gsz, 1)), GY.reshape((Gsz, 1))))
#
plt.figure(figsize=fsz)
P = np.asarray([np.exp(-e_func(g, f)) for g in G])
plt.contour(GX, GY, P.reshape((gsz, gsz)), cmap='Reds', linewidths=3, zorder=1);



# Initial state: something random and sensible
x0 = np.random.normal(size=2)
hmc.gradient_check(x0, e_func, e_grad, f)



#
np.random.seed(seed=1)  # For reproducibility
R = 10000  # More than really needed, but produces a nice dense plot
burn = int(R/10)  # A reasonable rule-of-thumb
L = 20  # OK here (should be larger in regression sampling)
eps = 0.3  # Trial-and-error ... feel free to experiment!
#
S, *_ = hmc.sample(x0, e_func, e_grad, R, L, eps, burn=burn, checkgrad=False, args=[f])
#
plt.figure(figsize=fsz)
plt.plot(S[:, 0], S[:, 1], '.', ms=6, color='CadetBlue', alpha=0.25, zorder=0)
plt.contour(GX, GY, P.reshape((gsz, gsz)), cmap='Reds', linewidths=3, zorder=1);
