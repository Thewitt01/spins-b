#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 09:45:56 2019

@author: tom
"""

# This file defines the nessecary components for the overlap

import copy
import numpy as np
import sys
import scipy.sparse as sparse
import scipy.sparse.linalg as spalg
from typing import Dict, List

from spins.fdfd_tools import vec, unvec, dx_lists_t, vfield_t, field_t
from spins.fdfd_tools import operators, waveguide, functional
from . import phc_mode_solver as phc

from spins.gridlock import Direction

#Some constants
g = 2*np.pi*5.9e6
c = 3e8
eps0 = 8.854187e-12
kb = 1.3806505e-23
uma = 1.66053886e-27
m = 87*uma

#Desired radius of trap
r0 = 9e-7

#Blue potential displacement
h = 1.5e-7

#Constants that alter the potential
#Need to adjust
A = 2.7e9;
b = 8e13;
C = 1e7;
d = 1.5e7;
E = 2e7;
f = 1.3e7;

cell = 40e-9

x, y, z = np.meshgrid(np.arange(-62.5*cell, 62*cell, cell),
                      np.arange(-62.5*cell, 62*cell, cell), 0)

# Values of the components of the vector field
f_x = -A/np.sqrt(2)*np.exp(-b*(np.sqrt(x**2+y**2)-r0)**2)

f_y = -A/np.sqrt(2)*np.exp(-b*(np.sqrt(x**2+y**2)-r0)**2)

f_z = z*0

E = [f_x, f_y, f_z]




def annulus(dimx, dimy, center, big_radius, small_radius):
    """This produces a mask of true in an annulus shape depending on the
    entered parameters"""

    Y, X = np.ogrid[:dimx, :dimy]
    distance_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = (small_radius <= distance_from_center) & \
    (distance_from_center <= big_radius)

    return mask
# Need to match the dimensions
t = 125

#might need to be smaller too
mask = annulus(t, t, [0, 0], 0.2*t, 0.05*t) # can change 125 to variable



#code copied and edited from waveguide_mode.py
def compute_overlap_annulus(
        E: field_t,
        omega : complex, 
        dxes: dx_lists_t,
        axis: int,
        mu: field_t = None,
        ) -> field_t:
    """This is hopefully going to calculate the overlap """
    # Domain is all zero
    domain = np.zeros_like(E[0], dtype=int)

    # Then set the slices part equal to 1 - may need to use our mask
    domain[mask] = 1
    
    npts = E[0].size
    dn = np.zeros(npts * 3, dtype=int)
    dn[0:npts] = 1
    dn = np.roll(dn, npts * axis)
    
    e2h = operators.e2h(omega, dxes, mu)
    # Hopefully this works with the mask
    ds = sparse.diags(vec([domain] * 3))

    e_cross_ = operators.poynting_e_cross(vec(E), dxes)

    # Difference is, we have no H field
    overlap_e = dn @ ds @ (e_cross_ @ e2h)

    # Normalize
    norm_factor = np.abs(overlap_e @ vec(E))
    overlap_e /= norm_factor

    return unvec(overlap_e, E[0].shape)



def build_overlap_annulus(omega: complex, dxes: List[np.ndarray], eps: List[np.ndarray],
                  mu: List[np.ndarray] or None, axis: Direction or int,
                  mode_num: int, power: float) -> List[np.ndarray]:
    """This should call the overlap and  increase/decrease it to emit the desired power"""
    if type(axis) is Direction:
        axis = axis.value


    arg_overlap = {
        'E': E, # where we insert our desired electric field
        'axis': axis,
        'omega': omega,
        'dxes': dxes,
        'mu': vec(mu)
    }
    C = compute_overlap_annulus(**arg_overlap)

    
    # Increase/decrease C to emit desired power.
    for k in range(len(C)):
        C[k] *= np.sqrt(power)


    return C

