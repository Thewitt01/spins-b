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
from spins.invdes.problem_graph.simspace import SimulationSpace

from spins.gridlock import Direction


def annulus_field(
        cell_num: int,
        cell_height: int,
        r0 : float,
        h: float
):
    """Defines our E-field

    r0: Desired radius of trap
    h: Blue potential displacement"""

    #Constants that alter the potential
    #TODO Need to adjust
    A = 2.7e9;
    b = 8e13;
    C = 1e7;
    d = 1.5e7;
    E = 2e7;
    f = 1.3e7;

    cell = 40e-9

    x, y, z = np.meshgrid(np.arange(-(cell_num/2)*cell, ((cell_num-1)/2)*cell, cell),
                          np.arange(-(cell_num/2)*cell, ((cell_num-1)/2)*cell, cell),
                          np.arange(-(cell_height/2)*cell, (cell_height/2)*cell, cell))
    #added a value for the height, which should hopefully not take too long
    theta = np.arctan(y/x)

    # Values of the components of the vector field
    f_x = -A*x/abs(x)*np.cos(theta)*np.exp(-b*(np.sqrt(x**2+y**2)-r0)**2)

    f_y = -A*y/abs(y)*np.sin(theta)*np.exp(-b*(np.sqrt(x**2+y**2)-r0)**2)

    f_z = z*0

    E = [f_x, f_y, f_z]
    return E


def annulus(dimx, dimy, center, big_radius, small_radius):
    """This produces a mask of true in an annulus shape depending on the
    entered parameters"""

    Y, X = np.ogrid[:dimx, :dimy]
    distance_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = (small_radius <= distance_from_center) & \
    (distance_from_center <= big_radius)

    return mask
# Need to match the dimensions


#code copied and edited from waveguide_mode.py
def compute_overlap_annulus(
        E: field_t,
        omega : complex, 
        dxes: dx_lists_t,
        axis: int,
        mu: field_t = None,
        ) -> field_t:
    """This is hopefully going to calculate the overlap """

    # need to extract the size of dxes to adjust the size of mask and the E field
    len_dxes = np.concatenate(dxes, axis=0)

    # want to extract the absolute value, x=0, y=1, z=2

    # Domain is all zero
    domain = np.zeros_like(E[0], dtype=int)



    # Then set the slices part equal to 1 - may need to use our mask
    t = np.abs(len_dxes[0].size)

    # TODO adjust these values, may call from problem
    mask = annulus(t, t, [0, 0], 0.6 * t, 0.5 * t)
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

    len_dxes = np.concatenate(dxes, axis=0)
    E = annulus_field(np.abs(len_dxes[0].size), np.abs(len_dxes[2].size), r0=9e-7, h=1.5e-7)

    arg_overlap = {
        'E': E,  # where we insert our desired electric field
        'axis': axis,
        'omega': omega,
        'dxes': dxes,
        'mu': vec(mu)
    }
    C = compute_overlap_annulus(**arg_overlap)

    # Increase/decrease C to emit desired power.
    for k in range(len(C)):
        C[k] *= np.sqrt(power)
    # Might want to return absolute value to remove phase - list or array
    [abs(x) for x in C]
    return C

