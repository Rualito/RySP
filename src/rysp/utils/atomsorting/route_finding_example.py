#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 12:02:01 2022

@author: robert
"""

import numpy as np
import matplotlib.pyplot as plt
import time


from RSP_Optimization.Atom_Sorting.routeoptimizerbrowaeys import *
from RSP_Other.Classes.TrapSiteClass import *
from RSP_Other.Classes.AtomClass import *
from RSP_Other.Classes.LaserClass import *

np.random.seed(10)

dims=20
iters=10
costs=np.zeros([dims,iters,2])

dim_in=4
dim_out=6
gridsize_x=6
gridsize_y=6
num_atoms=np.int64(dim_out**2/2)

R=1

#Trapping laser description
grid=create_grid([dim_out,dim_out])
template=create_template([dim_in,dim_in],[(dim_out-dim_in)/2,(dim_out-dim_in)/2])
num_traps=len(grid)
TrapLasers=np.ndarray([num_traps],dtype=object)
Traps=np.ndarray([num_traps],dtype=object)

laser_efficiency=0.10
laser_decrease_factor=0.02
P=np.zeros([num_traps])+8
P0=P*laser_efficiency*laser_decrease_factor
w0=np.zeros([num_traps])+0.8*10**(-6) #waist-size of laser
wavelength=np.zeros([num_traps])+813.035*10**(-9) #wave length trap in nm
n_ref=1
polarization=0
z_traps=0
sim_traps=True
Env_Temp=0

#Atom state space description
spin_basis=np.array([[5,0,0,0,0],[5,1,0,0,1],[61,0,1,0,1]],np.int32)
spin_basis_labels=["0","1","r"]
motional_basis=[[0,0,0]]
motional_basis_labels=["000"]
Temp=0
interacting_states=[2]
dims=[[len(spin_basis),len(motional_basis)]*num_atoms]*2


Atoms=np.ndarray([num_atoms],dtype=object)
sim_traps=True

for k in range(num_traps):
    TrapLasers[k]=Laser(P0[k],w0[k],polarization,wavelength[k])
    Traps[k]=Trap_Site(grid[k][0], grid[k][1], z_traps, TrapLasers[k], n_ref)

initially_filled_traps=np.random.permutation(range(num_traps))[range(num_atoms)]
for k in range(num_atoms):
    Atoms[k]=Atom_In_Trap(Strontium88(), Traps[initially_filled_traps[k]], "ground_state", spin_basis,spin_basis_labels,motional_basis,motional_basis_labels,Temp,interacting_states)
    if (not(k==0)) and sim_traps==True:
        Atoms[k].update_polarizability(Atoms[0].polarizability)
        Atoms[k].update_trap_frequencies(Atoms[0].trap_frequencies)
        Atoms[k].update_pos([Atoms[k].trap_site.x,Atoms[k].trap_site.y])
    else:
        Atoms[k].calc_polarizability()
        Atoms[k].calc_trap_frequencies()
        Atoms[k].update_pos([Atoms[k].trap_site.x,Atoms[k].trap_site.y])


ROB=RouteOptimizerBrowaeys(Atoms, template, grid)
ROB.calc_costs(2,"lp")

q=ROB.create_assignment("Hungarian")
print(q)
print(ROB.cost_of_assignment())
ROB.plot_situation()
time.sleep(5)

ROB.execute_reordering(order="shortest_first",move_coll="no_coll",draw=1)
print(ROB.cost_of_assignment())





