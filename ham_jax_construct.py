
from typing import Union, Any,Callable, Sequence,Dict,Tuple,Optional
from dataclasses import dataclass, field
import functools
from functools import partial


import numpy as np
import flax.linen as nn
import flax

import jax.numpy as jnp
import jax
from jax.tree_util import Partial

import netket as nk
from netket import jax as nkjax
import math


def Ham_Heisenberg_pcl(hilb,latt_dim,J_list,brill_vecs,msr=False):
  
  ham = nk.operator.LocalOperator(hilb)
  for J,bvec in zip(J_list,brill_vecs):

    xys = 1.
    if msr and np.sum(np.abs(bvec))%2 == 1:
      xys = -1.

    print(f" {bvec}: J={J} and xys={xys}")
    for i in range(np.prod(latt_dim)):
      vi = np.array(np.unravel_index(i,latt_dim))
      vj = (vi + bvec + latt_dim)%latt_dim
      j = np.ravel_multi_index(vj,latt_dim)
      
      ham = ham + J*( nk.operator.spin.sigmaz(hilb, i) @ nk.operator.spin.sigmaz(hilb, j) )
      ham = ham + 2.0*xys*J*( nk.operator.spin.sigmap(hilb, i) @ nk.operator.spin.sigmam(hilb, j) )
      ham = ham + 2.0*xys*J*( nk.operator.spin.sigmam(hilb, i) @ nk.operator.spin.sigmap(hilb, j) )
      
  return ham.to_jax_operator()
  
def Ham_Heisenberg_J1_pcl(hilb,latt_dim,J1=.25,msr=False,norm_per_site=True):
  
  if norm_per_site:
    J1/= np.prod(latt_dim)
  
  ndim = len(latt_dim)
  J_list = [J1]*ndim
  if ndim ==1:
    brill_vecs = [ np.array( (1,) ) ]
  if ndim ==2:
    brill_vecs = [ np.array( (1,0) ) ,np.array( (0,1) ) ]
  if ndim ==3:
    brill_vecs = [ np.array( (1,0,0) ) ,np.array( (0,1,0) ),np.array( (0,0,1) ) ]
    
  return Ham_Heisenberg_pcl(hilb,latt_dim,J_list,brill_vecs,msr)

def Ham_Heisenberg_J1_J2_pcl(hilb,latt_dim,J1=.25,J2=0.0,msr=False,norm_per_site=True):
  
  if norm_per_site:
    J1/= np.prod(latt_dim)
    J2/= np.prod(latt_dim)
  
  ndim = len(latt_dim)
  
  if ndim ==1:
    J_list = [J1,J2]
    brill_vecs = [ np.array( (1,) ) , np.array( (2,) ) ]
  if ndim ==2:
    #J_list = [J1,J1,J2,J2,J2,J2]
    #brill_vecs = [np.array((1,0)),np.array((0,1)) , np.array((2,0)),np.array((0,2)) , np.array((1,1)),np.array((1,-1))]
    J_list = [J1,J1,J2,J2]
    brill_vecs = [np.array((1,0)),np.array((0,1))  , np.array((1,1)),np.array((1,-1))]
    
  return Ham_Heisenberg_pcl(hilb,latt_dim,J_list,brill_vecs,msr)





def Ham_TF_Ising_pcl(hilb,latt_dim,J=1.0,g=.5,norm_per_site=True):
  
  if norm_per_site:
    J/= np.prod(latt_dim)
    g/= np.prod(latt_dim)
  
  ndim = len(latt_dim)
  J_list = [J]*ndim
  if ndim ==1:
    brill_vecs = [ np.array( (1,) ) ]
  if ndim ==2:
    brill_vecs = [ np.array( (1,0) ) ,np.array( (0,1) ) ]
  if ndim ==3:
    brill_vecs = [ np.array( (1,0,0) ) ,np.array( (0,1,0) ),np.array( (0,0,1) ) ]

  ham = nk.operator.LocalOperator(hilb)
  for i in range(np.prod(latt_dim)):
    ham = ham - (.5*g)*nk.operator.spin.sigmax(hilb,i)
    vi = np.array(np.unravel_index(i,latt_dim))
    for J,bvec in zip(J_list,brill_vecs):
      vj = (vi + bvec + latt_dim)%latt_dim
      j = np.ravel_multi_index(vj,latt_dim)
      ham = ham - (.25*J)*( nk.operator.spin.sigmaz(hilb, i) @ nk.operator.spin.sigmaz(hilb, j) )
        
  return ham.to_jax_operator()


  
