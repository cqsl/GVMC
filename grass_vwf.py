
from typing import (
  Any,
  Callable,
  Dict,
  Iterable,
  Mapping,
  Optional,
  Sequence,
  Tuple,
  Type,
  TypeVar,
  Union,
)
from dataclasses import dataclass, field

import netket as nk
from netket.utils import (
    maybe_wrap_module,
    mpi,
    wrap_afun,
    wrap_to_support_scalar,
)
from netket.utils.types import PyTree, PRNGKeyT
from netket import jax as nkjax

import json
import numpy as np

from functools import partial
from flax.linen import module as linen_module
from flax.linen.module import (
  Module,
  Variable,
  _derive_profiling_name,
  _get_unbound_fn,
  wrap_method_once,
)
from flax import struct
import flax.linen as nn
import flax

import jax.numpy as jnp
import jax
from jax import random
from jax.nn.initializers import normal
from jax.flatten_util import ravel_pytree
from flax.core import FrozenDict
from netket import nn as nknn

from vwf_custom import RBM_custom,RBM_custom2,RBMModPhase_custom


import math



class Grass_VWF_Vec(Module):
  VWF_Module: Type[Module]

    
  #ind_module_meta: dict

    
  #ind_module_meta: Dict[str,Any] = field(default_factory=dict) #
  #ind_module_meta: PyTree = struct.field(pytree_node=True)
  n_model: int 
  #vec_axis: int =0

  

  def setup(self):
    self.variable('norm', 'log_norm', jnp.zeros, (self.n_model,) ,jnp.complex128)
    #self.variable('norm', 'log_norm', nn.initializers.zeros, (self.n_model,) ,np.complex128)


    
  @nn.compact
  def __call__(self, input):


      
    vec_vwf = nn.vmap(self.VWF_Module, 
      variable_axes={'params': 0}, 
      split_rngs={'params': True}, 
      in_axes=None,
      out_axes=-1,
      axis_size=self.n_model,
    )
    #x = vec_vwf(name='ind_params',**self.ind_module_meta)(inputs)
    x = vec_vwf(name='ind_params_stacked')(input)


    #y = self.variable('norm', 'log_norm', jnp.zeros, (self.n_model,) ,jnp.complex128)
    #y = self.param('norm', 'log_norm', jnp.zeros, (self.n_model,) ,jnp.complex128)
    #log_norm = self.variable('norm', 'log_norm', nn.initializers.zeros, (self.n_model,) ,np.complex128)
    #log_norm = self.variable('norm', 'log_norm', nn.initializers.zeros, (self.n_model,) ,np.float64)
    y = self.variables['norm'][ 'log_norm']

    
    #return x - y.value
    return x - y
    #return x

  
#Creates a vectorized backflow vwf module whose applyfunction outputs each log vwf coefficents along the last axis dim.
# Inputs are fed into a single "BF_Module" 
# Output of "BF_Module" is fed into seperately into each n_model copy of  "Ind_Module"

class Grass_VWF_Vec_BF(Module):
  BF_Module: Type[Module] # Shared backflow modulde whose weights are shared by all vwfs
  Ind_Module: Type[Module] # Invdivdual module which is vectorized / duplicated for each vwf with a seperate set of weights.
  n_model: int # number of vwfs (could be changed to n_vwf)
  spac_dim: Tuple[int] # tuple of spatial/lattice dimensions of the input states: prod(spac_dim) = n_sites
  features_in: int =1 # optional additional feature dim for inputs (in most cases will be 1 can also be used for bravais lattices index )
  
   
  
  def setup(self):
    self.variable('norm', 'log_norm', jnp.zeros, (self.n_model,) ,jnp.complex128)
    # variable for shifting log vwf coeffecient vwf oututs to avoid overflows/ better num stability when taking exp
    #self.variable('norm', 'log_norm', nn.initializers.zeros, (self.n_model,) ,np.complex128)
      
  @nn.compact
  def __call__(self, inputs):
    batch_dim = inputs.shape[:-1]
    x = inputs.reshape(*batch_dim,*self.spac_dim,self.features_in) #Input basis states are assumed to be of the form of Netket sampler 
    # Neket arrays of shape [bath_dim, n_sites] are reshaped as [bath_dim, spac_dim,featurues_in] for input into CNNs and tranformers
    back_flow = self.BF_Module
    x = back_flow( name = 'bf_params')(x) # inputs are fed into shared back_flow modulde with dict key "bf_params"

    # Ind_Module is vectorized by flax.linen version of vmap for moduldes
    # vec_vwf is a modudle whose apply func is vmap of the Ind_Module (vec dim is the last)
    # vec_vwf params are Ind_Module params concanted in the last dim
    # [ind_dim_1,ind_dim_2,...] -> [ind_dim_1,ind_dim_2,...,n_model] so ith vwf params are [:,:,...,i]
    vec_vwf = nn.vmap(self.Ind_Module, 
      variable_axes={'params': 0}, 
      split_rngs={'params': True}, 
      in_axes=None,
      out_axes=-1,
      axis_size=self.n_model,
    )
    x = vec_vwf(name='ind_params')(x) # back_flow output is fed into vec_vwf
    return x - self.variables['norm']['log_norm'] # output is normalized by log_norm ie. log(psi_i(x)) -> log(psi_i(x)) - log_norm_i





    
'''
class Grass_VWF_Vec_BF(Module):
  BF_Module: Type[Module]
  #bf_module_meta: dict
  Ind_Module: Type[Module]
  #ind_module_meta: dict
  n_model: int 
  
   
  
  def setup(self):
  
    self.back_flow = self.BF_Module(**self.bf_module_meta)
  
    self.vec_vwf = nn.vmap(self.Ind_Module, 
      variable_axes={'params': 0}, 
      split_rngs={'params': True}, 
      in_axes=None,
      out_axes=-1,
      axis_size=self.n_model,
      )(**self.ind_module_meta)
      
    self.variable('norm', 'log_norm', jnp.zeros, (self.n_model,) ,jnp.complex128)# normalizes each vwf not used at the moment but may later if overflow issues arise
   
  @nn.compact
  def __call__(self, input):
    x = jnp.expand_dims(input, axis=-1)
    x = self.back_flow(x)
    x = self.vec_vwf(x)
    return x - self.variables['norm']['log_norm']

'''



def row_norm_func(mat_log_elem):
  return jnp.max(jnp.real(mat_log_elem),axis=-1)

#@jax.jit 
def overlap_mat(mat_log_elem ):
  #row_log_norm = jnp.max(jnp.real(mat_log_elem),axis=-1,keepdims=True)
  row_log_norm = row_norm_func(mat_log_elem)
  row_norm_overlap_mat = jnp.exp(mat_log_elem - jnp.expand_dims(row_log_norm,axis=-1))
  return row_norm_overlap_mat,row_log_norm
  

def det_and_inv_row_update(A0_inv,A_row_k,k):
  v= A_row_k @ A0_inv
  det_ratio = v.at[k].get()
  v = (1. / det_ratio) *( v.at[k].add(-1.) )
  return det_ratio , A0_inv - jnp.outer(A0_inv.at[:,k].get() , v)
  
#det_and_inv_row_update_batch = jax.jit(jax.vmap(det_and_inv_row_update))
det_and_inv_row_update_batch = jax.vmap(det_and_inv_row_update)


























