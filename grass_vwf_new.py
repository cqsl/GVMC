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
from jax.tree_util import Partial
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
from netket.jax import vmap_chunked

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
    # variable for shifting log vwf coeffecients outputs by a constant for each vwf to avoid overflows/ better num stability when taking exp
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




# function for normalizing rows of matrix coeffeicents given in terms of the log (can be overwritten)
# This default function normalizes each row the by the largest abs value of each row after expontntial.
def row_norm_func(mat_log_elem):
  return jnp.max(jnp.real(mat_log_elem),axis=-1)

#Input (mat_log_elem): array containing the logs of the matrix elements X_...ij = log( <sigma_...i | phi_j > )
#Output (row_norm_overlap_mat): the exponential of the inputed log matrix elements normalized by row_norm_func (f)
#Output (row_log_norm): the output of row_norm_func as the log of the row norms
# A -> X,c implies A_ij = exp( X_ij - c_i ) for c_i = f(X_i1,X_i2,...)
# Ie. A_ij = e^(-c_i) * <sigma_i | phi_j > so effectively each sampled basis state sigma_i is renormlized as <sigma_i | -> e^-c_i * <sigma_i |
#by defualt "row_norm_func" gives c_i = max( real(X_i1), real(X_i2),...) -> A_ij = exp( X_ij) / max( |exp( X_i1)|,|exp( X_i2)|,...)
# This row norm helps alot with the num stability for inverting 
# row_log_norm (c) needs to be saved for some calcs eg. local operators (next func)

def overlap_mat(mat_log_elem ):
  row_log_norm = row_norm_func(mat_log_elem)
  row_norm_overlap_mat = jnp.exp(mat_log_elem - jnp.expand_dims(row_log_norm,axis=-1))
  return row_norm_overlap_mat,row_log_norm



#Calculates row normalized Operator overlap matrices Op_Phi_...ij = e^(-c_i)*<sigma_i | O | phi_j > 
#mod_apply_fn,params,state: apply function, params, and state of Grass_VWF_Module
#sigma: input state array in standard Netket but n_chains and n_samps_per_chain combined as n_samp
#row_log_norm: the log row norms given by "overlap_mat" for inputs sigma
#Jax_Op: Operator assumed as Netket Jax type operator (Jax type needed for sharding)
# chunk_size: number of overlap matrices out of n_samp total to be calculated simetenosuely (memory issues arise if not chunked)

def operator_overlap_mat(mod_apply_fn,params,state,sigma,row_log_norm,Jax_Op,chunk_size=1 ):
#def HPhi_calc_batch(sigma,chunk_size,ham,Phi_log_norm,mod_apply_fn,params,state ):

  #apply func as only a function of inputs x/sigma
  maf = Partial( lambda x: mod_apply_fn({"params": params, **state}, x) )

  #function for calculating row normalized overlap matrices for a chunk/batch of inputs sigma
  def oomb(sigma_batch,rln ):
    eta_batch, Op_mat_elem = Jax_Op.get_conn_padded(sigma_batch)
    log_Phi_eta = maf(eta_batch )
    Op_Phi_batch = jnp.sum( jnp.expand_dims(Op_mat_elem,axis=-1)*jnp.exp( log_Phi_eta - rln ) , axis = -2)
    return Op_Phi_batch

  #Netket's chunked vmap of "oomb"
  Op_Phi = vmap_chunked(oomb,in_axes=(0,0) , chunk_size=chunk_size)( sigma,jnp.expand_dims(row_log_norm,axis=-1))
  return Op_Phi


    



# Calculates the individual vwf log norms based on sampled states so that Expected_sigma[ |phi_i(sigma)| ] = 1
#log_Phi_sigma: array of log vwf coefficents log_Phi_sigma...ij = 
def renorm_Phi(log_Phi_sigma,vec_state):

  vec_state_wo_norm, vec_state_norm = flax.core.pop(vec_state, 'norm') 
  pln = vec_state_norm['log_norm']
    
  a = jnp.max( jnp.real(log_Phi_sigma), axis=(0,1) )
  dpln = jnp.mean( jnp.exp( 2. * (jnp.real(log_Phi_sigma )- a))  , axis=(0,1) )
  dpln = a + .5 * jnp.log( dpln)
  #dpln *= .00
    
  #log_Phi_sigma = log_Phi_sigma - dpln
  #vec_state = {'norm': {'log_norm': pln+dpln } } 
  pln = pln + dpln
  #pln = 0.0 *pln +dpln
  vec_state = {'norm': {'log_norm': pln } , **vec_state_wo_norm}

    
  #vec_state = {'norm': {'log_norm': pln+.01*dpln} }
  log_Phi_sigma = log_Phi_sigma - dpln
  #log_Phi_sigma = mod_apply_fn_vec({'params': vec_params,**vec_state} , sigma)
  return log_Phi_sigma,vec_state





#used by sampler for quick matrix inverse and det calcs for rank 1 updates
def det_and_inv_row_update(A0_inv,A_row_k,k):
  v= A_row_k @ A0_inv
  det_ratio = v.at[k].get()
  v = (1. / det_ratio) *( v.at[k].add(-1.) )
  return det_ratio , A0_inv - jnp.outer(A0_inv.at[:,k].get() , v)
  
#det_and_inv_row_update_batch = jax.jit(jax.vmap(det_and_inv_row_update))
det_and_inv_row_update_batch = jax.vmap(det_and_inv_row_update)
