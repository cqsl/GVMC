


from typing import Union, Any,Callable, Sequence,Dict,Optional
from dataclasses import dataclass, field

import json

import numpy as np
#import numpy as scipy

import jax.numpy as jnp
import jax
from jax import random
from jax.flatten_util import ravel_pytree

import flax.linen as nn
import flax
from flax.core import FrozenDict
from flax.linen.dtypes import promote_dtype

import functools
from functools import partial
from jax.tree_util import Partial

import netket as nk
from netket import jax as nkjax
from netket.utils import (
    maybe_wrap_module,
    mpi,
    wrap_afun,
    wrap_to_support_scalar,
)
from netket.jax import vmap_chunked


import grass_sr as gsr
import grass_vwf as gvwf

import math







@partial(jax.jit, static_argnames=['mod_apply_fn_vec','params_unravel','chunk_sizes','mode'])
def excited_update_step(mod_apply_fn_vec,params_unravel,sigma,ham,chunk_sizes, vec_params,vec_state,opt_vars,grass_vars,mode='complex'):

  chunk_size_energy,chunk_size_derv  = chunk_sizes
  #nsamp,nvwf,nparam = arr_dim
  nsamp,nvwf = sigma.shape[0],sigma.shape[1]
  n_sites = np.prod( sigma.shape[2:])


  #calculate log(<sigma_i| phi_j >) as nsamp x nvwf x nvwf array
  log_Phi_sigma = mod_apply_fn_vec({'params': vec_params,**vec_state} , sigma)
  #renormalize and update vec_state ('log_norm' modudle variable array)
  log_Phi_sigma,vec_state = gvwf.renorm_Phi(log_Phi_sigma,vec_state)
  

  Phi,Phi_log_norm = gvwf.overlap_mat(log_Phi_sigma)
  Phi_log_norm = jnp.expand_dims(Phi_log_norm,axis=-1)
  Phi_inv =jnp.linalg.inv(Phi)

  
  HPhi = gvwf.operator_overlap_mat(mod_apply_fn_vec,vec_params,vec_state,sigma,Phi_log_norm,ham,chunk_size_energy )
  
  
  H_batch = jnp.matmul(Phi_inv,HPhi)
  H = jnp.mean(H_batch,axis=0)
  E_batch = jnp.trace(H_batch,axis1=1,axis2=2)
  E_H=jnp.mean( H_batch * E_batch.conj().reshape( (E_batch.shape[0],1,1)) , axis=0)
  E=jnp.mean(E_batch)
  H_var =  E_H - E.conj() * H
  E_avg = jnp.trace(H) / nvwf 
  E_var = jnp.real(jnp.trace(H_var)) / nvwf
  
 
  Phi_inv = jnp.swapaxes(Phi_inv,1,2)
  #J = gsr.vec_jacobian_shared_sum_batch(mod_apply_fn_vec,vec_params,sigma, Phi*Phi_inv,vec_state,chunk_size=chunk_size_derv)
  J = gsr.vwf_vec_jacobian(mod_apply_fn_vec,vec_params,sigma, Phi*Phi_inv,vec_state,chunk_size=chunk_size_derv)
  B = (1.0 / math.sqrt(nsamp)) * (E_batch - jnp.mean(E_batch))

  '''
  FE_Temp=.5
  sign, Q = jnp.linalg.slogdet(Phi)
  Q = -(2.0 / (nvwf*n_sites*np.log(2.0) ) )      * ( Q + jnp.sum( log_Phi_sigma ,axis=(-2,-1) ) );
  B = B - (FE_Temp / math.sqrt(nsamp)) * (Q- jnp.mean(Q))
  B = (2.0 / (2.0 +FE_Temp))*B
  '''
  

  
  if mode == "complex":
    B = jnp.concatenate( (jnp.real(B),jnp.imag(B)) , axis=0 )
    J = jnp.concatenate( (jnp.real(J),jnp.imag(J)) , axis=0 )
   
  vec_params,opt_vars = gsr.update_params(J,B,vec_params,opt_vars,params_unravel,nvwf)
  
  grass_vars = {'E_avg':E_avg, 'E_var': E_var ,'H':H ,'H_var':H_var}
  
 
  return vec_params,vec_state,opt_vars,grass_vars


