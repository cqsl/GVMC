from functools import partial
from typing import Optional
from netket.utils.types import Array, Callable, PyTree


import netket as nk
import json
import numpy as np
import flax.linen as nn
import flax

import jax.numpy as jnp
from netket import jax as nkjax
import jax
from jax.tree_util import Partial

from netket.jax import vmap_chunked
from netket.operator import AbstractOperator, DiscreteOperator


import math
from flax.linen.dtypes import promote_dtype

 
def ravel_no_uravel(pytree_ind):
    arr,_ = jax.flatten_util.ravel_pytree(pytree_ind)
    return arr

def vmap_ravel_pytree(pytree_vec):
  return jax.vmap( ravel_no_uravel)(pytree_vec)





#@partial(jax.jit,static_argnames=['params_unravel','nvwf'])
def update_params(J,B,params,opt_vars,params_unravel,nvwf):

  lr,it,lam0,mu,params_arr_grad0 = opt_vars['lr'],opt_vars['it'],opt_vars['lam0'],opt_vars['mu'],opt_vars['params_arr_grad0']
  #lr_it = lr*( 1 / (100+it) )
  lr_it,lam0_it,mu_it = lr,lam0,mu
  G_diag = opt_vars['diag_norm'] #not used should be deleted
  
  
  params_arr_grad = minSR_solve_Kacz(J, B , params_arr_grad0 ,lam0=lam0_it,lam1=1.,mu=mu_it)

  dtheta_max = .05
  dtheta = (lr_it / math.sqrt(nvwf)) * jnp.linalg.norm(jnp.dot(J,params_arr_grad))
  rth = (dtheta_max/dtheta) #also not used but should be used to prevent large grads
 
  params_arr_grad0 = params_arr_grad 
  
  '''
  if shared_weights:
    params_grad = params_unravel(params_arr_grad)  
    params = jax.tree_util.tree_map(lambda pars, grad: pars - lr_it * grad, params,  params_grad )
  else:
    params_arr_grad = params_arr_grad.reshape( nvwf, -1 )
    params_grad = {'ind_params_stacked': jax.vmap(params_unravel)(params_arr_grad)  }
    params = jax.tree_util.tree_map(lambda pars, grad: pars - lr_it * grad, params,  params_grad )
  '''
  params_grad = params_unravel(params_arr_grad)  
  params = jax.tree_util.tree_map(lambda pars, grad: pars - lr_it * grad, params,  params_grad )
  opt_vars = {'params_arr_grad0': params_arr_grad0,'diag_norm':G_diag , 'lr': lr ,'lam0':lam0,'mu':mu, 'it': it+1 }
  
  return params, opt_vars






#Jannes pointed out this can  be obtaind by simpling calculating the jacobian of the log(det(Phi)) which is probablly more stable.
# This calculates the jacobian by taking grad of log(Phi) matrices with Phi^-1 * Phi (elem wise) as the V in vjp
@partial(jax.jit,static_argnames=("vec_apply_fun","chunk_size","center_and_scale"))
def vwf_vec_jacobian(
    vec_apply_fun: Callable,
    vec_params: PyTree,
    samples: Array,
    VJ: Array,
    vec_state: Optional[PyTree] = None,
    *,
    chunk_size: Optional[int] = None,
    center_and_scale: bool = True,
):
  
  f = Partial(lambda W, sigma: vec_apply_fun( {'params': W, **vec_state} , sigma))

 
    
  def jac_fun_single(fun,params,sample,v):
    _, vjp_fun = jax.vjp(lambda pars: fun(pars, sample), params )
    #y,v = promote_dtype(y,v, dtype=None)
    (dp_re_pyt,) = vjp_fun(v)
    (dp_im_pyt,) = vjp_fun( -1.0j * v)
    return ravel_no_uravel( jax.tree_util.tree_map( jax.lax.complex,dp_re_pyt,dp_im_pyt) )
    
  #J = jax.vmap(jac_fun_single , in_axes=(None,None,0,0) ,out_axes=0)( f, vec_params,samples,VJ)
  J = vmap_chunked(jac_fun_single , in_axes=(None,None,0,0) , chunk_size=chunk_size)( f, vec_params,samples,VJ)
 
  if center_and_scale:
    sqrt_n_samp = math.sqrt(samples.shape[0]) 
    return (J - jnp.mean(J,0,keepdims=True))*(1.0 / sqrt_n_samp)
  else:
    return J















#-6 is good

def pinv_smooth(A, b, *, rcond: float = 1e-14, rcond_smooth: float = 1e-6,rcond_power: int = 6):
    
    Σ, U = jnp.linalg.eigh(A)
    Σ_inv = jnp.where(jnp.abs(Σ / Σ[-1]) > rcond, jnp.reciprocal(Σ), 0.0)
    regularizer = 1.0 / (1.0 + (rcond_smooth / jnp.abs(Σ / Σ[-1])) ** rcond_power)
    Σ_inv = Σ_inv * regularizer
    x = U @ (Σ_inv * (U.conj().T @ b))
    return x








#@jax.jit 
def minSR_solve(J,b,lam0=1e-04,lam1=1.):
  
  n=J.shape[0]
  Jh= J.conj().T
  
  A_scale= jnp.linalg.norm(J)**2 /n
  A= jnp.matmul(J , Jh) + A_scale*(lam1 / n)
  #A=A.at[jnp.diag_indices(n)].add(A_scale*lam0)
  A=A.at[jnp.diag_indices(n)].add(lam0)
  
  R=jax.scipy.linalg.cho_factor(A)
  y=jax.scipy.linalg.cho_solve(R,b)
  
    
  #y = pinv_smooth(A,b,rcond_smooth = lam0)
  
  return jnp.dot(Jh,y)
  
 
#@jax.jit 
def minSR_solve_Kacz(J,b,x0,lam0=1e-04,lam1=1.,mu=.99):
  db = b - mu* jnp.dot(J,x0)	
  x=minSR_solve(J,db,lam0=lam0,lam1=lam1)
  return x+mu*x0
