
from functools import partial
from typing import Any, Callable, Optional, Union
from textwrap import dedent


#import netket as nk
#import json

import numpy as np

import jax.numpy as jnp
#from netket import jax as nkjax
import jax

#import flax
from flax import linen as nn
from flax import serialization


from netket.hilbert import AbstractHilbert, ContinuousHilbert

from netket.utils import mpi, wrap_afun
from netket.utils.types import PyTree, DType,Array,Callable

from netket.utils.deprecation import deprecated, warn_deprecation
from netket.utils import struct

from netket.utils.config_flags import config
from netket.jax.sharding import (
    extract_replicated,
    gather,
    distribute_to_devices_along_axis,
    device_count_per_rank,
)

from netket.sampler.base import Sampler, SamplerState
from netket.sampler.rules import MetropolisRule

#import netket.sampler.metropolis as met
from netket.sampler.metropolis import MetropolisSampler
#from netket.sampler.metropolis import _assert_good_sample_shape,_assert_good_log_prob_shape

#from grass_vwf import overlap_mat,det_and_inv_row_update_batch,row_norm_func
from grass_vwf import overlap_mat,det_and_inv_row_update_batch,row_norm_func


class MetropolisSampler_Wedge(MetropolisSampler):


    @partial(jax.jit, static_argnums=(1, 4))
    #@partial(jax.jit, static_argnames=['model','chain_length'])
    def _sample_chain(sampler, model, variables, state, chain_length):
        
       
        #M=sampler.n_chains_per_rank
        M=sampler.n_chains
        N=sampler.hilbert._n_hilbert_spaces
        L=sampler.hilbert._sizes[0]
        
        def loop_body(sweep_ind, carry):
          
            σ, A_inv,row_log_norm, accepted, key0 = carry
            key0, key1, key2 = jax.random.split(key0, 3)
            
            
            σp_I, I, log_prob_correction = sampler.rule.transition(
                sampler, model, variables, state, key1, σ
            )
            
            Ap_row_I = model.apply(variables,σp_I)
            #prop_row_log_norm_I = jnp.max( jnp.real(Ap_row_I),axis=-1 )
            prop_row_log_norm_I = row_norm_func(Ap_row_I)
            Ap_row_I = jnp.exp( Ap_row_I - jnp.expand_dims(prop_row_log_norm_I, axis=-1) )
            det_ratio , Ap_inv = det_and_inv_row_update_batch( A_inv , Ap_row_I, I)
            
            
        
            #σp = (  ( σ.reshape((M,N,L)) ).at[ jnp.arange(M) , I , : ].set(σp_I) ).reshape( (M,N*L) )
            #prop_row_log_norm = row_log_norm.at[ jnp.arange(M) , I  ].set(prop_row_log_norm_I)
            σp =jax.vmap(jax.lax.dynamic_update_slice)(σ,σp_I,L*jnp.expand_dims(I,-1))
            prop_row_log_norm =jax.vmap(jax.lax.dynamic_update_slice)(row_log_norm,jnp.expand_dims(prop_row_log_norm_I,-1),jnp.expand_dims(I,-1))
            
            
            #log_prob_ratio = 2. * ( jnp.log(jnp.real(det_ratio)) + prop_row_log_norm_I - row_log_norm.at[ jnp.arange(M) , I ].get() )
            log_prob_ratio = 2. * ( jnp.real(jnp.log(det_ratio)) + jnp.sum(prop_row_log_norm,axis=-1) - jnp.sum(row_log_norm,axis=-1))
           
            log_uniform = jnp.log( jax.random.uniform(key2, shape=(sampler.n_batches,)) )
            
            if log_prob_correction is not None:
                do_accept = log_uniform < log_prob_ratio + log_prob_correction
            else:
                do_accept = log_uniform < log_prob_ratio


            σ = jnp.where(do_accept.reshape(M,1) , σp , σ )
            A_inv = jnp.where(do_accept.reshape(M,1,1) , Ap_inv, A_inv )
            row_log_norm = jnp.where(do_accept.reshape(M,1) , prop_row_log_norm , row_log_norm )
            accepted += do_accept
            
            return (σ, A_inv,row_log_norm, accepted, key0)
            
        
        def scan_fun(carry,_):
          carry = jax.lax.fori_loop(0, sampler.sweep_size, loop_body, carry)
          return carry, carry[0]
        
        #σ, A_inv,row_log_norm, accepted, key0 = carry
        σ=state.σ
        #A,row_log_norm = model.apply(variables, σ.reshape((M,N,L)), method = model.overlap_mat )
        #A,row_log_norm =overlap_mat( jax.vmap( lambda sig: model.apply(variables,sig),in_axes=1,out_axes=1)( σ.reshape((M,N,L))) )
        A,row_log_norm =overlap_mat(  model.apply(variables , σ.reshape((M,N,L)) ) )
        A_inv=jnp.linalg.inv(A)
        accepted = state.n_accepted_proc
        key1, key0 = jax.random.split(state.rng)
        carry =(σ,A_inv,row_log_norm,accepted, key0)
        
        carry, samples = jax.lax.scan(
            scan_fun,
            carry,
            xs=None,
            length=chain_length,
        )
        # make it (n_chains, n_samples_per_chain) as expected by netket.stats.statistics
        samples = jnp.swapaxes(samples, 0, 1)
        
        state = state.replace(
            rng=key1,
            σ=carry[0],
            n_accepted_proc=carry[3],
            n_steps_proc=state.n_steps_proc
            + sampler.sweep_size * sampler.n_batches * chain_length,
        )
        return samples, state    
            
            
           
def MetropolisExchange_Wedge(
    hilbert_ind,n_wedge, *, symm_prop=None,clusters=None, graph=None, d_max=1, **kwargs
) -> MetropolisSampler_Wedge:
    
    from netket.sampler.rules import ExchangeRule
    from grass_samp_rule import WedgeRule
    rule_ind = ExchangeRule(clusters=clusters, graph=graph, d_max=d_max)
    rule = WedgeRule( hilbert_ind, n_wedge,rule_ind,symm_prop)
    return MetropolisSampler_Wedge(rule.hilbert, rule, **kwargs)           
            
            
def MetropolisLocal_Wedge(
    hilbert_ind,n_wedge,*, symm_prop=None , **kwargs
) -> MetropolisSampler_Wedge:
    
    from netket.sampler.rules import LocalRule
    from grass_samp_rule import WedgeRule
    rule_ind = LocalRule()
    rule = WedgeRule( hilbert_ind, n_wedge,rule_ind)
    return MetropolisSampler_Wedge(rule.hilbert, rule, **kwargs)             
            
