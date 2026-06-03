from functools import partial
from typing import Any, Optional,Tuple,Callable,Union

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from flax import linen as nn

from netket import config
from netket.hilbert import TensorHilbert,AbstractHilbert
from netket.utils.types import PyTree, PRNGKeyT
from netket.utils import struct
# Necessary for the type annotation to work
if config.netket_sphinx_build:
    from netket import sampler

from netket.sampler.rules.base import MetropolisRule
 



#These symmetry proposoal classes are a mess and needs to be redone. All they do is apply random translations, rotatiions etc for each input
#THis is essentially a base class but does absolutely nothing (evrything is  as I only wrote a class for 2D square lattices for spins specfically where everything is rewritten 
class SymmetryProposal(struct.Pytree):
  spac_dim: Tuple[int] = struct.field(pytree_node=False)
  group_dim: Tuple[int] = struct.field(pytree_node=False)
  n_group: int = struct.field(pytree_node=False)
  #symm_fun: Callable
  acc_prob: float = struct.field(pytree_node=False)
  def __init__(
        self,
        spac_dim: Tuple[int],
        #group_dim: Tuple[int],
        #n_group: int,
        #symm_fun: Callable,
        acc_prob: float = 0.0,
  ):
       
    self.spac_dim = spac_dim
    self.group_dim = self.spac_dim +(4,2,2)
    self.n_group = np.prod(self.spac_dim)*16
    #self.symm_fun = partial(symm_fun,spac_dim,group_dim)
    self.acc_prob = acc_prob

    
  def transition(self,key,inputs):
      return inputs
      #return spins


#This what is used but is hyper specfic 
class SymmetryProposal_2D_Square_SpinHalf(SymmetryProposal):
  spac_dim: Tuple[int] = struct.field(pytree_node=False)
  group_dim: Tuple[int] = struct.field(pytree_node=False)
  n_group: int = struct.field(pytree_node=False)
  #symm_fun: Callable
  acc_prob: float = struct.field(pytree_node=False)
  def __init__(
        self,
        spac_dim: Tuple[int],
        #group_dim: Tuple[int],
        #n_group: int,
        #symm_fun: Callable,
        acc_prob: float = 0.0,
  ):
       
    self.spac_dim = spac_dim
    self.group_dim = self.spac_dim +(4,2,2)
    self.n_group = np.prod(self.spac_dim)*16
    #self.symm_fun = partial(symm_fun,spac_dim,group_dim)
    self.acc_prob = acc_prob

    
  def transition(self,key,inputs):
      n_samp = inputs.shape[0]

      key_group,key_acc = jax.random.split(key, 2)

      group_ind = jax.random.randint(key_group, shape=(n_samp,), minval=0, maxval=self.n_group)
      g_ind_vec = jnp.stack( jnp.unravel_index(group_ind,self.group_dim) , axis=-1)

  
      spins = jax.vmap( jnp.roll)( inputs.reshape(n_samp,*self.spac_dim) , g_ind_vec[:,0:2] )
      spins = jax.vmap(lambda s,nr: jax.lax.cond(nr == 1,lambda x: jnp.rot90(x,k=1) ,lambda x: x,s))(spins,g_ind_vec[:,2] )
      spins = jax.vmap(lambda s,nr: jax.lax.cond(nr == 2,lambda x: jnp.rot90(x,k=2) ,lambda x: x,s))(spins,g_ind_vec[:,2] )
      spins = jax.vmap(lambda s,nr: jax.lax.cond(nr == 3,lambda x: jnp.rot90(x,k=3) ,lambda x: x,s))(spins,g_ind_vec[:,2] )
      spins = jax.vmap(lambda s,nr: jax.lax.cond(nr,jnp.flipud,lambda x: x,s))(spins,g_ind_vec[:,3] )
      spins = jax.vmap(lambda s,nf: jax.lax.cond(nf,lambda x: -x,lambda x: x,s))(spins,g_ind_vec[:,4] )
  

      spins = spins.reshape(n_samp,-1)
      symm_prob = jax.random.uniform(key_acc, shape=(n_samp,)) 
      symm_accept = symm_prob < self.acc_prob
  
  
      return jnp.where(symm_accept.reshape(n_samp,1) , spins , inputs )
      #return spins
    
    













#MetropolisRule for wedge product state/ determiant state sampling as special case of TensorHilbert 
# For chosen input "rule" as a MetropolisRule defined over a invidual/single hilbert space "hilbert_ind" 
# this class then creates a rule over nvwf Hilbert tensor prod. copies "hilbert" = hilbert_ind**nvwf
# So each sample of hilbert is a batch nvwf different samples from hilbert_ind
# Each proposal given by picking one of the current nvwf samples at random and applying "rule" on that one chosen sample
#The reamaining nvwf-1 samples are unchanged
# symm_prop is a mess and is optional but adds random symmetry transformations to the proposals on top of what rule does

class WedgeRule(MetropolisRule):
  
    hilbert_ind: AbstractHilbert = struct.field(pytree_node=False) #invidual/single hilbert space
    hilbert: TensorHilbert = struct.field(pytree_node=False) #hilbert = hilbert_ind**nvwf hilberst space for this class
    rule: MetropolisRule # chosen rule (Local,Exchange,etc.) defined over  hilbert_ind
    #symm_prop: Union[Callable , None] = None
    symm_prop: Union[SymmetryProposal , None] = None # a mess of class that applies random symmetry transformations on top of rule
    
    
   

    def __init__(
        self,  hilbert_ind: AbstractHilbert,n_wedge: int, rule: MetropolisRule, symm_prop: Union[Callable , None] = None
    ) -> "TensorRule":
      
        
        self.hilbert_ind = hilbert_ind
        self.hilbert =  TensorHilbert( * (hilbert_ind,)*n_wedge) # n_wedge = n_vwf as the number of hilbert copies
        self.rule=rule
        self.symm_prop = symm_prop
      

    
    def init_state(
        self,
        sampler: "sampler.MetropolisSampler",  # noqa: F821
        machine: nn.Module,
        params: PyTree,
        key: PRNGKeyT,
    ) -> Optional[Any]:
    
        return self.rule.init_state(sampler.replace(hilbert=self.hilbert_ind),machine,params,key)
            
        
        
    def reset(
        self,
        sampler: "sampler.MetropolisSampler",  # noqa: F821
        machine: nn.Module,
        params: PyTree,
        sampler_state: "sampler.SamplerState",  # noqa: F821
    ) -> Optional[Any]:
        
        return self.rule.reset(sampler.replace(hilbert=self.hilbert_ind), machine, params, sampler_state)
        

    def transition(self, sampler, machine, parameters, state, key, σ):

        # σ is MxNxL (these vars should be changed to more clear)
        M=σ.shape[0] #n_chains
        N=self.hilbert._n_hilbert_spaces # = n_wedge = n_vwf
        L=self.hilbert_ind.size # n_sites

        # keys for picking which state indices to update, key for rule.transition, and key for symm prop respectively
        key_I,key_tran,key_symm = jax.random.split(key, 3)
        
        #for each chain uniformly picks one of the N=n_wedge=nvwf basis state to update as I as random index from 1 to N for each chain
        I = jax.random.randint(
                key_I, shape=(σ.shape[0],), minval=0, maxval=N
        )

        #isolates only the states which are being updated which is fed into to self.rule.transition 
        #giving σp_I as an proposed states same as if sampling a single hilb space
        σp_I = jax.vmap( lambda sig, i: jax.lax.dynamic_slice_in_dim(sig,L*i,L))(σ , I)
        σp_I, log_prob_corr = self.rule.transition(
                sampler.replace(hilbert=self.hilbert_ind), machine, parameters, state, key_tran, σp_I
        )

        #applies random symmetry transformations to op_I (so double updates not seperate)
        if self.symm_prop is not None:
            σp_I = self.symm_prop.transition(key_symm,σp_I)
        
        return σp_I, I, log_prob_corr 
