import os


os.environ["NETKET_EXPERIMENTAL_SHARDING"] = "1"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".25"


# I dont need most of these libs
import netket as nk
import functools
from typing import Optional


from typing import Union, Any,Callable, Sequence,Dict,Tuple
from dataclasses import dataclass, field


from netket.utils import (
    maybe_wrap_module,
    mpi,
    wrap_afun,
    wrap_to_support_scalar,
)
from netket import jax as nkjax

import json

import numpy as np
#import numpy as scipy
import scipy

from time import time
from functools import partial

import flax.linen as nn
import flax

import jax.numpy as jnp

import jax
from jax import random
from jax.flatten_util import ravel_pytree
from flax.core import FrozenDict
from jax.tree_util import Partial

from tqdm import tqdm
from jax import config
from itertools import chain

# The determiant sampling code: custuom sampler and proposal classes
from grass_samp_new import MetropolisSampler_Wedge,MetropolisExchange_Wedge
from grass_samp_rule_new import SymmetryProposal_2D_Square_SpinHalf

# This the core function that does everything for each weight update
from excited_state_calc import excited_update_step

# These create jax hams. They are very slow you can replace with your own.
from ham_jax_construct import Ham_Heisenberg_J1_pcl,Ham_Heisenberg_J1_J2_pcl

#from symm_test import symm_test,symm_test_2D_sf 
#this is just a function that checks the VWF has the righ symmetries w/ phases
#from symm_test import symm_test,symm_test_2D_sf


# Im an idiot and I never imported this file from the cluster so I have to rewrite this
# This just code which makes the models more simply.
from convnext_rbm_symm_create import create_model,create_model2

import math






seed= 2929424
key_use, key_split = random.split(random.key(seed))


n_dim=2
L=6 #square lattice side lengths
n_sites  = L**n_dim
spac_dim = ( (L,) *n_dim )

# I only wrote code for translational symmetry + spin flip
group_dim = spac_dim + (2,)
n_group = np.prod(group_dim)

#this only needed for vwf model to enforce the right symmetry phases
k_vec = (0,0,0) #integer momentum vec momentum is 2pi/L*(k1,k2) and k3=0 or 1 for +/- spin flip phases
pi2=6.28318530718
phases =( np.exp(0.0 + 1j *0.0)  ,  np.exp( 1j *pi2*(k_vec[0] / L)) , np.exp( 1j *pi2*(k_vec[1] / L)) , np.exp( 1j *pi2*(k_vec[2] / 2)) )


nvwf = 1 # number indivdual vwfs


graph = nk.graph.Hypercube(length=L, n_dim=n_dim, pbc=True)
hilb = nk.hilbert.Spin(s=0.5, total_sz=0, N=graph.n_nodes)
#ham = nk.operator.Heisenberg(hilbert=hilb, graph=graph,J=(.25 /n_sites),sign_rule=True)

#ham is normalized by .25 * 1/n_sites so gs energy should be 1D: -.43 and 2D: -.67
#msr: Marshal sign rule
ham = Ham_Heisenberg_J1_pcl(hilb,spac_dim,msr=True) 
#ham = Ham_Heisenberg_J1_J2_pcl(hilb,spac_dim,J2=.25*.50,msr=False)



load_params = False # True: load params and False: random params
load_sampler = False
save_params_and_sampler = False

# choose base name of load and save files then adds system size, momentum, num of vwfs
#file_name_save0 ="RBM_Symm_Model_0_Heis_Excited_States"
file_name_save0 ="test"
file_name_save = file_name_save0 + "_L_" + "_".join(  [ f"{n}" for n in spac_dim ] )
file_name_save = file_name_save + "_k_" + "_".join(  [ f"{n}" for n in k_vec ] ) + f"_nvwf_{nvwf}"
#print(file_name_save)
file_name_load = file_name_save



lr0=0.10 #learning rate /step size (Ham is normalized by 1/n_sites so not that big )
lam0=1e-3 # QGT matrix shift /regularization (forgot why it is lam0 not just lam but there is no lam anymore)
#lam0= 0.0
mu=.9 #SPRING decay/momentum term (settting to 0 gives minSR)( can be increased to upto about .99 may help or make worse)
nch=2**10 #num of chanins
chl=2**2 #chain length / samps per chain
nsamp=nch*chl # total samps
iterations=8000 #num of total weight updates that will be executed
#iterations=0



#6x6 model 0
chunk_size_energy=2**8 #chunk size for calculating local energy matrices (one nvwfxnvwf matrix =1 chunk)
chunk_size_derv=2**10 #chunk size for calculating jacobian (cant remember how one chunk is defined but not as much of a mem. issue)

#10x10 model 0
'''
chunk_size_energy=2**8
chunk_size_derv=2**9
'''

chunk_sizes = (chunk_size_energy,chunk_size_derv)





mode = "complex"
param_dtype = np.float64

# 6x6 model_0 beta=.9,lr=.15,lam=10-3,psi_std=.02
'''
meta_values={ 'spac_dim': spac_dim , 'group_dim': group_dim , 'k_vec': k_vec, 'nvwf': nvwf,
             'stage_sizes': [2,2,8],'features_in':4,'stage_features':[4,6,12],
             'kernel_width_in': 3,'kernel_width': 5,
             'sign_symm': True, 'sign_symm_pool':False,
             'RBM_features_out': 16, 'RBM_use_bias': True,
            }

'''

# 10x10 model_0 beta=.9,lr=.15,lam=10-3,psi_std=.02
'''
meta_values={ 'spac_dim': spac_dim , 'group_dim': group_dim , 'k_vec': k_vec, 'nvwf': nvwf,
             'stage_sizes': [2,2,8],'features_in':6,'stage_features':[8,12,20],
             'kernel_width_in': 3,'kernel_width': 5,
             'sign_symm': True, 'sign_symm_pool':False,
             'RBM_features_out': 12, 'RBM_use_bias': True,
            }

'''


# 6x6 model_test beta=.9,lr=.15,lam=10-3,psi_std=.02

meta_values={ 'spac_dim': spac_dim , 'group_dim': group_dim , 'k_vec': k_vec, 'nvwf': nvwf,
             'stage_sizes': [8],'features_in':4,'stage_features':[30],
             'kernel_width_in': 3,'kernel_width': 6,
             'sign_symm': True, 'sign_symm_pool':False,
             'RBM_features_out': 10, 'RBM_use_bias': True,
            }



def pyt_shape(tree):
  return jax.tree_util.tree_map(lambda x: x.shape, tree)

def count_params(params):
	n_leaves = jax.tree_util.tree_leaves( jax.tree_util.tree_map(lambda x: x.size, params) )
	return sum(n_leaves)



#creates a backflow model with shared convnext base and indvidual custom symmetry rbm heads
#model_vec = create_model(meta_values)
model_vec = create_model2(meta_values)


x = random.normal(key_use , (nsamp,n_sites) )
key_use, key_split = random.split(key_split)
vec_vars = model_vec.init(key_use, x)
key_use, key_split = random.split(key_split) 
vec_params_array,vec_params_unravel= ravel_pytree(vec_vars['params'])
vec_state, vec_params = flax.core.pop(vec_vars, 'params') 
print(pyt_shape(vec_vars['params']))
nparam = count_params(vec_params)
#print(nparam)
y=model_vec.apply({'params': vec_params,**vec_state},x)
print(f" out_shape {y.shape}")


nparam_bf = count_params(vec_params['bf_params'])
nparam_ind = count_params(vec_params['ind_params'])
print(f'params Tot: {nparam} BF: {nparam_bf} Ind: {nparam_ind} ')

#mod_apply_fn_vec = jax.jit(lambda p,x0,**kwargs: model_vec.apply(p,x0,**kwargs))
#mod_apply_fn_vec= jax.jit(model_vec.apply)
mod_apply_fn_vec= model_vec.apply
params_unravel = vec_params_unravel


opt_vars = {'params_arr_grad0': np.zeros((nparam,) ,dtype = param_dtype),'diag_norm': np.zeros((nparam,)) , 'lr': lr0 , 'lam0':lam0,'mu':mu, 'it': 0 }
grass_vars = {'E_avg':0.+0.j, 'E_var': 0. , 'H': np.zeros((nvwf,nvwf),dtype=np.complex128) ,'H_var': np.zeros((nvwf,nvwf),dtype=np.complex128) }




symm_prop_prob=.1 # prob of proposed state being a symmetry transformation of the original state
# This just a a function that transforms spin spin states by symmetries
symm_prop = SymmetryProposal_2D_Square_SpinHalf(spac_dim,symm_prop_prob)

#creates the sampler
sampler = MetropolisExchange_Wedge(
  hilb,
  nvwf,
  graph=graph,
  d_max=6,
  symm_prop=symm_prop,
  n_chains = nch,
  sweep_size = n_sites*nvwf,
  #sweep_size = 1,
)

#seed =1777
#seed  = key_use
key_use, key_split = random.split(key_split)

sampler_state = sampler.init_state(mod_apply_fn_vec, {'params': vec_params,**vec_state} )
sampler_state = sampler.reset(mod_apply_fn_vec, {'params': vec_params,**vec_state} , sampler_state)



if load_params:
  save_values = { 'params':vec_params , 'state': vec_state, 'opt':opt_vars,'grass':grass_vars }

  with open(file_name_save+"_values.mpack", 'rb') as file:
    save_values = flax.serialization.from_bytes(save_values, file.read())

  vec_params,vec_state,opt_vars,grass_vars  = save_values['params'],save_values['state'],save_values['opt'],save_values['grass'],

if load_sampler:
  with open(file_name_load+"_sampler.mpack", 'rb') as file:
    sampler_state = flax.serialization.from_bytes(sampler_state, file.read())


print("made it to here0")

def sampler_warmup(sampler_state , n_warmup):

  def samper_warmup_for(i,sampler_state):
    sampler_state = sampler.reset(mod_apply_fn_vec, {'params': vec_params,**vec_state} , sampler_state)
    _, sampler_state = sampler.sample(mod_apply_fn_vec, {'params': vec_params,**vec_state} , state=sampler_state, chain_length=chl)
    return sampler_state

  return jax.lax.fori_loop(0, n_warmup, samper_warmup_for , sampler_state )

sampler_state = sampler.reset(mod_apply_fn_vec, {'params': vec_params,**vec_state} , sampler_state)
samples, sampler_state = sampler.sample(mod_apply_fn_vec, {'params': vec_params,**vec_state} , state=sampler_state, chain_length=chl)



arr_dim = (nsamp,nvwf,nparam)
#print("made it to here1")

default_string = f"r{jax.process_index()}/{jax.process_count()}: "
print(default_string, jax.devices(), flush=True)
print(default_string, jax.local_devices(), flush=True)
print("---------------------------------------------\n", flush=True)
print(samples.sharding)


key_use, key_split = random.split(key_split)



E_ravg0,E_rvar0 = 0.0,0.0
beta=.99
for it in range(iterations):
	it_time = time()
	
	
	sampler_state = sampler.reset(mod_apply_fn_vec, {'params': vec_params,**vec_state} , sampler_state)
	samples, sampler_state = sampler.sample(mod_apply_fn_vec, {'params': vec_params,**vec_state} , state=sampler_state, chain_length=chl)
	sigma = samples.reshape( (nsamp,nvwf,-1))
	
	key_use, key_split = random.split(key_split)
	#vec_params,vec_state,opt_vars,grass_vars =excited_update_step(key_use,mod_apply_fn_vec,mod_apply_fn_ind,params_unravel, sigma,ham,chunk_sizes, vec_params,vec_state,opt_vars,grass_vars,arr_dim,mode,shared_weights,sum_batch)
	vec_params,vec_state,opt_vars,grass_vars =excited_update_step(mod_apply_fn_vec,params_unravel, sigma,ham,chunk_sizes, vec_params,vec_state,opt_vars,grass_vars,mode)
	
	
	save_values = { 'params':vec_params , 'state': vec_state, 'opt':opt_vars,'grass':grass_vars }
	if save_params_and_sampler and it%10 == 0:
	  with open(file_name_save+"_values.mpack", 'wb') as file:
	    file.write(flax.serialization.to_bytes(save_values))
	  with open(file_name_save+"_sampler.mpack", 'wb') as file:
	    file.write(flax.serialization.to_bytes(sampler_state))
	
	E_avg,E_var,H,H_var = grass_vars['E_avg'],grass_vars['E_var'],grass_vars['H'],grass_vars['H_var']
	E_avg,E_std = np.real(E_avg) , np.sqrt(E_var)
	E_eig,X = np.linalg.eig(H)
	E_min,E_max =np.min(np.real(E_eig)) , jnp.max(np.real(E_eig))
	E_eig = tuple(np.sort(np.real(E_eig)))
	
	E_ravg0 = beta*E_ravg0 +  (1.0 - beta)*E_avg
	E_ravg = E_ravg0 /(1.0 - beta**(it+1))
	E_rvar0 = beta*E_rvar0 +  (1.0 - beta)*E_var
	E_rvar = E_rvar0 /(1.0 - beta**(it+1))
	E_rstd = math.sqrt(E_rvar)
	
	
	pln = vec_state['norm']['log_norm']
	pln_min,pln_max = np.min(pln),np.max(pln)
	avg_dparam=np.linalg.norm(opt_vars['params_arr_grad0']) / math.sqrt(nparam)
	J_norm  = np.mean(opt_vars['diag_norm'])
	
	it_time = time()-it_time
	if it%50 == 0:
		print(f'it: {it} (E_avg,E_std): ({E_avg:.6f},{E_std:.6f}) (E_ravg,E_rstd): ({E_ravg:.8f},{E_rstd:.6f})   mdp: {avg_dparam} acc: {100.0  *sampler_state.acceptance: .1f}%   pln: ({pln_min},{pln_max}) J_norm:{J_norm} time:{it_time}',flush=True)
		print( E_eig)
