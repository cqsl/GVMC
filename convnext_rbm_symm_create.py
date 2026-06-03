
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



import numpy as np
#import numpy as scipy

import time
from functools import partial

import flax.linen as nn
import flax

import jax.numpy as jnp

import jax
from jax import random
from jax.flatten_util import ravel_pytree
from flax.core import FrozenDict
from jax.tree_util import Partial






from vwf_rbm_custom import freq_mask_hash, RBMModPhase_Mom_Proj_Group_TranSF2



#from convnext_custom import ConvNeXt
from convnext_custom2 import ConvNeXt2
from grass_vwf import Grass_VWF_Vec,Grass_VWF_Vec_BF

# IF you want to try a different model like transformer
'''
def create_model_custom(...,spac_dim):
  # Backflow model / shared DNN eg Convnext, transformer,etc
  BF_Module=partial(DNN,...)
  # Heads / Individual SNNs eg RBM, Jastrow,etc
  Out_Module= partial(SNN,...)
  model_vec = Grass_VWF_Vec_BF(BF_Module,Out_Module,nvwf,spac_dim)
  return model_vec
'''



def create_model(meta_values):

  spac_dim,group_dim,k_vec = meta_values['spac_dim'],meta_values['group_dim'],meta_values['k_vec']
  
  stage_sizes,features_in,stage_features = meta_values['stage_sizes'],meta_values['features_in'],meta_values['stage_features']
  kernel_width,kernel_width_in = meta_values['kernel_width'],meta_values['kernel_width_in']
  sign_symm,sign_symm_pool = meta_values['sign_symm'],meta_values['sign_symm_pool']

  RBM_features_out,RBM_use_bias = meta_values['RBM_features_out'],meta_values['RBM_use_bias']

  nvwf = meta_values['nvwf']

  BF_Module=partial(ConvNeXt2,
    spac_dim,
    stage_sizes,
    features_in,
    stage_features,
    kernel_width_in = kernel_width_in,
    kernel_width = kernel_width,
    global_pool = False,
    sign_symm = sign_symm,
    sign_symm_pool = sign_symm_pool,
  )


  freq_mask = freq_mask_hash(group_dim, [ np.array( k_vec ) ] )
  Out_Module= partial(RBMModPhase_Mom_Proj_Group_TranSF2,spac_dim,RBM_features_out,use_hidden_bias = RBM_use_bias,psi_std_init=.02,mask = freq_mask)
  model_vec = Grass_VWF_Vec_BF(BF_Module,Out_Module,nvwf,spac_dim)



  return model_vec



























