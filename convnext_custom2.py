from functools import partial
from typing import Union, Any,Callable, Sequence, Tuple,Optional

import numpy as np
import jax
from jax import numpy as jnp
from flax import linen as nn
from netket.nn.activation import relu
from jax.nn.initializers import normal
from jax import lax



from netket.utils import HashableArray
from netket.utils.types import NNInitFunc
from netket.utils.group import PermutationGroup
from netket import nn as nknn

from netket.utils.types import DType


defualt_dtype = np.float64
#default_kernel_init = normal(stddev=0.01)
default_kernel_init = nn.initializers.variance_scaling(
    0.2, "fan_in", distribution="truncated_normal",dtype=defualt_dtype
)



default_dconv_init = default_kernel_init
default_pwconv_init = default_kernel_init
ModuleDef = Any


from depth_conv_custom import Depth_Conv_FFT


class Depth_Conv(nn.Module):
  features: int
  kernel_size: tuple
  use_bias: bool = False
  #dtype: Any = defualt_dtype
  param_dtype: Any = defualt_dtype
  @nn.compact
  def __call__(self,inputs):
    return nn.Conv(self.features, self.kernel_size, padding="CIRCULAR",
        feature_group_count=self.features, param_dtype=self.param_dtype,
        use_bias=self.use_bias, kernel_init=default_dconv_init, name="dConv",
        )(inputs) 





class GRN(nn.Module):
    """GRN (Global Response Normalization) layer

    Transposed from the Pytorch implementation
    https://github.com/facebookresearch/ConvNeXt-V2/blob/2553895753323c6fe0b2bf390683f5ea358a42b9/models/utils.py#L105
    """

    #feature_axis: int = -1
    #normalise_axes: tuple = (-2, -3)
    #param_dtype: DType = defualt_dtype
    param_dtype: Any = defualt_dtype

    @nn.compact
    def __call__(self, x):
        features = x.shape[-1]
        spac_dim = x.shape[1:-1]
        n_sites = np.prod(spac_dim)
        gamma_shape = tuple(1 for _ in range(x.ndim-2)) 
        
        gamma = self.param(
            "gamma", nn.initializers.zeros, (features,), self.param_dtype
        )
        beta = self.param(
            "beta", nn.initializers.zeros, (features,), self.param_dtype
        )
        '''
        gamma_shape = list(1 for _ in range(x.ndim))
        gamma_shape[self.feature_axis] = -1
        gamma = gamma.reshape(tuple(gamma_shape))
        beta = beta.reshape(tuple(gamma_shape))
        '''
        #normalise_axes = tuple(range(1,x.ndim-1))
        # Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        #Gx = jnp.linalg.norm(x, ord="fro", axis=normalise_axes, keepdims=True)
        Gx = jnp.linalg.norm(x.reshape(-1,n_sites,features), axis=1, keepdims=True).reshape(-1,*gamma_shape,features)
        Nx = Gx / (jnp.mean(Gx, axis=-1, keepdims=True) + 1e-6)
        return gamma * (x * Nx) + beta + x











class ConvNeXtBlock(nn.Module):
  features: int
  dconv: ModuleDef
  norm: ModuleDef
  act: Callable
  feature_multiplier: int = 4
  layer_scale_init_value: float = -1e-4
  #layer_scale_init_value: float = 1e-8
  def init_fn(self, key, shape, fill_value):
        return np.full(shape, fill_value,dtype =defualt_dtype )
  
  
  @nn.compact
  def __call__(self,inputs):
  
    x = self.dconv(self.features)(inputs) 
    x = self.norm()(x)
    #x = nn.LayerNorm(name="norm")(x)
    x = nn.Dense(self.feature_multiplier * self.features,  kernel_init=default_pwconv_init, param_dtype=defualt_dtype, name="pwconv0")(x)
    x = self.act(x)
    x = GRN()(x)
    x = nn.Dense(self.features , kernel_init=default_pwconv_init,param_dtype=defualt_dtype, name="pwconv1")(x)

    
    if self.layer_scale_init_value > 0:
      gamma = self.param(
        "gamma", self.init_fn, (self.features,), self.layer_scale_init_value
      )
      x = gamma * x
    
    
    return x + inputs
    
 
class ConvNeXtStage(nn.Module):
  n_blocks: int
  features_out: int
  dconv: ModuleDef
  norm: ModuleDef
  act: Callable
  feature_multiplier: int = 4
  
  
  @nn.compact
  def __call__(self,inputs):
    x=inputs
    #x = self.norm()(inputs)
    #x = nn.LayerNorm(name="norm_stage")(x)
    x = nn.Dense(self.features_out , kernel_init=default_pwconv_init,param_dtype=defualt_dtype, name="pwconv_fm")(x)
    for i in range(self.n_blocks):
      x = ConvNeXtBlock(self.features_out,self.dconv,self.norm,self.act,self.feature_multiplier)(x) 
    return x 

    
   


class ConvNeXt2(nn.Module):
  spac_dim: Tuple[int]
  #spac_dim0: Sequence[int]
  stage_sizes: Sequence[int] #list of number of resblocks for each stage
  features_in: int #features after intial conv
  stage_features: Sequence[int] #list of features after each stage
  #features_out: int 
  kernel_width_in: int =3 #intial conv kernel width
  kernel_width: int =7  #each resblock depth wise conv kernel width
  
  #dtype: Any = defualt_dtype
  param_dtype: Any = defualt_dtype
  #act: Callable = nn.relu
  act: Callable = nn.gelu
  global_pool: bool =False #whether to do a global pool of the output
  sign_symm: bool =False # whether to enforse sign symmetry. If true for all inputs x,  f(-x) =-f(x)
  sign_symm_pool: bool = True # 
  

  def setup(self):
    self.sp_dims = len(self.spac_dim)
    
    if self.sign_symm:
      #self.sp_dims_eff=self.sp_dims+1
      self.kernel_size_in = ( (self.kernel_width_in,) *self.sp_dims ) +(1,)
      self.dconv_kernel_size = ( (self.kernel_width,) *self.sp_dims ) +(2,)
      if self.sign_symm_pool:
        self.spac_dimf = self.spac_dim
      else:
        self.spac_dimf = self.spac_dim +(2,)
    else:
      #self.sp_dims_eff=self.sp_dims
      self.kernel_size_in = ( (self.kernel_width_in,) *self.sp_dims )
      self.dconv_kernel_size = ( (self.kernel_width,) *self.sp_dims )
      self.spac_dimf = self.spac_dim
   


  @nn.compact
  def __call__(self, inputs):
  
    dconv = partial(Depth_Conv, kernel_size= self.dconv_kernel_size,param_dtype=self.param_dtype)
    #dconv = partial(Depth_Conv_FFT,self.spac_dimf, use_bias=False, real_in_out = True  ,  param_dtype=self.param_dtype)

      
    norm = partial( nn.LayerNorm,
        epsilon=1e-6,
        param_dtype=self.param_dtype,
        #reduction_axes =(1,2),
    )
    batch_dim = inputs.shape[: -(self.sp_dims+1)]
    features_in = inputs.shape[-1]
    #spac_dim = inputs.shape[ -(self.sp_dims+1):-1]
    
    x=inputs.reshape(-1,*self.spac_dim,features_in) 
    
    if self.sign_symm:
      x = jnp.stack((x,-x),axis=-2)
     
 
    
    
    x = nn.Conv(
            self.features_in, self.kernel_size_in, padding="CIRCULAR",kernel_init=default_kernel_init,param_dtype=defualt_dtype, name="conv_init"
        )(x)
    #x = norm()(x) 
    
    
    for i, (block_size,block_features) in enumerate(zip(self.stage_sizes,self.stage_features)):
      x=ConvNeXtStage(block_size, block_features,dconv,norm,self.act)(x)
    
    '''
    if self.sign_symm:
      x = jax.lax.index_in_dim(x,0,axis=-2,keepdims=False)-jax.lax.index_in_dim(x,1,axis=-2,keepdims=False)
    '''
    if self.sign_symm and self.sign_symm_pool:
      x = jax.lax.index_in_dim(x,0,axis=-2,keepdims=False)-jax.lax.index_in_dim(x,1,axis=-2,keepdims=False)
    
    if self.global_pool:
      x = jnp.mean(x, axis=tuple(range(1,self.sp_dims+1)),keepdims=True )
      x = norm(name="bn_final",use_scale=False, use_bias=False )(x) 
      x = x.reshape(*batch_dim,x.shape[-1])
    else:
      x = norm(name="bn_final",use_scale=False, use_bias=False )(x)  
      #x = x.reshape(*batch_dim,*spac_dim,x.shape[-1])
      #x = x.reshape(*batch_dim,*x.shape[1:])
      x = x.reshape(*batch_dim,*self.spac_dimf,x.shape[-1])
    #x = norm(name="bn_final",use_scale=False, use_bias=False )(x) 
    #x = nn.LayerNorm(use_scale=False, use_bias=False,epsilon=1e-6,param_dtype=self.param_dtype,name="bn_final")(x)
    #x = x.reshape(*batch_dim,*x.shape[1:])
    #x = jnp.asarray(x, self.dtype)
    return x


#convnext_example_1D = partial(ConvNeXt, sp_dims =1, stage_sizes=[4,4,2],  features_in=4,stage_features=[4,8,16])  
#convnext_example_2D = partial(ConvNeXt, sp_dims =2, stage_sizes=[4,4,2],  features_in=4,stage_features=[4,8,16])  

















