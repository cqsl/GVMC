from functools import partial
from typing import Union, Any,Callable, Sequence, Tuple,Optional

import numpy as np
import jax
from jax import numpy as jnp
from flax import linen as nn
from netket.nn.activation import relu
from jax.nn.initializers import normal
from jax import lax
from flax.linen.dtypes import promote_dtype


from netket.utils import HashableArray
from netket.utils.types import NNInitFunc
#from netket.utils.group import PermutationGroup
from netket import nn as nknn
import math
#from netket.utils.types import DType
is2 = 1.0 / math.sqrt(2.0)
'''
defualt_dtype = np.float64
default_kernel_init = normal(stddev=0.01)
default_kernel_init = nn.initializers.variance_scaling(
    0.2, "fan_in", distribution="truncated_normal",dtype=defualt_dtype
)
default_dconv_init = default_kernel_init
default_pwconv_init = default_kernel_init
'''
defualt_dtype = np.float64
default_kernel_init = nn.initializers.variance_scaling(
    is2, "fan_in", distribution="truncated_normal",dtype=defualt_dtype
)

class Point_Conv_RICO(nn.Module):
  features:int
  use_bias: bool = False
  #layer_norm: bool = False
  param_dtype: Any = defualt_dtype
  #kernel_init = nn.initializers.variance_scaling( .5, "fan_in", distribution="truncated_normal",dtype=defualt_dtype)
  kernel_init: NNInitFunc = default_kernel_init
  
  @nn.compact
  def __call__(self,inputs):

    
    
      
    Dense_re = nn.Dense(self.features , use_bias=self.use_bias,kernel_init=self.kernel_init,param_dtype=self.param_dtype, name="Dense_re")
    Dense_im = nn.Dense(self.features , use_bias=self.use_bias,kernel_init=self.kernel_init,param_dtype=self.param_dtype, name="Dense_im")
    #return jax.lax.complex( Dense_re(inputs) , Dense_im(inputs) )
    x=Dense_re(inputs)
    y=Dense_im(inputs)
    '''
    if self.layer_norm:
        x=nn.LayerNorm(use_scale=True, use_bias=self.use_bias, param_dtype=self.param_dtype)(x)
        y=nn.LayerNorm(use_scale=True, use_bias=self.use_bias, param_dtype=self.param_dtype)(y)
    '''
    return jax.lax.complex(x,y)
    
    '''
    print("afas")
    print(inputs.shape)
    print( (jnp.shape(inputs)[-1], self.features) )
    print(  self.param_dtype )
    '''
    #x_re = nn.Dense(self.features , use_bias=self.use_bias,kernel_init = default_kernel_init,param_dtype=defualt_dtype, name="Dense_re")(inputs)
    #x_im = nn.Dense(self.features , use_bias=self.use_bias,kernel_init = self.kernel_init,param_dtype=defualt_dtype, name="Dense_im")(inputs)
    #return jax.lax.complex( x_re , x_im )
    
    #x = nn.Dense(self.features ,use_bias=self.use_bias, kernel_init=self.kernel_init,param_dtype=defualt_dtype, name="pwconv1")(inputs)
    #return x




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
        


class Depth_Conv_FFT(nn.Module):
  spac_dim: Tuple[int]
  features: int
  use_bias: bool = False
  real_in_out: bool = False
  param_dtype: Any = defualt_dtype
  #kernel_init: NNInitFunc = normal(stddev=0.707106781186547)
  kernel_init: NNInitFunc = nn.initializers.variance_scaling( is2, "fan_in", distribution="truncated_normal",in_axis=1,out_axis=0,batch_axis=())
  bias_init: NNInitFunc = nn.initializers.zeros

  
  def setup(self):
      self.sd = len(self.spac_dim)
      self.spac_axes=range(1,self.sd+1)
      self.n_sites =  np.prod(self.spac_dim)
      if self.real_in_out:
        self.freq_dim = self.spac_dim[:-1] + (self.spac_dim[-1]//2 +1,)
      else:
        self.freq_dim = self.spac_dim
      self.n_freq =  np.prod(self.freq_dim)
  
    
  @nn.compact
  def __call__(self,inputs):

        batch_dim = inputs.shape[: -(self.sd+1)]
        '''
        x = inputs.reshape(-1,self.n_sites,self.features)
        x = x.transpose(0, 2, 1)
        x = x.reshape(*x.shape[:2], *self.spac_dim)

        if self.real_in_out:
          x = jnp.fft.rfftn(x, s=self.spac_dim).reshape(*x.shape[:2], self.n_freq)
        else:
          x = jnp.fft.fftn(x, s=self.spac_dim).reshape(*x.shape[:2], self.n_freq)
  
  
        kernel_re = self.param(
          "kernel_re",
          self.kernel_init,
          (self.features,self.n_freq),
          self.param_dtype, 
        )
        kernel_im = self.param(
          "kernel_im",
          self.kernel_init,
          (self.features,self.n_freq),
          self.param_dtype, 
        )
        kernel = jax.lax.complex(kernel_re,kernel_im)

        x, kernel = promote_dtype(x, kernel, dtype=None)
        x = x * kernel
        
        x = x.reshape(*x.shape[:2], *self.freq_dim)
        if self.real_in_out:
          x = jnp.real( jnp.fft.irfftn(x, s=self.spac_dim,norm="ortho").reshape(*x.shape[:2], self.n_sites) )
        else:
          x = jnp.fft.ifftn(x, s=self.spac_dim,norm="ortho").reshape(*x.shape[:2], self.n_sites)
        
        x = x.transpose(0, 2,1)
        '''
      
        
        x = inputs.reshape(-1,*self.spac_dim,self.features)
        if self.real_in_out:
          x = jnp.fft.rfftn(x, s=self.spac_dim,axes=self.spac_axes)
        else:
          x = jnp.fft.fftn(x, s=self.spac_dim,axes=self.spac_axes)

        
        kernel_re = self.param(
          "kernel_re",
          self.kernel_init,
          (*self.freq_dim,self.features),
          self.param_dtype, 
        )
        kernel_im = self.param(
          "kernel_im",
          self.kernel_init,
          (*self.freq_dim,self.features),
          self.param_dtype, 
        )
        kernel = jax.lax.complex(kernel_re,kernel_im)

        x, kernel = promote_dtype(x, kernel, dtype=None)
        x = x * kernel
        
       
        if self.real_in_out:
          x =  jnp.fft.irfftn(x, s=self.spac_dim,axes=self.spac_axes,norm="ortho")
        else:
          x = jnp.fft.ifftn(x, s=self.spac_dim,axes=self.spac_axes,norm="ortho")
        

      
        if self.use_bias:
          if self.real_in_out:
            bias = self.param("bias", self.bias_init, (self.features,), self.param_dtype)
          else:
            bias_re = self.param("bias_re", self.bias_init, (self.features,), self.param_dtype)
            bias_im = self.param("bias_im", self.bias_init, (self.features,), self.param_dtype)
            bias = jax.lax.complex(bias_re,bias_im)
        else:
            bias = None

        if self.use_bias:
          x = x + bias
          
        #return x.reshape(*inputs.shape)
        return x.reshape(*batch_dim,*self.spac_dim,self.features)


class Depth_Conv_FFT_Real(nn.Module):
  spac_dim: Tuple[int]
  features: int
  use_bias: bool = False
  param_dtype: Any = defualt_dtype
  kernel_init: NNInitFunc =  nn.initializers.normal(stddev=1.0, dtype=defualt_dtype)
  bias_init: NNInitFunc = nn.initializers.zeros


  def setup(self):
      self.sd = len(self.spac_dim)
      self.spac_axes=range(1,self.sd+1)
      self.spac_axes_W =range(0,self.sd)
      self.n_sites =  np.prod(self.spac_dim)
      
  
    
  @nn.compact
  def __call__(self,inputs):

        batch_dim = inputs.shape[: -(self.sd+1)]
        x = inputs.reshape(-1,*self.spac_dim,self.features)
        x = jnp.fft.rfftn(x, s=self.spac_dim,axes=self.spac_axes,norm="ortho")
       
        kernel = self.param(
          "kernel",
          self.kernel_init,
          (*self.spac_dim,self.features),
          self.param_dtype, 
        )
        W = jnp.fft.rfftn(kernel, s=self.spac_dim,axes=self.spac_axes_W,norm="ortho")

        #x, kernel = promote_dtype(x, W, dtype=None)
        x = x * W
        x =  jnp.fft.irfftn(x, s=self.spac_dim,axes=self.spac_axes,norm="ortho")
        
   
    
        if self.use_bias:
          bias = self.param("bias", self.bias_init, (self.features,), self.param_dtype)
          x = x + bias
        
        
        return x.reshape(*batch_dim,*self.spac_dim,self.features)
