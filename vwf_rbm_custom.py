from typing import Union, Any,Optional,Callable, Sequence,Dict,Tuple
from dataclasses import dataclass, field

import numpy as np
import jax
from jax import numpy as jnp
from flax import linen as nn
from netket.nn.activation import relu
from jax.nn.initializers import normal,uniform
import math
from flax.linen.dtypes import promote_dtype


import functools
from functools import partial
from jax.tree_util import Partial


from netket.utils import HashableArray
from netket.utils.types import NNInitFunc
from netket import nn as nknn
import netket as nk
default_kernel_init0 = normal(stddev=0.01)
default_kernel_init = nn.initializers.variance_scaling(
    0.1**2, "fan_in", distribution="truncated_normal"
)

import math
#from netket.utils.types import DType
is2 = 1.0 / math.sqrt(2.0)
from depth_conv_custom import Depth_Conv_FFT,Point_Conv_RICO



class RBMModPhase_Trans_Inv(nn.Module):
    spac_dim: Tuple[int]
    #features_in: int
    features: int

    param_dtype: Any = np.float64
    activation: Any = nknn.log_cosh
   
    use_hidden_bias: bool = True
    precision: Any = None
    #kernel_init: NNInitFunc = default_kernel_init
    hidden_bias_init: NNInitFunc = nn.initializers.zeros
    
    out_std_init: float = 1.
    psi_std_init: float = .0005
    
    def setup(self):
      self.sd = len(self.spac_dim)
      self.n_sites =  np.prod(self.spac_dim)
      self.kernel_init =nn.initializers.variance_scaling(self.out_std_init**2, "fan_in", distribution="truncated_normal",in_axis=(0,1))
      self.gamma0 = (1. / self.out_std_init) * ( ( 2. * self.psi_std_init**2 ) / ( self.n_sites * self.features) )**(.25)


    def init_fn(self, key, shape, fill_value):
        return jnp.full(shape, fill_value)
        
    @nn.compact
    def __call__(self, inputs):

        
        batch_dim = inputs.shape[: -(self.sd+1)]
        features_in = inputs.shape[-1]
        
        x = inputs.reshape(-1,self.n_sites,features_in)
        x = x.transpose(0, 2, 1)
        x = x.reshape(*x.shape[:2], *self.spac_dim)
        
       
        kernel_re = self.param(
          "kernel_re",
          self.kernel_init,
          (self.n_sites, features_in, self.features),
          self.param_dtype, 
        )

        kernel_im = self.param(
          "kernel_im",
          self.kernel_init,
          (self.n_sites, features_in, self.features),
          self.param_dtype, 
        )
          
        kernel = jax.lax.complex(kernel_re,kernel_im)
          
        if self.use_hidden_bias:
          bias_re = self.param("hidden_bias_re", self.hidden_bias_init, (self.features,), self.param_dtype)
          bias_im = self.param("hidden_bias_im", self.hidden_bias_init, (self.features,), self.param_dtype)
          bias = jax.lax.complex(bias_re,bias_im)
        else:
            bias = None

        x, kernel,bias = promote_dtype(x, kernel,bias, dtype=None)
        dtype = x.dtype
        x = jnp.fft.fftn(x, s=self.spac_dim).reshape(*x.shape[:2], self.n_sites)

        
        
        x = jax.lax.dot_general(
            x, kernel, (((1,), (1,)), ((2,), (0,))), precision=self.precision
        )
        x = x.transpose(1,2,0)
        x = x.reshape(*x.shape[:2], *self.spac_dim)
        x = jnp.fft.ifftn(x, s=self.spac_dim,norm="ortho").reshape(*x.shape[:2], self.n_sites)
        #x = jnp.fft.ifftn(x, s=self.spac_dim).reshape(*x.shape[:2], self.n_sites)
        
        
        x = x.transpose(0, 2,1)
        if self.use_hidden_bias:
          x = x + bias
        

        '''
        gamma = self.param(
          "gamma", self.init_fn, (self.features,), self.gamma0
        )
        x = gamma * x
        '''
        x = self.gamma0 * x
        
        x = x.reshape(*batch_dim, -1)
        #x = x / math.sqrt(2*x.shape[-1])
        
        return jnp.sum( self.activation(x) ,axis=-1)


class RBMModPhase_Mom_Proj2(nn.Module):
    spac_dim: Tuple[int]
    #features_in: int
    features: int

    param_dtype: Any = np.float64
    #activation: Any = nknn.log_cosh
   
    #use_hidden_bias: bool = True
    precision: Any = None
    #kernel_init: NNInitFunc = default_kernel_init
    hidden_bias_re_init: NNInitFunc = normal(stddev=0.01)
    #hidden_bias_im_init: NNInitFunc = uniform(scale =1.6)
    hidden_bias_im_init: NNInitFunc = uniform(scale = .25*6.28318530718)
    site_bias_init: NNInitFunc = normal(stddev=0.01)
    out_std_init: float = 1.
    psi_std_init: float = .005
    
    def setup(self):
      self.sd = len(self.spac_dim)
      self.n_sites =  np.prod(self.spac_dim)
      #self.n_freq =  np.prod(self.spac_dim)
      self.kernel_init =nn.initializers.variance_scaling(self.out_std_init**2, "fan_in", distribution="truncated_normal",in_axis=(0,1))
      self.gamma0 = (1. / self.out_std_init) * ( ( 2. * self.psi_std_init**2 ) / ( self.n_sites * self.features) )**(.25)


    def init_fn(self, key, shape, fill_value):
        return jnp.full(shape, fill_value)
        
    @nn.compact
    def __call__(self, inputs):

        
        batch_dim = inputs.shape[: -(self.sd+1)]
        features_in = inputs.shape[-1]
        
        x = inputs.reshape(-1,self.n_sites,features_in)
        x = x.transpose(0, 2, 1)
        x = x.reshape(*x.shape[:2], *self.spac_dim)
        
       
        kernel_re = self.param(
          "kernel_re",
          self.kernel_init,
          (self.n_sites, features_in, self.features),
          self.param_dtype, 
        )

        kernel_im = self.param(
          "kernel_im",
          self.kernel_init,
          (self.n_sites, features_in, self.features),
          self.param_dtype, 
        )
          
        kernel = jax.lax.complex(kernel_re,kernel_im)
          
        b_re = self.param("hidden_bias_re", self.hidden_bias_re_init, (self.n_sites,self.features), self.param_dtype)
        b_im = self.param("hidden_bias_im", self.hidden_bias_im_init, (self.n_sites,self.features), self.param_dtype)
        b = jax.lax.complex(b_re,b_im)

        a_re = self.param("site_bias_re", self.site_bias_init, (self.n_sites,1), self.param_dtype)
        a_im = self.param("site_bias_im", self.site_bias_init, (self.n_sites,1), self.param_dtype)
        a = jax.lax.complex(a_re,a_im)
        

        x, kernel,a,b = promote_dtype(x, kernel,a,b, dtype=None)
        dtype = x.dtype
        x = jnp.fft.fftn(x, s=self.spac_dim).reshape(*x.shape[:2], self.n_sites)
        
        x = jax.lax.dot_general(
            x, kernel, (((1,), (1,)), ((2,), (0,))), precision=self.precision
        )
        x = x.transpose(1,2,0)
        x = x.reshape(*x.shape[:2], *self.spac_dim)
        x = jnp.fft.ifftn(x, s=self.spac_dim,norm="ortho").reshape(*x.shape[:2], self.n_sites)
        x = x.transpose(0, 2,1)

      
        x = self.gamma0 * x
        y = jnp.sum( nknn.log_cosh(x) , (-2,-1) )
        #x = jnp.sum(jnp.log(jnp.cosh(b) + jnp.sinh(b) * jnp.tanh(x)),-1) + a
        
        #some how the good one
        x = jnp.exp(a) + jnp.cosh(b) + jnp.sinh(b) * jnp.tanh(x)
        
        #x = a + jnp.log( jnp.cosh(b) + jnp.sinh(b) * jnp.tanh(x) )
        
        y = y + nk.jax.logsumexp_cplx(x,None,axis=(-2,-1))
        return y.reshape(*batch_dim)


class RBMModPhase_Mom_Proj3(nn.Module):
    spac_dim: Tuple[int]
    #features_in: int
    features: int
    n_bias: int
    
    param_dtype: Any = np.float64
    #activation: Any = nknn.log_cosh
   
    #use_hidden_bias: bool = True
    precision: Any = None
    #kernel_init: NNInitFunc = default_kernel_init
    
    beta: float = 6.0
    out_std_init: float = 1.0
    psi0_weight_init: float = 1.0
    psi0_std_init: float = .005
    psi_mom_weight_init: float = 1.0
    psi_mom_std_init: float = .005
    #bias_w_std_init: float = 1.00001
    
    
    def setup(self):
      self.sd = len(self.spac_dim)
      self.n_sites =  np.prod(self.spac_dim)
      self.n_freq =  np.prod(self.spac_dim)

      is2 = 1.0 /math.sqrt(2.0)
      
     
      self.kernel_init =nn.initializers.variance_scaling( .5 * self.out_std_init**2, "fan_in", distribution="truncated_normal")
      #self.kernel_init =nn.initializers.variance_scaling( .5 * self.out_std_init**2, "fan_in", distribution="normal",in_axis=1,out_axis=2,batch_axis=0)
      
      self.gamma_x = ( ( 2. * self.psi0_std_init**2 ) / ( ( self.n_sites * self.features)  ))**(.25)

      #self.gamma_b = ( self.psi_mom_std_init / self.gamma_x) / math.sqrt( 2*self.features)
      self.gamma_b = self.beta * self.gamma_x
      #self.gamma_b = 2.0 * self.gamma_x
        
     
      self.hidden_bias_re_init = normal(stddev = is2 * self.gamma_b)
      #self.hidden_bias_im_init = uniform(scale = 6.28318530718)
      self.hidden_bias_im_init = normal(stddev = is2 * self.gamma_b)
        
      self.site_bias_re_init = normal( stddev = .01) 
      self.site_bias_im_init = uniform(scale = 1.0*6.28318530718)
      
      gamma_w0 = math.sqrt( self.n_sites / (self.n_bias *( self.n_sites +2.0 * self.beta**2))) * (self.psi_mom_std_init / self.psi0_std_init)
      self.gamma_wp = math.sqrt(self.psi_mom_weight_init) * gamma_w0
      self.gamma_wm = math.sqrt(self.psi_mom_weight_init) * gamma_w0
      #self.gamma_wp = ( math.sqrt(self.psi_mom_weight_init) * gamma_w0 ) / ( math.sqrt( self.n_bias) * self.gamma_x)
      #self.gamma_wm = math.sqrt(self.psi_mom_weight_init) * gamma_w0

    def init_fn(self, key, shape, fill_value):
        return jnp.full(shape, fill_value)
        
    @nn.compact
    def __call__(self, inputs):

        
        batch_dim = inputs.shape[: -(self.sd+1)]
        features_in = inputs.shape[-1]
        
        x = inputs.reshape(-1,self.n_sites,features_in)
        #x = x - jnp.mean(x,axis=1,keepdims=True)
        x = x.transpose(0, 2, 1)
        x = x.reshape(*x.shape[:2], *self.spac_dim)
        
       
        kernel_re = self.param(
          "kernel_re",
          self.kernel_init,
          (self.n_sites, features_in, self.features),
          self.param_dtype, 
        )

        kernel_im = self.param(
          "kernel_im",
          self.kernel_init,
          (self.n_sites, features_in, self.features),
          self.param_dtype, 
        )
          
        kernel = jax.lax.complex(kernel_re,kernel_im)

        b_re = self.param("hidden_bias_re", self.hidden_bias_re_init, (self.features,self.n_bias), self.param_dtype)
        b_im = self.param("hidden_bias_im", self.hidden_bias_im_init, (self.features,self.n_bias), self.param_dtype)
        #b = self.gamma_b * jax.lax.complex(b_re,b_im)
        b =  jax.lax.complex(b_re,b_im+.0*6.28318530718)
        
        ap_re = self.param("mom_bias_plus_re", self.site_bias_re_init, (self.n_bias,self.n_freq), self.param_dtype)
        ap_im = self.param("mom_bias_plus_im", self.site_bias_im_init, (self.n_bias,self.n_freq), self.param_dtype)
        wp = self.gamma_wp * jnp.exp( jax.lax.complex(ap_re,-ap_im) )
        
        am_re = self.param("mom_bias_minus_re", self.site_bias_re_init, (self.n_bias,self.n_freq), self.param_dtype)
        am_im = self.param("mom_bias_minus_im", self.site_bias_im_init, (self.n_bias,self.n_freq), self.param_dtype)
        wm = self.gamma_wm * jnp.exp( jax.lax.complex(am_re,-am_im) )



        
        c_re = self.param("site0_bias_re", normal( stddev = .01), (1,), self.param_dtype)
        c_im = self.param("site0_bias_im", uniform(scale = .01*6.28318530718) , (1,), self.param_dtype)
        c = .5*jnp.log(self.psi0_weight_init) + jax.lax.complex(c_re,c_im)
        
        
        x, kernel,wp,wm,b = promote_dtype(x, kernel,wp,wm,b, dtype=None)
        dtype = x.dtype
        #x = jnp.fft.fftn(x, s=self.spac_dim,norm="ortho").reshape(*x.shape[:2], self.n_sites)
        x = jnp.fft.fftn(x, s=self.spac_dim).reshape(*x.shape[:2], self.n_sites)
        x = jax.lax.dot_general(
            x, kernel, (((1,), (1,)), ((2,), (0,))), precision=self.precision
        )
        x = x.transpose(1,2,0)
        x = x.reshape(*x.shape[:2], *self.spac_dim)
        x = jnp.fft.ifftn(x, s=self.spac_dim,norm="ortho").reshape(*x.shape[:2], self.n_sites)
        #x = x.transpose(0, 2,1)
        x = jnp.expand_dims(x,axis=-2)
        x = (self.gamma_x/self.out_std_init) * x


        
        
        
        #w = jnp.fft.ifftn(w.reshape(-1,*self.spac_dim), s=self.spac_dim,norm="ortho").reshape(-1,self.n_sites)
        wp = jnp.fft.ifftn(wp.reshape(-1,*self.spac_dim), s=self.spac_dim,norm="ortho").reshape(-1,self.n_sites)
        wp = wp.conj()
        wm = jnp.fft.ifftn(wm.reshape(-1,*self.spac_dim), s=self.spac_dim,norm="ortho").reshape(-1,self.n_sites)
        wm = wm.conj()

        y = nknn.log_cosh(x)
        b=jnp.expand_dims(b,axis=-1)
        
        ch_b,sh_b,th_x = jnp.cosh(b),jnp.sinh(b),jnp.tanh(x)
        Bp = jnp.sum(jnp.log( ch_b + sh_b*th_x) ,axis=1)
        #Bp = jnp.log( jnp.exp(Bp) - jnp.exp(jnp.sum(jnp.log( ch_b),axis=0)))
        Bm = jnp.sum(jnp.log( ch_b - sh_b*th_x), axis=1)
        #Bm = jnp.log( jnp.exp(Bm) - jnp.exp(jnp.sum(jnp.log( ch_b),axis=0)))
        
        #Bp = jnp.sum(nknn.log_cosh(x + b) - y,axis=1)
        #Bm = jnp.sum(nknn.log_cosh(x - b) - y,axis=1)
      
        
        y = jnp.sum(y,axis =(-3,-2,-1))
        ypp = nk.jax.logsumexp_cplx(Bp,wp,axis=(-2,-1) ) + y
        ypm = nk.jax.logsumexp_cplx(Bm,wp,axis=(-2,-1) ) + y
        
        ymp = nk.jax.logsumexp_cplx(Bp,wm,axis=(-2,-1) ) + y
        ymm = nk.jax.logsumexp_cplx(Bm,-wm,axis=(-2,-1) ) + y
        y0 = c + y

        y = nk.jax.logsumexp_cplx( jnp.stack( (y0,ypp,ypm,ymp,ymm) ,axis=-1),None,axis=-1)
        #y = nk.jax.logsumexp_cplx( jnp.stack( (y0,ymp,ymm) ,axis=-1),None,axis=-1)

        
        
        return y.reshape(*batch_dim)



def freq_mask_hash(spac_dim , freq_list):
    n_sites = np.prod(spac_dim)
    mask = np.zeros((n_sites, ), dtype=bool)
    for freq_vec in freq_list:
        i = np.ravel_multi_index(freq_vec,spac_dim)
        mask[i] = True

    return HashableArray(mask)
    



class RBMModPhase_Mom_Proj_Group_TranSF(nn.Module):
    spac_dim: Tuple[int]
    features: int
    
    param_dtype: Any = np.float64
    use_hidden_bias: bool = True
    precision: Any = None
    psi_std_init: float = .005
    #mask: HashableArray | None = None
    mask: Union[HashableArray , None] = None
    
    def setup(self):
      self.sd = len(self.spac_dim)
      self.n_sites =  np.prod(self.spac_dim)
      #self.n_freq =  np.prod(self.spac_dim) 
    
      self.group_dim = self.spac_dim + (2,)
      self.n_group =  np.prod(self.group_dim)

      if self.mask is not None:
        (self.freq_indices,) = np.nonzero(self.mask.wrapped)
        self.freq_indices = jnp.sort(self.freq_indices)
        self.n_freq = len(self.freq_indices)
        
      else:
        self.n_freq =  np.prod(self.group_dim)
        
      self.gamma_x = ( ( 2. * self.psi_std_init**2 ) / (    self.n_group * self.features       ) )**(.25)
  
      self.mom_bias_re_init = normal( stddev = .01) 
      self.mom_bias_im_init = uniform(scale = 1.0*6.28318530718)
      self.gamma_w = 1.0 
     
        
    @nn.compact
    def __call__(self, inputs):



        
        batch_dim = inputs.shape[: -(self.sd+2)]
        features_in = inputs.shape[-1]


        x = Point_Conv_RICO(self.features,use_bias=False,name="RBM_Point_Conv")(inputs)
        x = Depth_Conv_FFT(self.group_dim,self.features,use_bias=self.use_hidden_bias,name="RBM_Depth_Conv")(x)
        
        x = self.gamma_x * x.reshape(-1,self.n_group,self.features)
       
        
        
        
        a_re = self.param("mom_bias_re", self.mom_bias_re_init, (self.features,self.n_freq), self.param_dtype)
        a_im = self.param("mom_bias_im", self.mom_bias_im_init, (self.features,self.n_freq), self.param_dtype)
        w_nzf = self.gamma_w * jnp.exp(jax.lax.complex(a_re,a_im))

        if self.mask is not None:
            w_re = (
                jnp.zeros((self.features,self.n_group), dtype=self.param_dtype)
                .at[:,self.freq_indices]
                .set(w_nzf.real, unique_indices=True, indices_are_sorted=True)
                #.set(w_nzf.real, unique_indices=True)
            )
            w_im = (
                jnp.zeros((self.features,self.n_group), dtype=self.param_dtype)
                .at[:,self.freq_indices]
                .set(w_nzf.imag, unique_indices=True, indices_are_sorted=True)
                #.set(w_nzf.imag, unique_indices=True)
            )
            w = jax.lax.complex(w_re,w_im)
        else:
            w = w_nzf
           
        w = jnp.fft.ifftn(w.reshape(self.features,*self.group_dim), s=self.group_dim,norm="ortho").reshape(self.features,self.n_group)
        w = w.T

        y = jnp.sum(nknn.log_cosh(x),axis=(-2,-1))
        x = jnp.log(jnp.tanh( x))
        y = y + nk.jax.logsumexp_cplx(x,w,axis = (-2,-1) )
        

        return y.reshape(*batch_dim)
      



class RBMModPhase_Mom_Proj_Group_TranSF2(nn.Module):
    spac_dim: Tuple[int]
    features: int
    
    param_dtype: Any = np.float64
    use_hidden_bias: bool = True
    precision: Any = None
    psi_std_init: float = .005
    #mask: HashableArray | None = None
    mask: Union[HashableArray , None] = None
    
    
    def setup(self):
      self.sd = len(self.spac_dim)
      self.n_sites =  np.prod(self.spac_dim)
      #self.n_freq =  np.prod(self.spac_dim) 
    
      self.group_dim = self.spac_dim + (2,)
      self.n_group =  np.prod(self.group_dim)

      if self.mask is not None:
        (self.freq_indices,) = np.nonzero(self.mask.wrapped)
        self.freq_indices = jnp.sort(self.freq_indices)
        self.n_freq = len(self.freq_indices)
        
      else:
        self.n_freq =  np.prod(self.group_dim)
        
      self.gamma_x = ( ( 2. * self.psi_std_init**2 ) / (    self.n_group * self.features       ) )**(.25)
  
      self.mom_weight_re_init = normal( stddev = .01) 
      self.mom_weight_im_init = uniform(scale = 1.0*6.28318530718)
        
      self.feat_bias_re_init = normal( stddev = is2) 
      self.feat_bias_im_init = normal( stddev = is2)

    
      self.gamma_b = self.gamma_x * is2
     
        
    @nn.compact
    def __call__(self, inputs):



        
        batch_dim = inputs.shape[: -(self.sd+2)]
        features_in = inputs.shape[-1]

        x = inputs.reshape(-1,*self.group_dim,features_in)
        x = Point_Conv_RICO(self.features,use_bias=False,name="RBM_Point_Conv")(x)
        x = Depth_Conv_FFT(self.group_dim,self.features,use_bias=self.use_hidden_bias,name="RBM_Depth_Conv")(x)
        
        x = self.gamma_x * x.reshape(-1,self.n_group,self.features)
       
        
        
        
        a_re = self.param("mom_weight_re", self.mom_weight_re_init, (self.n_freq,), self.param_dtype)
        a_im = self.param("mom_weight_im", self.mom_weight_im_init, (self.n_freq,), self.param_dtype)
        w_nzf =  jnp.exp(jax.lax.complex(a_re,a_im))

        if self.mask is not None:
            w_re = (
                jnp.zeros((self.n_group,), dtype=self.param_dtype)
                .at[self.freq_indices]
                .set(w_nzf.real, unique_indices=True, indices_are_sorted=True)
                #.set(w_nzf.real, unique_indices=True)
            )
            w_im = (
                jnp.zeros((self.n_group,), dtype=self.param_dtype)
                .at[self.freq_indices]
                .set(w_nzf.imag, unique_indices=True, indices_are_sorted=True)
                #.set(w_nzf.imag, unique_indices=True)
            )
            w = jax.lax.complex(w_re,w_im)
        else:
            w = w_nzf
           
        w = jnp.fft.ifftn(w.reshape(*self.group_dim), s=self.group_dim,norm="ortho").reshape(self.n_group)
        

        b_re = self.param("feat_bias_re", self.feat_bias_re_init, (self.features,), self.param_dtype)
        b_im = self.param("feat_bias_im", self.feat_bias_im_init, (self.features,), self.param_dtype)
        b = self.gamma_b * jax.lax.complex(b_re,b_im)


        y = jnp.sum(nknn.log_cosh(x),axis=(-2,-1))
        q = jnp.sum(nknn.log_cosh(x+b) - nknn.log_cosh(x) , axis=-1)

        y = y + nk.jax.logsumexp_cplx(q,w,axis = -1 )


        return y.reshape(*batch_dim)
      


class RBMModPhase_Mom_Proj_Group_TranSF3(nn.Module):
    spac_dim: Tuple[int]
    features: int
    
    param_dtype: Any = np.float64
    use_hidden_bias: bool = True
    precision: Any = None
    psi_std_init: float = .005
    #mask: HashableArray | None = None
    mask: Union[HashableArray , None] = None
    
    
    def setup(self):
      self.sd = len(self.spac_dim)
      self.n_sites =  np.prod(self.spac_dim)
      #self.n_freq =  np.prod(self.spac_dim) 
    
      self.group_dim = self.spac_dim + (2,)
      self.n_group =  np.prod(self.group_dim)

      if self.mask is not None:
        (self.freq_indices,) = np.nonzero(self.mask.wrapped)
        self.freq_indices = jnp.sort(self.freq_indices)
        self.n_freq = len(self.freq_indices)
        
      else:
        self.n_freq =  np.prod(self.group_dim)
        
      self.gamma_x = ( ( 2. * self.psi_std_init**2 ) / (     self.features       ) )**(.25)
  
      self.mom_weight_re_init = normal( stddev = .01) 
      self.mom_weight_im_init = uniform(scale = 1.0*6.28318530718)
        
     

    
      
     
        
    @nn.compact
    def __call__(self, inputs):



        
        batch_dim = inputs.shape[: -(self.sd+2)]
        features_in = inputs.shape[-1]

        x = inputs.reshape(-1,*self.group_dim,features_in)
        x = Point_Conv_RICO(self.features,use_bias=False,name="RBM_Point_Conv")(x)
        x = Depth_Conv_FFT(self.group_dim,self.features,use_bias=self.use_hidden_bias,name="RBM_Depth_Conv")(x)
        
        x = self.gamma_x * x.reshape(-1,self.n_group,self.features)
        
        a_re = self.param("mom_weight_re", self.mom_weight_re_init, (self.n_freq,), self.param_dtype)
        a_im = self.param("mom_weight_im", self.mom_weight_im_init, (self.n_freq,), self.param_dtype)
        w_nzf =  jnp.exp(jax.lax.complex(a_re,a_im))

        if self.mask is not None:
            w_re = (
                jnp.zeros((self.n_group,), dtype=self.param_dtype)
                .at[self.freq_indices]
                .set(w_nzf.real, unique_indices=True, indices_are_sorted=True)
                #.set(w_nzf.real, unique_indices=True)
            )
            w_im = (
                jnp.zeros((self.n_group,), dtype=self.param_dtype)
                .at[self.freq_indices]
                .set(w_nzf.imag, unique_indices=True, indices_are_sorted=True)
                #.set(w_nzf.imag, unique_indices=True)
            )
            w = jax.lax.complex(w_re,w_im)
        else:
            w = w_nzf
           
        w = jnp.fft.ifftn(w.reshape(*self.group_dim), s=self.group_dim,norm="ortho").reshape(self.n_group)
        


        x = jnp.sum(nknn.log_cosh(x),axis=-1)
        

        x = nk.jax.logsumexp_cplx(x,w,axis = -1 )


        return x.reshape(*batch_dim)



