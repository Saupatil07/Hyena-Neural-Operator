import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
import numpy as np
from .attention_module import PreNorm, PostNorm, LinearAttention, CrossLinearAttention,\
    FeedForward, GeGELU, ProjDotProduct
from torch.nn.init import xavier_uniform_, orthogonal_
from .hyenaoperator import HyenaOperator
from .encoder_module import TransformerCatNoCls


class FeedForward_q(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, dim),
        )
    def forward(self, x):
        return self.net(x)
    
# code copied from: https://github.com/ndahlquist/pytorch-fourier-feature-networks
# author: Nic Dahlquist
class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
    Given an input of size [batches, n, num_input_channels],
     returns a tensor of size [batches, n, mapping_size*2].
    """

    def __init__(self, num_input_channels, mapping_size=256, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = nn.Parameter(torch.randn((num_input_channels, mapping_size)) * scale, requires_grad=False)

    def forward(self, x):

        batches, num_of_points, channels = x.shape

        # Make shape compatible for matmul with _B.
        # From [B, N, C] to [(B*N), C].
        x = rearrange(x, 'b n c -> (b n) c')

        x = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        x = rearrange(x, '(b n) c -> b n c', b=batches)

        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
class CrossFormer_burger(nn.Module):
    def __init__(self,
                 dim,
                 attn_type,
                 heads,
                 dim_head,
                 mlp_dim,
                 residual=True,
                 use_ffn=True,
                 use_ln=True,
                 relative_emb=False,
                 scale=1.,
                 relative_emb_dim=2,
                 min_freq=1/64,
                 dropout=0.,
                 cat_pos=False,
                 ):
        super().__init__()

        self.cross_attn_module = CrossLinearAttention(dim, attn_type,
                                                       heads=heads, dim_head=dim_head, dropout=dropout,
                                                       relative_emb=relative_emb,
                                                       scale=scale,

                                                       relative_emb_dim=relative_emb_dim,
                                                       min_freq=min_freq,
                                                       init_method='orthogonal',
                                                       cat_pos=cat_pos,
                                                       pos_dim=relative_emb_dim,
                                                  )
        self.use_ln = use_ln
        self.residual = residual
        self.use_ffn = use_ffn
        #self.h1 = HyenaOperator(d_model=dim,l_max=4096*2)
        #self.h2 = HyenaOperator(d_model=dim,l_max=4096*2)
        #self.h3 = HyenaOperator(d_model=dim,l_max=4096*2)
        if self.use_ln:
            self.ln1 = nn.LayerNorm(dim)
            self.ln2 = nn.LayerNorm(dim)

        if self.use_ffn:
            self.ffn = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x, z, x_pos=None, z_pos=None):
        # x in [b n1 c]
        # b, n1, c = x.shape   # coordinate encoding
        # b, n2, c = z.shape   # system encoding
        if self.use_ln:
            z = self.ln1(z)
            if self.residual:
                x = self.ln2(self.cross_attn_module(x, z, x_pos, z_pos)) + x
                #x = self.ln2(self.h1(self.ln2(x))) + x
                #x = self.ln2(self.h2(self.ln2(x))) + x
                #x = self.ln2(self.h3(self.ln2(x))) + x
            else:
                x = self.ln2(self.cross_attn_module(x, z, x_pos, z_pos))
        else:
            if self.residual:
                x = self.cross_attn_module(x, z, x_pos, z_pos) + x
                #x = self.h1(x) + x
                #x = self.h2(x) + x
                #x = self.h3(x) + x
            else:
                x = self.cross_attn_module(x, z, x_pos, z_pos)

        if self.use_ffn:
            x = self.ffn(x) + x

        return x

class CrossFormer(nn.Module):
    def __init__(self,
                 dim,
                 attn_type,
                 heads,
                 dim_head,
                 mlp_dim,
                 residual=True,
                 use_ffn=True,
                 use_ln=True,
                 relative_emb=False,
                 scale=1.,
                 relative_emb_dim=2,
                 min_freq=1/64,
                 dropout=0.,
                 cat_pos=False,
                 ):
        super().__init__()

        self.cross_attn_module = CrossLinearAttention(dim, attn_type,
                                                       heads=heads, dim_head=dim_head, dropout=dropout,
                                                       relative_emb=relative_emb,
                                                       scale=scale,

                                                       relative_emb_dim=relative_emb_dim,
                                                       min_freq=min_freq,
                                                       init_method='orthogonal',
                                                       cat_pos=cat_pos,
                                                       pos_dim=relative_emb_dim,
                                                  )
        self.use_ln = use_ln
        self.residual = residual
        self.use_ffn = use_ffn
        self.h1 = HyenaOperator(d_model=dim,l_max=4096)
        self.h2 = HyenaOperator(d_model=dim,l_max=4096)
        self.h3 = HyenaOperator(d_model=dim,l_max=4096)
        if self.use_ln:
            self.ln1 = nn.LayerNorm(dim)
            self.ln2 = nn.LayerNorm(dim)

        if self.use_ffn:
            self.ffn = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x, z, x_pos=None, z_pos=None):
        # x in [b n1 c]
        # b, n1, c = x.shape   # coordinate encoding
        # b, n2, c = z.shape   # system encoding
        if self.use_ln:
            z = self.ln1(z)
            if self.residual:
                x = self.ln2(self.cross_attn_module(x, z, x_pos, z_pos)) + x
                x = self.ln2(self.h1(self.ln2(x))) + x
                x = self.ln2(self.h2(self.ln2(x))) + x
                x = self.ln2(self.h3(self.ln2(x))) + x
            else:
                x = self.ln2(self.cross_attn_module(x, z, x_pos, z_pos))
        else:
            if self.residual:
                x = self.cross_attn_module(x, z, x_pos, z_pos) + x
                x = self.h1(x) + x
                x = self.h2(x) + x
                x = self.h3(x) + x
            else:
                x = self.cross_attn_module(x, z, x_pos, z_pos)

        if self.use_ffn:
            x = self.ffn(x) + x

        return x

class PointWiseDecoder2D(nn.Module):
    def __init__(self,
                 latent_channels,  # 256??
                 out_channels,  # 1 or 2?
                 out_steps,  # 10
                 propagator_depth,
                 scale=8,
                 dropout=0.,
                 **kwargs,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.out_channels = out_channels
        self.out_steps = out_steps
        self.latent_channels = latent_channels

        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(2, self.latent_channels//2, scale=scale),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels//2, bias=False),
        )

        self.decoding_transformer = CrossFormer(self.latent_channels//2, 'galerkin', 4,
                                                self.latent_channels//2, self.latent_channels//2,
                                                relative_emb=True,
                                                scale=16.,
                                                relative_emb_dim=2,
                                                min_freq=1/64)

        self.expand_feat = nn.Linear(self.latent_channels//2, self.latent_channels)

        self.propagator = nn.ModuleList([
               nn.ModuleList([nn.LayerNorm(self.latent_channels),
               nn.Sequential(
                    nn.Linear(self.latent_channels + 2, self.latent_channels, bias=False),
                    nn.GELU(),
                    nn.Linear(self.latent_channels, self.latent_channels, bias=False),
                    nn.GELU(),
                    nn.Linear(self.latent_channels, self.latent_channels, bias=False))])
            for _ in range(propagator_depth)])

        self.to_out = nn.Sequential(
            nn.LayerNorm(self.latent_channels),
            nn.Linear(self.latent_channels, self.latent_channels//2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels // 2, self.latent_channels // 2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels//2, self.out_channels * self.out_steps, bias=True))
        
        self.h1 = HyenaOperator(d_model=self.latent_channels//2,l_max=4096,order=2)
        self.norm = nn.LayerNorm(self.latent_channels//2)
        self.f1 = FeedForward_q(self.latent_channels//2,self.latent_channels//2, dropout=0.03)
        
        self.h2 = HyenaOperator(d_model=self.latent_channels//2,l_max=4096,order=2)
        self.f2 = FeedForward_q(self.latent_channels//2,self.latent_channels//2, dropout=0.03)
        
        self.h3 = HyenaOperator(d_model=self.latent_channels//2,l_max=4096,order=2)
        self.f3 = FeedForward_q(self.latent_channels//2,self.latent_channels//2, dropout=0.03)
        
        self.h4 = HyenaOperator(d_model=self.latent_channels//2,l_max=4096,order=2)
        self.f4 = FeedForward_q(self.latent_channels//2,self.latent_channels//2, dropout=0.03)
        
        self.h5 = HyenaOperator(d_model=self.latent_channels//2,l_max=4096,order=2)
        self.f5 = FeedForward_q(self.latent_channels//2,self.latent_channels//2, dropout=0.03)
        
        self.h6 = HyenaOperator(d_model=self.latent_channels//2,l_max=4096,order=2)
        self.f6 = FeedForward_q(self.latent_channels//2,self.latent_channels//2, dropout=0.03)
        
        self.h7 = HyenaOperator(d_model=self.latent_channels//2,l_max=4096,order=2)
        self.f7 = FeedForward_q(self.latent_channels//2,self.latent_channels//2, dropout=0.03)
        
        self.h8 = HyenaOperator(d_model=self.latent_channels//2,l_max=4096,order=2)
        self.f8 = FeedForward_q(self.latent_channels//2,self.latent_channels//2, dropout=0.03)
        
    
    def propagate(self, z, pos):
        for layer in self.propagator:
            norm_fn, ffn = layer
            z = ffn(torch.cat((norm_fn(z), pos), dim=-1)) + z
        return z

    def decode(self, z):
        z = self.to_out(z)
        return z

    def get_embedding(self,
                      z,  # [b, n c]
                      propagate_pos,  # [b, n, 2]
                      input_pos
                      ):
        x = self.coordinate_projection.forward(propagate_pos)
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)
        z = self.expand_feat(z)
        return z

    def forward(self,
                z,              # [b, n, c]
                propagate_pos   # [b, n, 2]
                ):
        z = self.propagate(z, propagate_pos)
        u = self.decode(z)
        u = rearrange(u, 'b n (t c) -> b (t c) n', c=self.out_channels, t=self.out_steps)
        return u, z                # [b c_out t n], [b c_latent t n]

    def rollout(self,
                z,  # [b, n c]
                propagate_pos,  # [b, n, 2]
                forward_steps,
                input_pos):
        history = []
        
        x = self.coordinate_projection.forward(propagate_pos)
        
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)
        
        x1 = self.norm(self.h1(self.norm(z))) + z
        x1 = self.f1(x1)
        
        x2 = self.norm(self.h2(self.norm(z))) + z
        x2 = self.f2(x2)
        
        x3 = self.norm(self.h3(self.norm(z))) + z
        x3 = self.f3(x3)
        
        x4 = self.norm(self.h4(self.norm(z))) + z
        x4 = self.f4(x4)

        x5 = self.norm(self.h5(self.norm(z))) + z
        x5 = self.f5(x5)
        
        x6 = self.norm(self.h6(self.norm(z))) + z
        x6 = self.f6(x6)
        
        x7 = self.norm(self.h7(self.norm(z))) + z
        x7 = self.f7(x7)
        
        x8 = self.norm(self.h8(self.norm(z))) + z
        x8 = self.f8(x8)

        z = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8
        z = self.expand_feat(self.norm(z))
        
        #print(z.shape)
        #exit()
        for step in range(forward_steps//self.out_steps):
            z = self.propagate(z, propagate_pos)
            u = self.decode(z)
            history.append(rearrange(u, 'b n (t c) -> b (t c) n', c=self.out_channels, t=self.out_steps))
        history = torch.cat(history, dim=-2)  # concatenate in temporal dimension
        return history  # [b, length_of_history*c, n]


class PointWiseDecoder1D(nn.Module):
    # for Burgers equation
    def __init__(self,
                 latent_channels,  # 256??
                 out_channels,  # 1 or 2?
                 decoding_depth,  # 4?
                 scale=8,
                 res=2048,
                 **kwargs,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.out_channels = out_channels
        self.latent_channels = latent_channels

        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(1, self.latent_channels, scale=scale),
            nn.GELU(),
            nn.Linear(self.latent_channels*2, self.latent_channels, bias=False),
        )

        self.decoding_transformer = CrossFormer_burger(self.latent_channels, 'fourier', 8,
                                                self.latent_channels, self.latent_channels,
                                                relative_emb=True,
                                                scale=1,
                                                relative_emb_dim=1,
                                                min_freq=1/res)
        

        self.propagator = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.latent_channels, self.latent_channels, bias=False),
                nn.GELU(),
                nn.Linear(self.latent_channels, self.latent_channels, bias=False),
                nn.GELU(),
                nn.Linear(self.latent_channels, self.latent_channels, bias=False),)
            for _ in range(decoding_depth)])

        self.init_propagator_params()
        self.to_out = nn.Sequential(
            nn.Linear(self.latent_channels, self.latent_channels//2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels//2, self.out_channels, bias=True))
        
        self.h1 = HyenaOperator(d_model=self.latent_channels,l_max=4096)
        self.norm = nn.LayerNorm(self.latent_channels)
        self.f1 = FeedForward_q(self.latent_channels,self.latent_channels)
        self.h2 = HyenaOperator(d_model=self.latent_channels,l_max=4096)
        self.f2 = FeedForward_q(self.latent_channels,self.latent_channels)
        self.h3 = HyenaOperator(d_model=self.latent_channels,l_max=4096)
        self.f3 = FeedForward_q(self.latent_channels,self.latent_channels)
        self.h4 = HyenaOperator(d_model=self.latent_channels,l_max=4096)
        self.f4 = FeedForward_q(self.latent_channels,self.latent_channels)
        
    def propagate(self, z):
        for num_l, layer in enumerate(self.propagator):
            z = z + layer(z)
        return z

    def decode(self, z):
        z = self.to_out(z)
        return z

    def init_propagator_params(self):
        for block in self.propagator:
            for layers in block:
                    for param in layers.parameters():
                        if param.ndim > 1:
                            in_c = param.size(-1)
                            orthogonal_(param[:in_c], gain=1/in_c)
                            param.data[:in_c] += 1/in_c * torch.diag(torch.ones(param.size(-1), dtype=torch.float32))
                            if param.size(-2) != param.size(-1):
                                orthogonal_(param[in_c:], gain=1/in_c)
                                param.data[in_c:] += 1/in_c * torch.diag(torch.ones(param.size(-1), dtype=torch.float32))

    def forward(self,
                z,  # [b, n c]
                propagate_pos,  # [b, n, 1]
                input_pos=None,
                ):

        x = self.coordinate_projection.forward(propagate_pos)
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)
        x1 = self.norm(self.h1(self.norm(z))) + z
        x1 = self.f1(x1)
        
        x2 = self.norm(self.h2(self.norm(z))) + z
        x2 = self.f2(x2)

        x3 = self.norm(self.h3(self.norm(z))) + z
        x3 = self.f3(x3)
        
        x4 = self.norm(self.h4(self.norm(z))) + z
        x4 = self.f4(x4)

        z = x1 + x2 + x3 + x4 
        z = self.propagate(z)
        z = self.decode(z)
        return z  # [b, n, c]


class PointWiseDecoder2DSimple(nn.Module):
    # for Darcy equation
    def __init__(self,
                 latent_channels,  # 256??
                 out_channels,  # 1 or 2?
                 res=211,
                 scale=0.5,
                 **kwargs,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.out_channels = out_channels
        self.latent_channels = latent_channels

        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(2, self.latent_channels//2, scale=scale),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
        )

        self.decoding_transformer = CrossFormer_burger(self.latent_channels, 'galerkin', 4,
                                                self.latent_channels, self.latent_channels,
                                                use_ln=True,
                                                residual=True,
                                                relative_emb=True,
                                                scale=16,
                                                relative_emb_dim=2,
                                                min_freq=1/res)

        # self.init_propagator_params()
        self.h1 = HyenaOperator(d_model=self.latent_channels,l_max=8192*2)
        self.norm = nn.LayerNorm(self.latent_channels)
        self.f1 = FeedForward_q(self.latent_channels,self.latent_channels, dropout=0.03)
        
        self.h2 = HyenaOperator(d_model=self.latent_channels,l_max=8192*2)
        self.f2 = FeedForward_q(self.latent_channels,self.latent_channels, dropout=0.03)
        
        self.h3 = HyenaOperator(d_model=self.latent_channels,l_max=8192*2)
        self.f3 = FeedForward_q(self.latent_channels,self.latent_channels, dropout=0.03)
        
        self.h4 = HyenaOperator(d_model=self.latent_channels,l_max=8192*2)
        self.f4 = FeedForward_q(self.latent_channels,self.latent_channels, dropout=0.03)
        
        self.h5 = HyenaOperator(d_model=self.latent_channels,l_max=8192*2)
        self.f5 = FeedForward_q(self.latent_channels,self.latent_channels, dropout=0.03)
        
        self.h6 = HyenaOperator(d_model=self.latent_channels,l_max=8192*2)
        self.f6 = FeedForward_q(self.latent_channels,self.latent_channels, dropout=0.03)
        
        self.h7 = HyenaOperator(d_model=self.latent_channels,l_max=8192*2)
        self.f7 = FeedForward_q(self.latent_channels,self.latent_channels, dropout=0.03)
        
        self.h8 = HyenaOperator(d_model=self.latent_channels,l_max=8192*2)
        self.f8 = FeedForward_q(self.latent_channels,self.latent_channels, dropout=0.03)
        
        self.to_out = nn.Sequential(
            #nn.Linear(self.latent_channels+2, self.latent_channels, bias=False),
            #nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels//2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels//2, self.out_channels, bias=True))

    def decode(self, z):
        z = self.to_out(z)
        return z

    def forward(self,
                z,  # [b, n c]
                propagate_pos,  # [b, n, 1]
                input_pos=None,
                ):

        x = self.coordinate_projection.forward(propagate_pos)
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)
        
        x1 = self.norm(self.h1(self.norm(z))) + z
        x1 = self.f1(x1)
        
        x2 = self.norm(self.h2(self.norm(z))) + z
        x2 = self.f2(x2)

        x3 = self.norm(self.h3(self.norm(z))) + z
        x3 = self.f3(x3)
        
        x4 = self.norm(self.h4(self.norm(z))) + z
        x4 = self.f4(x4)

        x5 = self.norm(self.h5(self.norm(z))) + z
        x5 = self.f5(x5)
        
        x6 = self.norm(self.h6(self.norm(z))) + z
        x6 = self.f6(x6)
        
        x7 = self.norm(self.h7(self.norm(z))) + z
        x7 = self.f7(x7)
        
        x8 = self.norm(self.h8(self.norm(z))) + z
        x8 = self.f8(x8)

        z = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8
        #z = self.decode(torch.cat((z, propagate_pos), dim=-1))
        z = self.decode(self.norm(z))
        return z  # [b, n, c]

# class PieceWiseDecoder2DSimple(nn.Module):
#     # for Darcy flow inverse problem
#     def __init__(self,
#                  latent_channels,  # 256??
#                  out_channels,  # 1 or 2?
#                  res=141,
#                  scale=0.5,
#                  **kwargs,
#                  ):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         self.out_channels = out_channels
#         self.latent_channels = latent_channels

#         self.coordinate_projection = nn.Sequential(
#             GaussianFourierFeatureTransform(2, self.latent_channels//2, scale=scale),
#             # nn.Linear(2, self.latent_channels, bias=False),
#             # nn.GELU(),
#             # nn.Linear(self.latent_channels*2, self.latent_channels, bias=False),
#             # nn.GELU(),
#             nn.GELU(),
#             nn.Linear(self.latent_channels, self.latent_channels, bias=False),
#             nn.Dropout(0.05),
#         )

#         self.decoding_transformer = CrossFormer(self.latent_channels, 'galerkin', 4,
#                                                 self.latent_channels, self.latent_channels,
#                                                 use_ln=False,
#                                                 residual=True,
#                                                 use_ffn=False,
#                                                 relative_emb=True,
#                                                 scale=16,
#                                                 relative_emb_dim=2,
#                                                 min_freq=1/res)

#         # self.init_propagator_params()
#         self.to_out = nn.Sequential(
#             nn.Linear(self.latent_channels+2, self.latent_channels, bias=False),
#             nn.ReLU(),
#             nn.Linear(self.latent_channels, self.latent_channels, bias=False),
#             nn.ReLU(),
#             nn.Linear(self.latent_channels, self.latent_channels//2, bias=False),
#             nn.ReLU(),
#             nn.Linear(self.latent_channels//2, self.out_channels, bias=True))

#     def decode(self, z):
#         z = self.to_out(z)
#         return z

#     def forward(self,
#                 z,  # [b, n c]
#                 propagate_pos,  # [b, n, 1]
#                 input_pos=None,
#                 ):

#         x = self.coordinate_projection.forward(propagate_pos)
#         z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)

#         z = self.decode(torch.cat((z, propagate_pos), dim=-1))
#         return z  # [b, n, c]
