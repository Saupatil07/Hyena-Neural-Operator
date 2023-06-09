import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
from .attention_module import pair, PreNorm, PostNorm,\
    StandardAttention, FeedForward, LinearAttention, ReLUFeedForward
from .hyenaoperator import HyenaOperator
ACTIVATION_REGISTRY = {
    "relu": nn.ReLU(),
    "silu": nn.SiLU(),
    "gelu": nn.GELU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
}
class FeedForward_q(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*2,hidden_dim*2),
            nn.GELU(),
            nn.Linear(hidden_dim*2,dim)
        )
    def forward(self, x):
        return self.net(x)

class TransformerCatNoCls(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 heads,
                 dim_head,
                 mlp_dim,
                 attn_type,  # ['standard', 'galerkin', 'fourier']
                 use_ln=False,
                 scale=16,     # can be list, or an int
                 dropout=0.,
                 relative_emb_dim=2,
                 min_freq=1/64,
                 attention_init='orthogonal',
                 init_gain=None,
                 use_relu=False,
                 cat_pos=False,
                 ):
        super().__init__()
        assert attn_type in ['standard', 'galerkin', 'fourier']

        if isinstance(scale, int):
            scale = [scale] * depth
        assert len(scale) == depth

        self.layers = nn.ModuleList([])
        self.attn_type = attn_type
        self.use_ln = use_ln

        if attn_type == 'standard':
            for _ in range(depth):
                self.layers.append(
                    nn.ModuleList([
                    PreNorm(dim, StandardAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                    PreNorm(dim,  FeedForward(dim, mlp_dim, dropout=dropout)
                                  if not use_relu else ReLUFeedForward(dim, mlp_dim, dropout=dropout))]),
                )
        else:
            for d in range(depth):
                if scale[d] != -1 or cat_pos:
                    attn_module = LinearAttention(dim, attn_type,
                                                   heads=heads, dim_head=dim_head, dropout=dropout,
                                                   relative_emb=True, scale=scale[d],
                                                   relative_emb_dim=relative_emb_dim,
                                                   min_freq=min_freq,
                                                   init_method=attention_init,
                                                   init_gain=init_gain
                                                   )
                else:
                    attn_module = LinearAttention(dim, attn_type,
                                                  heads=heads, dim_head=dim_head, dropout=dropout,
                                                  cat_pos=True,
                                                  pos_dim=relative_emb_dim,
                                                  relative_emb=False,
                                                  init_method=attention_init,
                                                  init_gain=init_gain
                                                  )
                if not use_ln:
                    self.layers.append(
                        nn.ModuleList([
                                        attn_module,
                                        FeedForward(dim, mlp_dim, dropout=dropout)
                                        if not use_relu else ReLUFeedForward(dim, mlp_dim, dropout=dropout)
                        ]),
                        )
                else:
                    self.layers.append(
                        nn.ModuleList([
                            nn.LayerNorm(dim),
                            attn_module,
                            nn.LayerNorm(dim),
                            FeedForward(dim, mlp_dim, dropout=dropout)
                            if not use_relu else ReLUFeedForward(dim, mlp_dim, dropout=dropout),
                        ]),
                    )

    def forward(self, x, pos_embedding):
        # x in [b n c], pos_embedding in [b n 2]
        b, n, c = x.shape

        for layer_no, attn_layer in enumerate(self.layers):
            if not self.use_ln:
                [attn, ffn] = attn_layer

                x = attn(x, pos_embedding) + x
                x = ffn(x) + x
            else:
                [ln1, attn, ln2, ffn] = attn_layer
                x = ln1(x)
                x = attn(x, pos_embedding) + x
                x = ln2(x)
                x = ffn(x) + x
        return x

class SpatialTemporalEncoder2D(nn.Module):
    def __init__(self,
                 input_channels,           # how many channels
                 in_emb_dim,               # embedding dim of token                 (how about 512)
                 out_seq_emb_dim,          # embedding dim of encoded sequence      (how about 256)
                 heads,
                 depth,                    # depth of transformer / how many layers of attention    (4)
                 ):
        super().__init__()

        self.to_embedding = nn.Sequential(
            # Rearrange('b c n -> b n c'),
            nn.Linear(input_channels, in_emb_dim, bias=False),
        )
        
        self.h1 = HyenaOperator(d_model=in_emb_dim,l_max=4096)
        self.norm = nn.LayerNorm(in_emb_dim)
        self.f1 = FeedForward_q(in_emb_dim,in_emb_dim, dropout=0.03)
        
        self.h2 = HyenaOperator(d_model=in_emb_dim,l_max=4096)
        self.f2 = FeedForward_q(in_emb_dim,in_emb_dim, dropout=0.03)
        
        self.h3 = HyenaOperator(d_model=in_emb_dim,l_max=4096)
        self.f3 = FeedForward_q(in_emb_dim,in_emb_dim, dropout=0.03)
        
        self.h4 = HyenaOperator(d_model=in_emb_dim,l_max=4096)
        self.f4 = FeedForward_q(in_emb_dim,in_emb_dim, dropout=0.03)
        
        self.h5 = HyenaOperator(d_model=in_emb_dim,l_max=4096)
        self.f5 = FeedForward_q(in_emb_dim,in_emb_dim, dropout=0.03)
        
        self.h6 = HyenaOperator(d_model=in_emb_dim,l_max=4096)
        self.f6 = FeedForward_q(in_emb_dim,in_emb_dim, dropout=0.03)
        
        self.h7 = HyenaOperator(d_model=in_emb_dim,l_max=4096)
        self.f7 = FeedForward_q(in_emb_dim,in_emb_dim, dropout=0.03)
        
        self.h8 = HyenaOperator(d_model=in_emb_dim,l_max=4096)
        self.f8 = FeedForward_q(in_emb_dim,in_emb_dim, dropout=0.03)
        
        self.project_to_latent = nn.Sequential(
            nn.Linear(in_emb_dim, out_seq_emb_dim, bias=False))
    
        
    def forward(self,
                x,  # [b, t(*c)+2, n]
                input_pos,  # [b, n, 2]
                ):
        x = self.to_embedding(x)
        
        x1 = self.norm(self.h1(self.norm(x))) + x
        x1 = self.f1(x1)

        x2 = self.norm(self.h2(self.norm(x))) + x
        x2 = self.f2(x2)

        x3 = self.norm(self.h3(self.norm(x))) + x
        x3 = self.f3(x3)

        x4 = self.norm(self.h4(self.norm(x))) + x
        x4 = self.f4(x4)

        x5 = self.norm(self.h5(self.norm(x))) + x
        x5 = self.f5(x5)

        x6 = self.norm(self.h6(self.norm(x))) + x
        x6 = self.f6(x6)

        x7 = self.norm(self.h7(self.norm(x))) + x
        x7 = self.f7(x7)

        x8 = self.norm(self.h8(self.norm(x))) + x
        x8 = self.f8(x8)
        
        x = x1 + x2 + x3 + x4 + x5 + x6 +x7 +x8
        x = self.project_to_latent(self.norm(x))

        return x


class SpatialEncoder2D(nn.Module):
    def __init__(self,
                 input_channels,           # how many channels
                 in_emb_dim,               # embedding dim of token                 (how about 512)
                 out_seq_emb_dim,          # embedding dim of encoded sequence      (how about 256)
                 heads,
                 depth,                    # depth of transformer / how many layers of attention    (4)
                 res,
                 use_ln=True,
                 emb_dropout=0.05,           # dropout of embedding
                 ):
        super().__init__()

        self.to_embedding = nn.Sequential(
            nn.Linear(input_channels, in_emb_dim, bias=False),
        )

        self.dropout = nn.Dropout(emb_dropout)
        self.h1 = HyenaOperator(d_model=in_emb_dim,l_max=8192*2)
        self.norm = nn.LayerNorm(in_emb_dim)
        self.f1 = FeedForward_q(in_emb_dim,in_emb_dim, dropout=0.03)
        
        self.h2 = HyenaOperator(d_model=in_emb_dim,l_max=8192*2)
        self.f2 = FeedForward_q(in_emb_dim,in_emb_dim, dropout=0.03)
        
        self.h3 = HyenaOperator(d_model=in_emb_dim,l_max=8192*2)
        self.f3 = FeedForward_q(in_emb_dim,in_emb_dim, dropout=0.03)
        
        self.h4 = HyenaOperator(d_model=in_emb_dim,l_max=8192*2)
        self.f4 = FeedForward_q(in_emb_dim,in_emb_dim, dropout=0.03)
        
        self.h5 = HyenaOperator(d_model=in_emb_dim,l_max=8192*2)
        self.f5 = FeedForward_q(in_emb_dim,in_emb_dim, dropout=0.03)
        
        self.h6 = HyenaOperator(d_model=in_emb_dim,l_max=8192*2)
        self.f6 = FeedForward_q(in_emb_dim,in_emb_dim, dropout=0.03)
        
        self.h7 = HyenaOperator(d_model=in_emb_dim,l_max=8192*2)
        self.f7 = FeedForward_q(in_emb_dim,in_emb_dim, dropout=0.03)
        
        self.h8 = HyenaOperator(d_model=in_emb_dim,l_max=8192*2)
        self.f8 = FeedForward_q(in_emb_dim,in_emb_dim, dropout=0.03)
        
        self.to_out = nn.Sequential(
            nn.Linear(in_emb_dim, out_seq_emb_dim, bias=False))

    def forward(self,
                x,  # [b, n, c]
                input_pos,  # [b, n, 2]
                ):

        x = self.to_embedding(x)

        x1 = self.norm(self.h1(self.norm(x))) + x
        x1 = self.f1(x1)

        x2 = self.norm(self.h2(self.norm(x))) + x
        x2 = self.f2(x2)

        x3 = self.norm(self.h3(self.norm(x))) + x
        x3 = self.f3(x3)

        x4 = self.norm(self.h4(self.norm(x))) + x
        x4 = self.f4(x4)

        x5 = self.norm(self.h5(self.norm(x))) + x
        x5 = self.f5(x5)

        x6 = self.norm(self.h6(self.norm(x))) + x
        x6 = self.f6(x6)

        x7 = self.norm(self.h7(self.norm(x))) + x
        x7 = self.f7(x7)

        x8 = self.norm(self.h8(self.norm(x))) + x
        x8 = self.f8(x8)
        
        x = x1 + x2 + x3 + x4 + x5 + x6 +x7 +x8
        x = self.to_out(self.norm(x))
        
        return x


class Encoder1D(nn.Module):
    def __init__(self,
                 input_channels,           # how many channels
                 in_emb_dim,               # embedding dim of token                 (how about 512)
                 out_seq_emb_dim,          # embedding dim of encoded sequence      (how about 256)
                 depth,                    # depth of transformer / how many layers of attention    (4)
                 emb_dropout=0.05,           # dropout of embedding
                 res=2048,
                 ):
        super().__init__()

        # self.dropout = nn.Dropout(emb_dropout)

        self.to_embedding = nn.Sequential(
            nn.Linear(input_channels, in_emb_dim-1, bias=False),
        )

        self.h1 = HyenaOperator(d_model=in_emb_dim,l_max=4096)
        self.norm = nn.LayerNorm(in_emb_dim)
        self.f1 = FeedForward_q(in_emb_dim,in_emb_dim, dropout=0.03)
        
        self.h2 = HyenaOperator(d_model=in_emb_dim,l_max=4096)
        self.f2 = FeedForward_q(in_emb_dim,in_emb_dim, dropout=0.03)
        
        self.h3 = HyenaOperator(d_model=in_emb_dim,l_max=4096)
        self.f3 = FeedForward_q(in_emb_dim,in_emb_dim, dropout=0.03)
        
        self.h4 = HyenaOperator(d_model=in_emb_dim,l_max=4096)
        self.f4 = FeedForward_q(in_emb_dim,in_emb_dim, dropout=0.03)
        
        self.h5 = HyenaOperator(d_model=in_emb_dim,l_max=4096)
        self.f5 = FeedForward_q(in_emb_dim,in_emb_dim, dropout=0.03)
        
        self.h6 = HyenaOperator(d_model=in_emb_dim,l_max=4096)
        self.f6 = FeedForward_q(in_emb_dim,in_emb_dim, dropout=0.03)
        
        self.h7 = HyenaOperator(d_model=in_emb_dim,l_max=4096)
        self.f7 = FeedForward_q(in_emb_dim,in_emb_dim, dropout=0.03)
        
        self.h8 = HyenaOperator(d_model=in_emb_dim,l_max=4096)
        self.f8 = FeedForward_q(in_emb_dim,in_emb_dim, dropout=0.03)
        
        self.project_to_latent = nn.Sequential(
            nn.Linear(in_emb_dim, out_seq_emb_dim, bias=False))

    def forward(self,
                x,  # [b, n, c]
                input_pos,  # [b, n, 1]
                ):
        x = self.to_embedding(x)
        x = torch.cat((x, input_pos), dim=-1)
        x1 = self.norm(self.h1(self.norm(x))) + x
        x1 = self.f1(x1)

        x2 = self.norm(self.h2(self.norm(x))) + x
        x2 = self.f2(x2)

        x3 = self.norm(self.h3(self.norm(x))) + x
        x3 = self.f3(x3)

        x4 = self.norm(self.h4(self.norm(x))) + x
        x4 = self.f4(x4)

        x5 = self.norm(self.h5(self.norm(x))) + x
        x5 = self.f5(x5)

        x6 = self.norm(self.h6(self.norm(x))) + x
        x6 = self.f6(x6)

        x7 = self.norm(self.h7(self.norm(x))) + x
        x7 = self.f7(x7)

        x8 = self.norm(self.h8(self.norm(x))) + x
        x8 = self.f8(x8)
        
        x = x1 + x2 + x3 + x4 + x5 + x6 +x7 +x8
        x = self.project_to_latent(x)

        return x


