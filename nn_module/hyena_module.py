from .hyenaoperator import HyenaOperator
import torch
import torch.nn as nn
ACTIVATION_REGISTRY = {
    "relu": nn.ReLU(),
    "silu": nn.SiLU(),
    "gelu": nn.GELU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
}

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*2,dim),
        )
    def forward(self, x):
        return self.net(x)

class hyena1d(nn.Module):
    def __init__(self,in_emb_dim):
        super().__init__()
        self.h1 = HyenaOperator(d_model=in_emb_dim,l_max=4096)
        self.h2 = HyenaOperator(d_model=in_emb_dim,l_max=4096)
        self.h3 = HyenaOperator(d_model=in_emb_dim,l_max=4096)
        self.h4 = HyenaOperator(d_model=in_emb_dim,l_max=4096)
        self.h5 = HyenaOperator(d_model=in_emb_dim,l_max=4096)
        self.h6 = HyenaOperator(d_model=in_emb_dim,l_max=4096)
        self.h7 = HyenaOperator(d_model=in_emb_dim,l_max=4096)
        self.h8 = HyenaOperator(d_model=in_emb_dim,l_max=4096)
        self.norm = nn.InstanceNorm1d(4096)
        self.f1 = FeedForward(in_emb_dim,in_emb_dim, dropout=0.03)
        self.f2 = FeedForward(in_emb_dim,in_emb_dim, dropout=0.03)
        self.f3 = FeedForward(in_emb_dim,in_emb_dim, dropout=0.03)
        self.f4 = FeedForward(in_emb_dim,in_emb_dim, dropout=0.03)
        self.f5 = FeedForward(in_emb_dim,in_emb_dim, dropout=0.03)
        self.f6 = FeedForward(in_emb_dim,in_emb_dim, dropout=0.03)
        self.f7 = FeedForward(in_emb_dim,in_emb_dim, dropout=0.03)
        self.f8 = FeedForward(in_emb_dim,in_emb_dim, dropout=0.03)
    
    def forward(self,x):
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
        x = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8
        return x