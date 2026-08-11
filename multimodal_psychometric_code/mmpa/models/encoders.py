import math
import torch
import torch.nn as nn

class TextWindowEncoder(nn.Module):
    """Self-contained Transformer text encoder; no external model download required."""
    def __init__(self, vocab_size, token_dim, d_model, n_heads=4, dropout=.1):
        super().__init__()
        self.emb=nn.Embedding(vocab_size, token_dim, padding_idx=0)
        layer=nn.TransformerEncoderLayer(d_model=token_dim,nhead=max(1,min(n_heads,token_dim//16)),
                                         dim_feedforward=token_dim*4,dropout=dropout,batch_first=True,
                                         activation='gelu',norm_first=True)
        self.tr=nn.TransformerEncoder(layer,num_layers=2)
        self.proj=nn.Sequential(nn.Linear(token_dim,d_model),nn.LayerNorm(d_model))
    def forward(self, ids, attn):
        # ids [N,T]
        empty=(attn.sum(1)<=0)
        ids_safe=ids.clone(); attn_safe=attn.clone()
        if empty.any():
            ids_safe[empty,0]=1; attn_safe[empty,0]=1
        x=self.emb(ids_safe)
        x=self.tr(x,src_key_padding_mask=(attn_safe<=0))
        denom=attn_safe.sum(1,keepdim=True).clamp_min(1)
        pooled=(x*attn_safe.unsqueeze(-1)).sum(1)/denom
        if empty.any(): pooled[empty]=0
        return self.proj(pooled)

class MLPEncoder(nn.Module):
    def __init__(self, in_dim, d_model, dropout=.1):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(in_dim,64),nn.GELU(),nn.Dropout(dropout),
                               nn.Linear(64,d_model),nn.LayerNorm(d_model))
    def forward(self,x): return self.net(x)

class LogSequenceEncoder(nn.Module):
    """Window-level log projection; temporal dependencies are handled at session level."""
    def __init__(self,in_dim,d_model,dropout=.1):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(in_dim,64),nn.GELU(),nn.Dropout(dropout),
                               nn.Linear(64,d_model),nn.LayerNorm(d_model))
    def forward(self,x): return self.net(x)
