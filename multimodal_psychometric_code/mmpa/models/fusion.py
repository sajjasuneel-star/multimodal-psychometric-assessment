import torch
import torch.nn as nn

class CrossModalGatedFusion(nn.Module):
    def __init__(self,d_model,n_heads=4,dropout=.1):
        super().__init__()
        self.attn=nn.MultiheadAttention(d_model,n_heads,dropout=dropout,batch_first=True)
        self.norm=nn.LayerNorm(d_model)
        self.gate=nn.Sequential(nn.Linear(d_model,d_model//2),nn.GELU(),nn.Linear(d_model//2,1))
    def forward(self, modal, availability):
        # modal [N,M,D], availability [N,M]
        key_padding=(availability<=0)
        att,_=self.attn(modal,modal,modal,key_padding_mask=key_padding,need_weights=False)
        att=self.norm(modal+att)
        scores=self.gate(att).squeeze(-1)
        scores=scores.masked_fill(key_padding,-1e9)
        weights=torch.softmax(scores,dim=-1)
        fused=(weights.unsqueeze(-1)*att).sum(dim=1)
        return fused, att, weights

class TemporalAggregator(nn.Module):
    def __init__(self,d_model,hidden,dropout=.1):
        super().__init__()
        self.gru=nn.GRU(d_model,hidden,batch_first=True,bidirectional=True)
        self.score=nn.Sequential(nn.Linear(hidden*2,hidden),nn.Tanh(),nn.Linear(hidden,1))
        self.proj=nn.Sequential(nn.Linear(hidden*2,d_model),nn.LayerNorm(d_model),nn.Dropout(dropout))
    def forward(self,z,window_mask):
        h,_=self.gru(z)
        scores=self.score(h).squeeze(-1).masked_fill(window_mask<=0,-1e9)
        a=torch.softmax(scores,dim=-1)
        s=(a.unsqueeze(-1)*h).sum(1)
        return self.proj(s), a
