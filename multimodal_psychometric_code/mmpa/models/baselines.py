import torch
import torch.nn as nn
from .model import MultimodalPsychometricModel

class _Heads(nn.Module):
    def __init__(self,cfg,in_dim):
        super().__init__(); self.cfg=cfg
        self.shared=nn.Sequential(nn.Linear(in_dim,128),nn.GELU(),nn.Dropout(cfg.dropout))
        self.mean=nn.Linear(128,cfg.n_traits); self.logvar=nn.Linear(128,cfg.n_traits)
        self.ord=nn.Linear(128,cfg.n_traits*3); self.aux=nn.Linear(128,1)
    def forward(self,x):
        h=self.shared(x)
        return torch.sigmoid(self.mean(h)),torch.clamp(self.logvar(h),-6,3),self.ord(h).view(-1,self.cfg.n_traits,3),self.aux(h).squeeze(-1)

class EarlyConcatBaseline(MultimodalPsychometricModel):
    def __init__(self,cfg):
        super().__init__(cfg)
        self.early=nn.Sequential(nn.Linear(cfg.d_model*4,cfg.d_model),nn.GELU(),nn.LayerNorm(cfg.d_model))
    def encode_windows(self,b,modality_override=None):
        modal=self.encode_modalities(b); B,W,M,D=modal.shape
        avail=b['modality_mask'].clone()
        if modality_override is not None:
            keep=torch.tensor(modality_override,device=avail.device,dtype=avail.dtype).view(1,1,M); avail*=keep
        modal=modal*avail.unsqueeze(-1)
        z=self.early(modal.reshape(B,W,M*D))*b['window_mask'].unsqueeze(-1)
        gates=avail/avail.sum(-1,keepdim=True).clamp_min(1)
        return z,gates,modal

class AverageFusionBaseline(MultimodalPsychometricModel):
    def encode_windows(self,b,modality_override=None):
        modal=self.encode_modalities(b); avail=b['modality_mask'].clone()
        if modality_override is not None:
            keep=torch.tensor(modality_override,device=avail.device,dtype=avail.dtype).view(1,1,4); avail*=keep
        gates=avail/avail.sum(-1,keepdim=True).clamp_min(1)
        z=(modal*gates.unsqueeze(-1)).sum(2)*b['window_mask'].unsqueeze(-1)
        return z,gates,modal

class RecurrentFusionBaseline(AverageFusionBaseline):
    pass

class TransformerFusionBaseline(MultimodalPsychometricModel):
    """Uses modality-token self-attention without learned gating; equal masked pooling."""
    def encode_windows(self,b,modality_override=None):
        modal=self.encode_modalities(b); B,W,M,D=modal.shape; avail=b['modality_mask'].clone()
        if modality_override is not None:
            keep=torch.tensor(modality_override,device=avail.device,dtype=avail.dtype).view(1,1,M); avail*=keep
        key=(avail.reshape(B*W,M)<=0)
        att,_=self.fusion.attn(modal.reshape(B*W,M,D),modal.reshape(B*W,M,D),modal.reshape(B*W,M,D),key_padding_mask=key,need_weights=False)
        att=self.fusion.norm(modal.reshape(B*W,M,D)+att).reshape(B,W,M,D)
        gates=avail/avail.sum(-1,keepdim=True).clamp_min(1)
        z=(att*gates.unsqueeze(-1)).sum(2)*b['window_mask'].unsqueeze(-1)
        return z,gates,att

class LateFusionBaseline(MultimodalPsychometricModel):
    """Approximate late fusion via independently pooled modality streams then equal prediction-level mixing."""
    def __init__(self,cfg):
        super().__init__(cfg)
        self.mod_heads=nn.ModuleList([_Heads(cfg,cfg.d_model) for _ in range(4)])
    def forward(self,b,modality_override=None):
        modal=self.encode_modalities(b); avail=b['modality_mask']
        if modality_override is not None:
            keep=torch.tensor(modality_override,device=avail.device,dtype=avail.dtype).view(1,1,4); avail=avail*keep
        win=b['window_mask'].unsqueeze(-1).unsqueeze(-1)
        valid=avail.unsqueeze(-1)*win
        pooled=(modal*valid).sum(1)/(valid.sum(1).clamp_min(1))
        outs=[]
        for m in range(4): outs.append(self.mod_heads[m](pooled[:,m]))
        present=(avail.sum(1)>0).float(); weights=present/present.sum(1,keepdim=True).clamp_min(1)
        means=sum(weights[:,m:m+1]*outs[m][0] for m in range(4))
        logvars=sum(weights[:,m:m+1]*outs[m][1] for m in range(4))
        ords=sum(weights[:,m,None,None]*outs[m][2] for m in range(4))
        aux=sum(weights[:,m]*outs[m][3] for m in range(4))
        # auxiliary placeholders needed by common evaluator
        B,W=avail.shape[:2]; z=(modal*valid).sum(2)/(valid.sum(2).clamp_min(1)); z=z*b['window_mask'].unsqueeze(-1)
        window_mean=torch.sigmoid(self.window_trait(z))
        gates=avail/avail.sum(-1,keepdim=True).clamp_min(1)
        session=z.sum(1)/b['window_mask'].sum(1,keepdim=True).clamp_min(1)
        return {'mean':means,'logvar':logvars,'ordinal_logits':ords,'aux_logit':aux,'window_mean':window_mean,
                'z':z,'gates':gates,'modal_attended':modal,'session':session,
                'temporal_attention':b['window_mask']/b['window_mask'].sum(1,keepdim=True).clamp_min(1)}

BASELINES={
    'early_concat':EarlyConcatBaseline,
    'late_fusion':LateFusionBaseline,
    'multimodal_transformer':TransformerFusionBaseline,
    'recurrent_fusion':RecurrentFusionBaseline,
    'conventional_average':AverageFusionBaseline,
}
