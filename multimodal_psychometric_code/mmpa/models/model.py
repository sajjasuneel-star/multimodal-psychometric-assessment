import torch
import torch.nn as nn
from .encoders import TextWindowEncoder, MLPEncoder, LogSequenceEncoder
from .fusion import CrossModalGatedFusion, TemporalAggregator

class MultimodalPsychometricModel(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg
        self.text=TextWindowEncoder(cfg.vocab_size,cfg.text_embed_dim,cfg.d_model,cfg.n_heads,cfg.dropout)
        self.audio=MLPEncoder(cfg.audio_dim,cfg.d_model,cfg.dropout)
        self.logs=LogSequenceEncoder(cfg.log_dim,cfg.d_model,cfg.dropout)
        self.visual=MLPEncoder(cfg.visual_dim,cfg.d_model,cfg.dropout)
        self.fusion=CrossModalGatedFusion(cfg.d_model,cfg.n_heads,cfg.dropout)
        self.temporal=TemporalAggregator(cfg.d_model,cfg.temporal_hidden,cfg.dropout)
        self.window_trait=nn.Sequential(nn.Linear(cfg.d_model,64),nn.GELU(),nn.Linear(64,cfg.n_traits))
        self.shared=nn.Sequential(nn.Linear(cfg.d_model,128),nn.GELU(),nn.Dropout(cfg.dropout))
        self.mean_head=nn.Linear(128,cfg.n_traits)
        self.logvar_head=nn.Linear(128,cfg.n_traits)
        self.ordinal_head=nn.Linear(128,cfg.n_traits*3)
        self.aux_head=nn.Linear(128,1)
        self.pretrain_proj=nn.Sequential(nn.Linear(cfg.d_model,cfg.d_model),nn.GELU(),nn.Linear(cfg.d_model,64))

    def encode_modalities(self,b):
        B,W,T=b['input_ids'].shape; N=B*W
        te=self.text(b['input_ids'].reshape(N,T),b['text_attn'].reshape(N,T)).reshape(B,W,-1)
        ae=self.audio(b['audio'].reshape(N,-1)).reshape(B,W,-1)
        le=self.logs(b['logs'].reshape(N,-1)).reshape(B,W,-1)
        ve=self.visual(b['visual'].reshape(N,-1)).reshape(B,W,-1)
        return torch.stack([te,ae,le,ve],dim=2)

    def encode_windows(self,b, modality_override=None):
        modal=self.encode_modalities(b); B,W,M,D=modal.shape
        avail=b['modality_mask'].clone()
        # Padded temporal positions have no real modality; keep a dummy text token unmasked
        # to avoid all-masked MultiheadAttention, then zero the fused output by window_mask.
        padded=(b['window_mask']<=0)
        if padded.any(): avail[...,0]=torch.where(padded,torch.ones_like(avail[...,0]),avail[...,0])
        if modality_override is not None:
            keep=torch.tensor(modality_override,device=avail.device,dtype=avail.dtype).view(1,1,M)
            avail=avail*keep
            # text must remain available to prevent all-masked attention
            if keep[...,0].item()==0: avail[...,0]=1
        z,att,gates=self.fusion(modal.reshape(B*W,M,D),avail.reshape(B*W,M))
        z=z.reshape(B,W,D); gates=gates.reshape(B,W,M); att=att.reshape(B,W,M,D)
        z=z*b['window_mask'].unsqueeze(-1)
        return z,gates,att

    def forward(self,b, modality_override=None):
        z,gates,att=self.encode_windows(b,modality_override)
        session,temp_att=self.temporal(z,b['window_mask'])
        h=self.shared(session)
        mean=torch.sigmoid(self.mean_head(h))
        logvar=torch.clamp(self.logvar_head(h),-6,3)
        ordinal=self.ordinal_head(h).view(-1,self.cfg.n_traits,3)
        aux=self.aux_head(h).squeeze(-1)
        window_mean=torch.sigmoid(self.window_trait(z))
        return {'mean':mean,'logvar':logvar,'ordinal_logits':ordinal,'aux_logit':aux,
                'window_mean':window_mean,'z':z,'gates':gates,'modal_attended':att,
                'session':session,'temporal_attention':temp_att}
