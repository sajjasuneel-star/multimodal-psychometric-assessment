from __future__ import annotations
import torch
import numpy as np

@torch.no_grad()
def _session_from_z(model,z,window_mask):
    session,temp=model.temporal(z,window_mask)
    h=model.shared(session); mean=torch.sigmoid(model.mean_head(h))
    return mean,temp

def integrated_gradients_windows(model,batch,trait_index=0,steps=32):
    """Integrated Gradients over fused window embeddings, yielding one normalized score per window."""
    model.eval()
    with torch.no_grad():
        z,_,_=model.encode_windows(batch)
    baseline=torch.zeros_like(z); total_grad=torch.zeros_like(z)
    for alpha in torch.linspace(0,1,steps,device=z.device):
        zi=(baseline+alpha*(z-baseline)).detach().requires_grad_(True)
        session,_=model.temporal(zi,batch['window_mask']); h=model.shared(session)
        pred=torch.sigmoid(model.mean_head(h))[:,trait_index].sum()
        grad=torch.autograd.grad(pred,zi,retain_graph=False)[0]
        total_grad += grad.detach()
    ig=(z-baseline)*(total_grad/steps)
    scores=ig.abs().sum(-1)*batch['window_mask']
    scores=scores/(scores.sum(-1,keepdim=True).clamp_min(1e-12))
    return scores.detach().cpu().numpy()
