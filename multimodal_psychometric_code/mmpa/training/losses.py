import torch
import torch.nn.functional as F


def gaussian_nll(mean,logvar,target):
    inv=torch.exp(-logvar)
    return 0.5*((target-mean)**2*inv+logvar).mean()

def ordinal_loss(logits,target):
    B,K,C=logits.shape
    ce=F.cross_entropy(logits.reshape(B*K,C),target.reshape(B*K))
    probs=torch.softmax(logits,-1)
    classes=torch.arange(C,device=logits.device,dtype=probs.dtype)
    expected=(probs*classes).sum(-1)
    dist=((expected-target.float())**2).mean()
    return ce+0.05*dist

def stability_loss(window_mean,window_mask):
    if window_mean.shape[1]<2: return window_mean.sum()*0
    d=(window_mean[:,1:]-window_mean[:,:-1])**2
    m=(window_mask[:,1:]*window_mask[:,:-1]).unsqueeze(-1)
    return (d*m).sum()/m.sum().clamp_min(1)

def multitask_loss(out,b,cfg):
    reg=gaussian_nll(out['mean'],out['logvar'],b['y_session'])
    ordl=ordinal_loss(out['ordinal_logits'],b['y_ord'])
    aux=F.binary_cross_entropy_with_logits(out['aux_logit'],b['y_aux'])
    stab=stability_loss(out['window_mean'],b['window_mask'])
    total=cfg.lambda_reg*reg+cfg.lambda_ord*ordl+cfg.lambda_aux*aux+cfg.lambda_stability*stab
    return total,{'reg':reg.item(),'ord':ordl.item(),'aux':aux.item(),'stability':stab.item(),'total':total.item()}

def info_nce(z,window_mask,projection,temperature=.07):
    # Adjacent valid windows are positives. Negatives are all valid embeddings in the mini-batch.
    B,W,D=z.shape
    p=projection(z)
    p=F.normalize(p,dim=-1)
    anchors=[]; positives=[]
    for i in range(B):
        L=int(window_mask[i].sum().item())
        if L>=2:
            anchors.append(p[i,:L-1]); positives.append(p[i,1:L])
    if not anchors:
        return z.sum()*0
    a=torch.cat(anchors,0); q=torch.cat(positives,0)
    # symmetric in-batch contrastive objective
    logits=a@q.T/temperature
    labels=torch.arange(len(a),device=z.device)
    return 0.5*(F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels))
