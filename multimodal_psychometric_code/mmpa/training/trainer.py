from __future__ import annotations
import copy, math
import numpy as np
import torch
from torch.utils.data import DataLoader
from .losses import multitask_loss, info_nce
from mmpa.data.dataset import SessionDataset, collate_sessions
from mmpa.utils import get_device, seed_everything


def move_batch(b,device):
    return {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in b.items()}


def make_loader(sessions,cfg,shuffle=False):
    return DataLoader(SessionDataset(sessions,cfg),batch_size=cfg.batch_size,shuffle=shuffle,
                      num_workers=cfg.num_workers,collate_fn=collate_sessions)


def pretrain(model,train_sessions,cfg,device=None):
    device=device or get_device(cfg.device); model.to(device); model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
    loader=make_loader(train_sessions,cfg,True)
    history=[]
    for ep in range(cfg.pretrain_epochs):
        vals=[]
        for b in loader:
            b=move_batch(b,device); opt.zero_grad(set_to_none=True)
            z,_,_=model.encode_windows(b)
            loss=info_nce(z,b['window_mask'],model.pretrain_proj,cfg.contrastive_temperature)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip); opt.step(); vals.append(loss.item())
        history.append(float(np.mean(vals) if vals else 0))
    return history

@torch.no_grad()
def evaluate_loss(model,sessions,cfg,device=None):
    device=device or get_device(cfg.device); model.eval(); vals=[]
    for b in make_loader(sessions,cfg,False):
        b=move_batch(b,device); out=model(b); loss,_=multitask_loss(out,b,cfg); vals.append(loss.item())
    return float(np.mean(vals) if vals else np.inf)


def train_model(model,train_sessions,val_sessions,cfg,seed=42,device=None):
    seed_everything(seed); device=device or get_device(cfg.device); model.to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=3)
    loader=make_loader(train_sessions,cfg,True)
    best=math.inf; best_state=None; patience=0; hist=[]
    for ep in range(cfg.epochs):
        model.train(); batch_losses=[]
        for b in loader:
            b=move_batch(b,device); opt.zero_grad(set_to_none=True); out=model(b)
            loss,parts=multitask_loss(out,b,cfg); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip); opt.step(); batch_losses.append(parts['total'])
        val=evaluate_loss(model,val_sessions,cfg,device); scheduler.step(val)
        train=float(np.mean(batch_losses) if batch_losses else np.nan); hist.append({'epoch':ep+1,'train_loss':train,'val_loss':val})
        if val < best-1e-5:
            best=val; best_state=copy.deepcopy(model.state_dict()); patience=0
        else:
            patience+=1
            if patience>=cfg.early_stopping_patience: break
    if best_state is not None: model.load_state_dict(best_state)
    return model,hist

@torch.no_grad()
def predict(model,sessions,cfg,device=None,modality_override=None):
    device=device or get_device(cfg.device); model.to(device); model.eval(); rows=[]
    for b in make_loader(sessions,cfg,False):
        ids=b['session_id'].numpy(); cohorts=b['cohort_id'].numpy(); y=b['y_session'].numpy(); yo=b['y_ord'].numpy(); ya=b['y_aux'].numpy()
        b=move_batch(b,device); out=model(b,modality_override=modality_override)
        mean=out['mean'].cpu().numpy(); sigma=np.exp(.5*out['logvar'].cpu().numpy()); logits=out['ordinal_logits'].cpu().numpy(); aux=out['aux_logit'].cpu().numpy(); gates=out['gates'].cpu().numpy()
        for i in range(len(ids)):
            L=int(b['lengths'][i].item())
            rows.append({'session_id':int(ids[i]),'cohort_id':int(cohorts[i]),'y_true':y[i], 'y_pred':mean[i],
                         'sigma':sigma[i],'y_ord':yo[i],'ord_logits':logits[i],'y_aux':float(ya[i]),'aux_logit':float(aux[i]),
                         'mean_gates':gates[i,:L].mean(0)})
    return rows
