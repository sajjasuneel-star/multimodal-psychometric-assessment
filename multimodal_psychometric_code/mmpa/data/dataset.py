from __future__ import annotations
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from mmpa.utils import stable_token_id

class HashTokenizer:
    def __init__(self, vocab_size=4096, max_tokens=24):
        self.vocab_size=vocab_size; self.max_tokens=max_tokens
    def encode(self, text):
        toks = re.findall(r"[A-Za-z']+|[0-9]+", text.lower())[:self.max_tokens]
        ids = [1] + [stable_token_id(t,self.vocab_size) for t in toks]
        ids = ids[:self.max_tokens]
        mask = [1]*len(ids)
        if len(ids)<self.max_tokens:
            pad=self.max_tokens-len(ids); ids += [0]*pad; mask += [0]*pad
        return np.asarray(ids,np.int64), np.asarray(mask,np.float32)

class SessionDataset(Dataset):
    def __init__(self, sessions, cfg):
        self.sessions=sessions; self.cfg=cfg; self.tok=HashTokenizer(cfg.vocab_size,cfg.max_tokens)
    def __len__(self): return len(self.sessions)
    def __getitem__(self, idx):
        s=self.sessions[idx]
        ids=[]; attn=[]
        for txt in s['text']:
            a,b=self.tok.encode(txt); ids.append(a); attn.append(b)
        return {
            'session_id': torch.tensor(s['session_id'],dtype=torch.long),
            'cohort_id': torch.tensor(s['cohort_id'],dtype=torch.long),
            'input_ids': torch.tensor(np.stack(ids),dtype=torch.long),
            'text_attn': torch.tensor(np.stack(attn),dtype=torch.float32),
            'audio': torch.tensor(s['audio'],dtype=torch.float32),
            'logs': torch.tensor(s['logs'],dtype=torch.float32),
            'visual': torch.tensor(s['visual'],dtype=torch.float32),
            'modality_mask': torch.tensor(s['mask'],dtype=torch.float32),
            'y_window': torch.tensor(s['y_window'],dtype=torch.float32),
            'y_session': torch.tensor(s['y_session'],dtype=torch.float32),
            'y_ord': torch.tensor(s['y_ord'],dtype=torch.long),
            'y_aux': torch.tensor(s['y_aux'],dtype=torch.float32),
            'length': torch.tensor(s['n_windows'],dtype=torch.long),
        }

def collate_sessions(batch):
    B=len(batch); max_w=max(int(x['length']) for x in batch); T=batch[0]['input_ids'].shape[-1]
    def z(shape,dtype=torch.float32): return torch.zeros(shape,dtype=dtype)
    out={
        'session_id': torch.stack([x['session_id'] for x in batch]),
        'cohort_id': torch.stack([x['cohort_id'] for x in batch]),
        'lengths': torch.stack([x['length'] for x in batch]),
        'input_ids': z((B,max_w,T),torch.long), 'text_attn': z((B,max_w,T)),
        'audio': z((B,max_w,batch[0]['audio'].shape[-1])),
        'logs': z((B,max_w,batch[0]['logs'].shape[-1])),
        'visual': z((B,max_w,batch[0]['visual'].shape[-1])),
        'modality_mask': z((B,max_w,4)), 'window_mask': z((B,max_w)),
        'y_window': z((B,max_w,batch[0]['y_window'].shape[-1])),
        'y_session': torch.stack([x['y_session'] for x in batch]),
        'y_ord': torch.stack([x['y_ord'] for x in batch]),
        'y_aux': torch.stack([x['y_aux'] for x in batch]),
    }
    for i,x in enumerate(batch):
        w=int(x['length']); out['window_mask'][i,:w]=1
        for k in ['input_ids','text_attn','audio','logs','visual','modality_mask','y_window']:
            out[k][i,:w]=x[k]
    return out
