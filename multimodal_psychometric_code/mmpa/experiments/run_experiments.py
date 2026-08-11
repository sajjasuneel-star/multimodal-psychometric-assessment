from __future__ import annotations
import copy, json, os, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from scipy.special import expit

from mmpa.config import Config
from mmpa.utils import ensure_dir, save_json, get_device, seed_everything
from mmpa.data.generator import generate_sessions, save_sessions
from mmpa.data.dataset import SessionDataset, collate_sessions
from mmpa.models.model import MultimodalPsychometricModel
from mmpa.models.baselines import BASELINES
from mmpa.training.trainer import pretrain, train_model, predict, make_loader, move_batch
from mmpa.evaluation.metrics import predictive_metrics, rows_to_arrays
from mmpa.evaluation.uncertainty import fit_variance_scale, uncertainty_metrics, fit_temperature
from mmpa.evaluation.statistics import paired_test
from mmpa.evaluation.leakage import univariate_screen, permutation_baseline
from mmpa.evaluation.psychometrics import construct_quality, htmt_matrix, measurement_invariance_proxy
from mmpa.explainability.integrated_gradients import integrated_gradients_windows
from mmpa.explainability.intervention import generate_intervention


def split_train_val(sessions, test_cohort, seed):
    pool=[s for s in sessions if s['cohort_id']!=test_cohort]
    test=[s for s in sessions if s['cohort_id']==test_cohort]
    tr,va=train_test_split(pool,test_size=.2,random_state=seed,shuffle=True)
    return tr,va,test


def calibrate_rows(val_rows,test_rows):
    yv,pv,sv,yov,lov,_,_=rows_to_arrays(val_rows)
    scale=fit_variance_scale(yv,pv,sv)
    temp=fit_temperature(lov,yov)
    out=[]
    for r in test_rows:
        q=dict(r); q['sigma']=r['sigma']*scale; q['ord_logits']=r['ord_logits']/temp; out.append(q)
    return out,{'variance_scale':scale,'ordinal_temperature':temp}


def flatten_summary(df,aux,unc,model_name,seed,fold):
    row={'model':model_name,'seed':seed,'held_out_cohort':fold,
         'MAE_mean':float(df.MAE.mean()),'RMSE_mean':float(df.RMSE.mean()),'R2_mean':float(df.R2.mean()),
         'Ordinal_Accuracy_mean':float(df.Ordinal_Accuracy.mean()),'Ordinal_MacroF1_mean':float(df.Ordinal_MacroF1.mean()),
         'Weighted_Kappa_mean':float(df.Weighted_Kappa.mean()),
         'Aux_Precision':aux['Precision'],'Aux_Recall':aux['Recall'],'Aux_F1':aux['F1']}
    row.update({f'Uncertainty_{k}':v for k,v in unc.items()})
    return row


def train_eval_one(model_cls,cfg,tr,va,te,seed,name,do_pretrain=True):
    device=get_device(cfg.device); seed_everything(seed); model=model_cls(cfg)
    pre_hist=[]
    if do_pretrain and hasattr(model,'pretrain_proj') and cfg.pretrain_epochs>0 and name!='late_fusion':
        pre_hist=pretrain(model,tr,cfg,device)
    model,hist=train_model(model,tr,va,cfg,seed,device)
    val_rows=predict(model,va,cfg,device); test_rows=predict(model,te,cfg,device)
    calibrated,calib=calibrate_rows(val_rows,test_rows)
    metric_df,aux=predictive_metrics(calibrated,cfg.trait_names)
    y,p,s,_,_,_,_=rows_to_arrays(calibrated); unc=uncertainty_metrics(y,p,s)
    return model,calibrated,metric_df,aux,unc,calib,hist,pre_hist


def save_rows(rows,path,cfg):
    rec=[]
    for r in rows:
        d={'session_id':r['session_id'],'cohort_id':r['cohort_id'],'y_aux':r['y_aux'],'aux_logit':r['aux_logit']}
        for k,n in enumerate(cfg.trait_names):
            d[f'{n}_true']=r['y_true'][k]; d[f'{n}_pred']=r['y_pred'][k]; d[f'{n}_sigma']=r['sigma'][k]; d[f'{n}_ord_true']=r['y_ord'][k]; d[f'{n}_ord_pred']=int(np.argmax(r['ord_logits'][k]))
        for m,n in enumerate(['text','audio','logs','visual']): d[f'gate_{n}']=r['mean_gates'][m]
        rec.append(d)
    pd.DataFrame(rec).to_csv(path,index=False)


def make_figures(summary_df,pred_df,outdir):
    import matplotlib.pyplot as plt
    ensure_dir(outdir)
    if len(summary_df):
        agg=summary_df.groupby('model',as_index=False)['MAE_mean'].mean().sort_values('MAE_mean')
        fig,ax=plt.subplots(figsize=(8,5)); ax.bar(agg.model,agg.MAE_mean); ax.set_ylabel('Mean MAE'); ax.set_xlabel('Model'); ax.tick_params(axis='x',rotation=35); fig.tight_layout(); fig.savefig(Path(outdir)/'model_mae_comparison.png',dpi=300); plt.close(fig)
    gate_cols=[c for c in pred_df.columns if c.startswith('gate_')]
    if gate_cols:
        g=pred_df[gate_cols].mean(); fig,ax=plt.subplots(figsize=(6,4)); ax.bar([x.replace('gate_','') for x in gate_cols],g.values); ax.set_ylabel('Mean learned contribution'); ax.set_xlabel('Modality'); fig.tight_layout(); fig.savefig(Path(outdir)/'modality_contributions.png',dpi=300); plt.close(fig)


def run_ablation_inference(model,test_sessions,cfg,seed,fold,outdir):
    configs={'full':[1,1,1,1],'text_only':[1,0,0,0],'text_audio_logs':[1,1,1,0],
             'no_audio':[1,0,1,1],'no_logs':[1,1,0,1],'no_visual':[1,1,1,0]}
    rows=[]
    for name,mask in configs.items():
        pr=predict(model,test_sessions,cfg,modality_override=mask)
        df,aux=predictive_metrics(pr,cfg.trait_names)
        rows.append({'seed':seed,'held_out_cohort':fold,'ablation':name,'MAE_mean':df.MAE.mean(),'R2_mean':df.R2.mean(),'MacroF1_mean':df.Ordinal_MacroF1.mean(),'Aux_F1':aux['F1']})
    return rows


def stability_sensitivity(cfg,sessions,seed,fold,values=None):
    values=values or [0,.01,.05,.10,.20,.50]; tr,va,te=split_train_val(sessions,fold,seed); rows=[]
    for lam in values:
        c=copy.deepcopy(cfg); c.lambda_stability=lam
        # sensitivity is a controlled supervised retraining; skip contrastive pretraining for tractability and isolation.
        model=MultimodalPsychometricModel(c); model,_=train_model(model,tr,va,c,seed)
        pr=predict(model,te,c); df,aux=predictive_metrics(pr,c.trait_names)
        # trajectory smoothness from predictions
        sm=[]; model.eval(); device=get_device(c.device)
        with torch.no_grad():
            for b in make_loader(te,c,False):
                b=move_batch(b,device); o=model.to(device)(b); d=(o['window_mean'][:,1:]-o['window_mean'][:,:-1]).abs().mean(-1); m=b['window_mask'][:,1:]*b['window_mask'][:,:-1]; sm.append(float((d*m).sum().cpu()/m.sum().clamp_min(1).cpu()))
        rows.append({'lambda_stability':lam,'MAE_mean':df.MAE.mean(),'R2_mean':df.R2.mean(),'trajectory_delta':float(np.mean(sm))})
    return rows


def explain_one(model,test_sessions,cfg,outdir):
    device=get_device(cfg.device); loader=make_loader(test_sessions[:1],cfg,False); b=move_batch(next(iter(loader)),device); model=model.to(device)
    with torch.no_grad(): out=model(b)
    explanations=[]
    for k,t in enumerate(cfg.trait_names):
        scores=integrated_gradients_windows(model,b,k,steps=16 if cfg.mode=='quick' else 32)[0]
        L=int(b['lengths'][0]); top=np.argsort(scores[:L])[::-1][:5]
        explanations.append({'trait':t,'top_windows':[{'window':int(i),'score':float(scores[i])} for i in top]})
    traits=out['mean'][0].detach().cpu().numpy(); sigma=np.exp(.5*out['logvar'][0].detach().cpu().numpy())
    intervention=generate_intervention(traits,sigma)
    save_json({'session_id':int(b['session_id'][0]),'trait_predictions':dict(zip(cfg.trait_names,traits.tolist())),
               'uncertainty':dict(zip(cfg.trait_names,sigma.tolist())),'integrated_gradients':explanations,'interventions':intervention},
              Path(outdir)/'explainability_case.json')


def run(mode='quick',output_dir='outputs',include_baselines=True,include_sensitivity=None):
    cfg=Config.quick() if mode=='quick' else Config(); cfg.output_dir=output_dir; cfg.mode=mode
    root=Path(output_dir); ensure_dir(root); ensure_dir(root/'predictions'); ensure_dir(root/'tables'); ensure_dir(root/'models'); ensure_dir(root/'figures')
    save_json(cfg.to_dict(),root/'config.json')

    # Phase 1: deterministic synthetic multimodal data generation
    sessions=generate_sessions(cfg,cfg.seeds[0]); save_sessions(sessions,root/'synthetic_sessions.pt')

    # Phase 2: leakage diagnostics before model training
    screen=univariate_screen(sessions,cfg.trait_names); screen.to_csv(root/'tables'/'univariate_leakage_screen.csv',index=False)
    save_json(permutation_baseline(sessions,cfg.seeds[0]),root/'tables'/'permutation_linear_diagnostic.json')

    # Phase 3: construct-level psychometric diagnostics on the synthetic measurement indicators
    cq,scores,idf=construct_quality(sessions); cq.to_csv(root/'tables'/'construct_quality.csv',index=False)
    htmt_matrix(sessions).to_csv(root/'tables'/'htmt_matrix.csv')
    measurement_invariance_proxy(sessions).to_csv(root/'tables'/'measurement_invariance_proxy.csv',index=False)

    summary=[]; all_pred=[]; ablations=[]; model_cache=None; test_cache=None
    seeds=cfg.seeds if mode=='paper' else [cfg.seeds[0]]
    folds=list(range(cfg.n_cohorts)) if mode=='paper' else [0]
    model_set={'proposed':MultimodalPsychometricModel}
    if include_baselines: model_set.update(BASELINES)

    # Phases 4-10: representation learning, fusion, temporal aggregation, multitask inference,
    # calibration, uncertainty and baseline comparison under cohort-held-out evaluation.
    for seed in seeds:
        for fold in folds:
            tr,va,te=split_train_val(sessions,fold,seed)
            for name,cls in model_set.items():
                print(f'[{mode}] seed={seed} fold={fold} model={name}',flush=True)
                model,rows,mdf,aux,unc,calib,hist,preh=train_eval_one(cls,cfg,tr,va,te,seed,name,do_pretrain=(name=='proposed'))
                s=flatten_summary(mdf,aux,unc,name,seed,fold); s.update(calib); summary.append(s)
                mdf.assign(model=name,seed=seed,held_out_cohort=fold).to_csv(root/'tables'/f'metrics_{name}_s{seed}_c{fold}.csv',index=False)
                save_rows(rows,root/'predictions'/f'{name}_s{seed}_c{fold}.csv',cfg)
                pd.DataFrame(hist).to_csv(root/'tables'/f'training_{name}_s{seed}_c{fold}.csv',index=False)
                if name=='proposed':
                    torch.save(model.state_dict(),root/'models'/f'proposed_s{seed}_c{fold}.pt')
                    ablations += run_ablation_inference(model,te,cfg,seed,fold,root)
                    if model_cache is None: model_cache=model; test_cache=te
                for r in rows:
                    rr={'model':name,'seed':seed,'fold':fold,'session_id':r['session_id']}
                    rr.update({f'gate_{m}':r['mean_gates'][j] for j,m in enumerate(['text','audio','logs','visual'])}); all_pred.append(rr)

    summary_df=pd.DataFrame(summary); summary_df.to_csv(root/'tables'/'experiment_summary.csv',index=False)
    pd.DataFrame(ablations).to_csv(root/'tables'/'modality_ablation.csv',index=False)

    # Phase 11: paired statistical comparison using model-level fold/seed MAE summaries.
    stats_rows=[]
    if 'proposed' in summary_df.model.unique():
        p=summary_df[summary_df.model=='proposed'].sort_values(['seed','held_out_cohort'])
        for name in [x for x in summary_df.model.unique() if x!='proposed']:
            q=summary_df[summary_df.model==name].sort_values(['seed','held_out_cohort'])
            if len(p)==len(q) and len(p)>0:
                st=paired_test(p.MAE_mean.values,q.MAE_mean.values); st.update({'comparison':f'proposed_vs_{name}','metric':'MAE_mean'}); stats_rows.append(st)
    pd.DataFrame(stats_rows).to_csv(root/'tables'/'paired_statistical_tests.csv',index=False)

    # Phase 12: stability-coefficient sensitivity.
    if include_sensitivity is None: include_sensitivity=(mode=='paper')
    if include_sensitivity:
        vals=[0,.10] if mode=='quick' else [0,.01,.05,.10,.20,.50]
        sens=stability_sensitivity(cfg,sessions,seeds[0],folds[0],vals)
        pd.DataFrame(sens).to_csv(root/'tables'/'stability_sensitivity.csv',index=False)

    # Phase 13: local explainability + rule-based uncertainty-aware intervention.
    if model_cache is not None and test_cache:
        explain_one(model_cache,test_cache,cfg,root)

    pred_df=pd.DataFrame(all_pred); make_figures(summary_df,pred_df,root/'figures')
    print(f'Completed. Outputs: {root.resolve()}')
    return summary_df


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['quick','paper'],default='quick')
    ap.add_argument('--output-dir',default='outputs')
    ap.add_argument('--no-baselines',action='store_true')
    ap.add_argument('--sensitivity',action='store_true',help='Run stability sensitivity even in quick mode.')
    a=ap.parse_args()
    run(a.mode,a.output_dir,not a.no_baselines,True if a.sensitivity else None)

if __name__=='__main__': main()
