from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Observable indicator mapping mirrors the manuscript's construct operationalization.
INDICATORS = {
    'engagement': [('audio',0),('audio',2),('audio',7),('logs',0),('visual',3)],
    'collaboration_quality': [('logs',2),('logs',3),('logs',5),('logs',6),('audio',6)],
    'socio_emotional_presence': [('audio',3),('audio',4),('visual',0),('visual',1),('visual',5)],
    'cognitive_participation': [('logs',1),('logs',0),('audio',2),('logs',6),('audio',0)],
    'participation_balance': [('logs',4),('logs',5),('logs',7),('audio',5),('audio',6)],
}

def _indicator_df(sessions):
    data={}
    for cname,inds in INDICATORS.items():
        for q,(mod,j) in enumerate(inds,1):
            vals=[]
            for s in sessions:
                arr=s[mod]
                vals.append(float(arr[:,j].mean()))
            data[f'{cname}__i{q}']=vals
    data['cohort_id']=[s['cohort_id'] for s in sessions]
    return pd.DataFrame(data)

def _loadings_and_scores(df,cols):
    X=StandardScaler().fit_transform(df[cols].values)
    pca=PCA(n_components=1).fit(X)
    load=np.abs(pca.components_[0])
    score=pca.transform(X).ravel()
    return load,score

def construct_quality(sessions):
    df=_indicator_df(sessions); rows=[]; scores={}; loads={}
    for c in INDICATORS:
        cols=[x for x in df.columns if x.startswith(c+'__')]
        l,s=_loadings_and_scores(df,cols); loads[c]=l; scores[c]=s
        err=1-l**2
        cr=float((l.sum()**2)/((l.sum()**2)+err.sum()+1e-12))
        ave=float(np.mean(l**2))
        rows.append({'construct':c,'Composite_Reliability':cr,'AVE':ave})
    score_df=pd.DataFrame(scores)
    return pd.DataFrame(rows),score_df,df

def htmt_matrix(sessions):
    _,_,df=construct_quality(sessions); cs=list(INDICATORS); out=pd.DataFrame(np.eye(len(cs)),index=cs,columns=cs)
    for ia,a in enumerate(cs):
        ca=[c for c in df.columns if c.startswith(a+'__')]
        for ib,b in enumerate(cs):
            if ib<=ia: continue
            cb=[c for c in df.columns if c.startswith(b+'__')]
            heter=[]
            for x in ca:
                for y in cb: heter.append(abs(np.corrcoef(df[x],df[y])[0,1]))
            mono_a=[abs(np.corrcoef(df[ca[i]],df[ca[j]])[0,1]) for i in range(len(ca)) for j in range(i+1,len(ca))]
            mono_b=[abs(np.corrcoef(df[cb[i]],df[cb[j]])[0,1]) for i in range(len(cb)) for j in range(i+1,len(cb))]
            den=np.sqrt(max(np.mean(mono_a),1e-12)*max(np.mean(mono_b),1e-12))
            v=float(np.mean(heter)/den); out.loc[a,b]=out.loc[b,a]=v
    return out

def measurement_invariance_proxy(sessions):
    """Reproducible computational invariance diagnostic.
    It reports cross-cohort loading and intercept drift. A strict SEM/CFA implementation can
    optionally be substituted; this proxy remains dependency-free and is explicit about its scope.
    """
    df=_indicator_df(sessions); cohorts=sorted(df.cohort_id.unique()); rows=[]
    ref=cohorts[0]
    ref_load={}; ref_mean={}
    for c in INDICATORS:
        cols=[x for x in df.columns if x.startswith(c+'__')]
        ref_load[c]=_loadings_and_scores(df[df.cohort_id==ref],cols)[0]
        ref_mean[c]=df[df.cohort_id==ref][cols].mean().values
    for co in cohorts[1:]:
        metric=[]; scalar=[]
        for c in INDICATORS:
            cols=[x for x in df.columns if x.startswith(c+'__')]
            l,_=_loadings_and_scores(df[df.cohort_id==co],cols)
            metric.extend(np.abs(l-ref_load[c]))
            scalar.extend(np.abs(df[df.cohort_id==co][cols].mean().values-ref_mean[c]))
        rows.append({'reference_cohort':int(ref),'comparison_cohort':int(co),
                     'mean_abs_loading_drift_metric':float(np.mean(metric)),
                     'mean_abs_intercept_drift_scalar':float(np.mean(scalar)),
                     'configural_structure':'same_5_construct_indicator_map'})
    return pd.DataFrame(rows)
