import numpy as np
from scipy import stats


def paired_test(a,b,alpha=.05):
    a=np.asarray(a,float); b=np.asarray(b,float); d=a-b
    if len(d)<3: return {'test':'insufficient_n','p':np.nan,'effect':np.nan,'normality_p':np.nan}
    sw=stats.shapiro(d).pvalue
    if sw>=alpha:
        res=stats.ttest_rel(a,b); sd=d.std(ddof=1); dz=float(d.mean()/sd) if sd>0 else 0.0
        return {'test':'paired_t','p':float(res.pvalue),'effect_name':'Cohen_dz','effect':dz,'normality_p':float(sw)}
    res=stats.wilcoxon(a,b,zero_method='wilcox',alternative='two-sided')
    # rank-biserial from signed ranks
    nz=d[d!=0]; ranks=stats.rankdata(np.abs(nz)); rp=ranks[nz>0].sum(); rn=ranks[nz<0].sum(); den=rp+rn
    rb=float((rp-rn)/den) if den else 0.0
    return {'test':'wilcoxon','p':float(res.pvalue),'effect_name':'rank_biserial','effect':rb,'normality_p':float(sw)}
