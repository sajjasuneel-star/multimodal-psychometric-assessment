import numpy as np
from scipy.stats import norm


def gaussian_nll_np(y,mu,sigma):
    v=np.maximum(sigma**2,1e-8)
    return float(np.mean(.5*(np.log(2*np.pi*v)+(y-mu)**2/v)))

def fit_variance_scale(y,mu,sigma):
    scales=np.logspace(-1,1,200); vals=[gaussian_nll_np(y,mu,sigma*s) for s in scales]
    return float(scales[int(np.argmin(vals))])

def interval_coverage(y,mu,sigma,level):
    z=norm.ppf((1+level)/2); lo=mu-z*sigma; hi=mu+z*sigma
    return float(np.mean((y>=lo)&(y<=hi)))

def regression_ece(y,mu,sigma,levels=(.5,.6,.7,.8,.9,.95)):
    return float(np.mean([abs(interval_coverage(y,mu,sigma,l)-l) for l in levels]))

def uncertainty_metrics(y,mu,sigma):
    err=np.abs(y-mu).reshape(-1); sig=sigma.reshape(-1)
    corr=float(np.corrcoef(err,sig)[0,1]) if np.std(sig)>0 and np.std(err)>0 else 0.0
    return {'NLL':gaussian_nll_np(y,mu,sigma),'ECE':regression_ece(y,mu,sigma),
            'Coverage90':interval_coverage(y,mu,sigma,.90),'Coverage95':interval_coverage(y,mu,sigma,.95),
            'Sigma_Error_Correlation':corr}

def fit_temperature(logits,targets):
    # scalar temperature selected on validation ordinal NLL
    temps=np.linspace(.5,3.0,126); best=(1.0,1e9)
    for t in temps:
        x=logits/t; x=x-x.max(-1,keepdims=True); p=np.exp(x); p/=p.sum(-1,keepdims=True)
        ll=-np.log(np.take_along_axis(p,targets[...,None],axis=-1).squeeze(-1)+1e-12).mean()
        if ll<best[1]: best=(float(t),float(ll))
    return best[0]
