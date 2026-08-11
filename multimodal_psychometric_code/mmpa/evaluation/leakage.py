import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr


def session_feature_matrix(sessions):
    X=[]; Y=[]; names=[]
    names += [f'audio_{i}' for i in range(sessions[0]['audio'].shape[1])]
    names += [f'log_{i}' for i in range(sessions[0]['logs'].shape[1])]
    names += [f'visual_{i}' for i in range(sessions[0]['visual'].shape[1])]
    for s in sessions:
        X.append(np.r_[s['audio'].mean(0),s['logs'].mean(0),s['visual'].mean(0)])
        Y.append(s['y_session'])
    return np.asarray(X),np.asarray(Y),names

def univariate_screen(sessions,trait_names):
    X,Y,names=session_feature_matrix(sessions); rows=[]
    for j,n in enumerate(names):
        for k,t in enumerate(trait_names):
            pr=pearsonr(X[:,j],Y[:,k]).statistic if np.std(X[:,j]) and np.std(Y[:,k]) else 0
            sr=spearmanr(X[:,j],Y[:,k]).statistic if np.std(X[:,j]) and np.std(Y[:,k]) else 0
            lr=LinearRegression().fit(X[:,[j]],Y[:,k]); r2=r2_score(Y[:,k],lr.predict(X[:,[j]]))
            rows.append({'feature':n,'trait':t,'pearson_r':pr,'spearman_rho':sr,'single_feature_R2':r2})
    return pd.DataFrame(rows)

def permutation_baseline(sessions,seed=42):
    X,Y,_=session_feature_matrix(sessions); rng=np.random.default_rng(seed); Yp=Y.copy(); rng.shuffle(Yp,axis=0)
    # 70/30 quick diagnostic using linear model; full permutation retraining is implemented in experiment runner.
    n=int(.7*len(X)); model=LinearRegression().fit(X[:n],Yp[:n]); pred=model.predict(X[n:])
    return {'permuted_linear_R2':float(r2_score(Yp[n:],pred,multioutput='uniform_average'))}
