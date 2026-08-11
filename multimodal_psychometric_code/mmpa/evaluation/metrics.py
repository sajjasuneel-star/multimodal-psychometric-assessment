import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score, cohen_kappa_score, precision_score, recall_score
from scipy.special import softmax, expit


def rows_to_arrays(rows):
    y=np.stack([r['y_true'] for r in rows]); p=np.stack([r['y_pred'] for r in rows]); s=np.stack([r['sigma'] for r in rows])
    yo=np.stack([r['y_ord'] for r in rows]); logits=np.stack([r['ord_logits'] for r in rows]); ya=np.array([r['y_aux'] for r in rows]); al=np.array([r['aux_logit'] for r in rows])
    return y,p,s,yo,logits,ya,al

def predictive_metrics(rows,trait_names):
    y,p,s,yo,logits,ya,al=rows_to_arrays(rows); pred_ord=logits.argmax(-1); out=[]
    for k,name in enumerate(trait_names):
        out.append({'trait':name,'MAE':mean_absolute_error(y[:,k],p[:,k]),'RMSE':mean_squared_error(y[:,k],p[:,k])**.5,
                    'R2':r2_score(y[:,k],p[:,k]),'Ordinal_Accuracy':accuracy_score(yo[:,k],pred_ord[:,k]),
                    'Ordinal_MacroF1':f1_score(yo[:,k],pred_ord[:,k],average='macro',zero_division=0),
                    'Weighted_Kappa':cohen_kappa_score(yo[:,k],pred_ord[:,k],weights='quadratic')})
    aux_pred=(expit(al)>=.5).astype(int)
    aux={'Precision':precision_score(ya,aux_pred,zero_division=0),'Recall':recall_score(ya,aux_pred,zero_division=0),'F1':f1_score(ya,aux_pred,zero_division=0)}
    return pd.DataFrame(out),aux
