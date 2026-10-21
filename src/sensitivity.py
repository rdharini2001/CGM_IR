"""Measurement-noise sensitivity and GCV, estimated from the observed cohort."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor,cho_solve
from scipy.spatial.distance import cdist
from .study import ROOT, SHAPES, REGRESSORS, INPUTS
from .physics import scores


def noise_from_identical_inputs(df):
    d=[]
    for _,group in df.groupby(INPUTS):
        if len(group)>1:d.append(group.reference_mgdl.to_numpy(float))
    if not d:return None
    # For the single duplicate pair, the conditional-mean residuals are ±1.
    ss=sum(float(np.sum((z-z.mean())**2)) for z in d)
    n=sum(len(z) for z in d)
    dfree=sum(len(z)-1 for z in d)
    return {'duplicate_groups':len(d), 'within_group_mle_variance':ss/n,
            'within_group_unbiased_variance':ss/dfree,
            'based_on_reference_pairs_only':True}


def calculate():
    df=pd.read_csv(ROOT/'data'/'observed_30.csv')
    x=df[INPUTS].to_numpy(float); y=df.reference_mgdl.to_numpy(float)
    duplicate=noise_from_identical_inputs(df)
    specs=[('display_quantization',1./12.), ('repeated_input_mle',duplicate['within_group_mle_variance']),
           ('repeated_input_unbiased',duplicate['within_group_unbiased_variance'])]
    rows=[]
    for name in SHAPES+REGRESSORS:
        model=joblib.load(ROOT/'models'/(name.lower().replace(' ','_').replace('-','_')+'.joblib'))
        baseline=model.base.predict(x)
        kernel=model.residual
        K=np.exp(-kernel.gamma*cdist(kernel.anchors,kernel.anchors,'sqeuclidean'))
        residual=y-baseline
        for tag,noisevar in specs:
            lam=noisevar/max(np.var(y),1e-8)
            fit=baseline+K@cho_solve(cho_factor(K+lam*np.eye(len(y))),residual)
            rows.append(dict(model=name,regularization=tag,noise_variance_mgdl2=noisevar,alpha=lam,**scores(y,fit)))
        # Additional sensitivity estimate: generalized cross-validation does
        # not see any held-out labels or published aggregate model scores.
        gcv=[]
        for lam in np.geomspace(1e-6,1e2,51):
            ck=cho_factor(K+lam*np.eye(len(y)))
            sol=cho_solve(ck,residual)
            hat=K@cho_solve(ck,np.eye(len(y)))
            loss=np.mean((residual-K@sol)**2)/max((1-np.trace(hat)/len(y))**2,1e-12)
            gcv.append((float(loss),float(lam)))
        _,best=min(gcv)
        p=baseline+K@cho_solve(cho_factor(K+best*np.eye(len(y))),residual)
        rows.append(dict(model=name,regularization='gcv',noise_variance_mgdl2=np.nan,alpha=best,**scores(y,p)))
    pd.DataFrame(rows).to_csv(ROOT/'results'/'noise_assumption_sensitivity.csv',index=False)
    (ROOT/'results'/'repeated_input_noise.json').write_text(json.dumps(duplicate,indent=2)+'\n')
    return pd.DataFrame(rows)

if __name__=='__main__':
    o=calculate();print(o[o.model.eq('3-3-2-1')].to_string(index=False))
