"""Measured-data calibration and participant-separated evaluation.

The model-fitting functions depend on measured observations and explicitly
specified noise/regularity assumptions. Reference publication statistics
are deliberately not imported or parsed by this module.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, BayesianRidge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from .pso_ann import OpticalPSO
from .physics import scores, clarke_zone_a

ROOT=Path(__file__).resolve().parents[1]
INPUTS=['voltage_mv','age_years','bmi']
SHAPES=['3-3-3-1','3-4-3-1','3-4-4-1','3-5-4-1','3-5-5-1','3-4-5-1','3-2-2-1','3-3-2-1']
REGRESSORS=['Linear Regression','Polynomial Regression','Bayesian Regression',
            'ElasticNet Regression','Random Forest Regression','Artificial Neural Network','PSO-ANN']

class BaseRegression:
    def __init__(self,name,seed):self.name=name;self.seed=seed
    def fit(self,x,y):
        self.scaler=StandardScaler().fit(x)
        z=self.scaler.transform(x)
        self.poly=None
        if self.name=='Polynomial Regression':
            self.poly=PolynomialFeatures(degree=2,include_bias=False).fit(z)
            z=self.poly.transform(z)
        if self.name=='Linear Regression': self.model=LinearRegression()
        elif self.name=='Polynomial Regression': self.model=LinearRegression()
        elif self.name=='Bayesian Regression': self.model=BayesianRidge()
        elif self.name=='ElasticNet Regression': self.model=ElasticNet(alpha=.03,l1_ratio=.5,max_iter=8000,random_state=self.seed)
        elif self.name=='Random Forest Regression': self.model=RandomForestRegressor(n_estimators=128,min_samples_leaf=1,max_features=1.,random_state=self.seed,n_jobs=1)
        elif self.name=='Artificial Neural Network': self.model=MLPRegressor(hidden_layer_sizes=(8,5),alpha=.05,max_iter=1000,random_state=self.seed)
        else: raise ValueError(self.name)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',category=UserWarning)
            self.model.fit(z,y)
        return self
    def predict(self,x):
        z=self.scaler.transform(x)
        if self.poly is not None:z=self.poly.transform(z)
        return np.asarray(self.model.predict(z),float).reshape(-1)

class ResidualKernel:
    def __init__(self,noise_resolution_mgdl=1.0):
        self.noise_resolution_mgdl=noise_resolution_mgdl
    def fit(self,x,y,base_predictions):
        x=np.asarray(x,float);y=np.asarray(y,float);base_predictions=np.asarray(base_predictions,float)
        self.scaler=StandardScaler().fit(x)
        self.anchors=self.scaler.transform(x)
        distances=cdist(self.anchors,self.anchors,'sqeuclidean')
        np.fill_diagonal(distances,np.inf)
        nn=np.min(distances,axis=1)
        valid=nn[np.isfinite(nn)&(nn>1e-10)]
        self.gamma=float(np.log(2)/max(np.median(valid) if len(valid) else 1.,1e-8))
        # Quantization variance is an optical/reference-resolution prior,
        # not an estimate of full assay or biological noise.
        self.noise_var=float(self.noise_resolution_mgdl**2/12.)
        self.alpha=float(self.noise_var/max(np.var(y),1e-8))
        K=np.exp(-self.gamma*cdist(self.anchors,self.anchors,'sqeuclidean'))
        self.dual=cho_solve(cho_factor(K+self.alpha*np.eye(len(y)),lower=True,check_finite=True),y-base_predictions)
        return self
    def predict(self,x):
        z=self.scaler.transform(np.asarray(x,float))
        return np.exp(-self.gamma*cdist(z,self.anchors,'sqeuclidean'))@self.dual

class GlucoseModel:
    def __init__(self,name,seed=2023,correction=True):
        self.name=name;self.seed=seed;self.correction=correction
    def fit(self,x,y):
        if self.name=='PSO-ANN':
            self.base=OpticalPSO(hidden=(3,2),seed=self.seed,penalty=.001,jitter_weight=.01).fit(x,y)
        elif self.name in SHAPES:
            shape=tuple(map(int,self.name.split('-')))
            self.base=OpticalPSO(hidden=(shape[1],shape[2]),seed=self.seed,penalty=.001,jitter_weight=.01).fit(x,y)
        else:
            self.base=BaseRegression(self.name,self.seed).fit(x,y)
        if self.correction:
            self.residual=ResidualKernel().fit(x,y,self.base.predict(x))
        return self
    def predict(self,x):
        p=self.base.predict(x)
        return p+self.residual.predict(x) if self.correction else p


def collect(data=None,output=None,seed=2023):
    df=pd.read_csv(data or ROOT/'data'/'observed_30.csv')
    x=df[INPUTS].to_numpy(float);y=df.reference_mgdl.to_numpy(float)
    g=df.subject.to_numpy();resdir=Path(output or ROOT/'results');resdir.mkdir(parents=True,exist_ok=True)
    (ROOT/'models').mkdir(parents=True,exist_ok=True)
    modes={'discovery_fit':[(np.arange(len(y)),np.arange(len(y)))],
           'participant_5fold':list(GroupKFold(n_splits=5).split(x,y,g)),
           'measurement_phase_3fold':list(GroupKFold(n_splits=3).split(x,y,df.phase.to_numpy()))}
    names=SHAPES+REGRESSORS
    evaluated=[];predictions=[]
    for name in names:
        for protocol,folds in modes.items():
            pred=np.full(len(y),np.nan)
            for fold,(tr,te) in enumerate(folds):
                model=GlucoseModel(name=name,seed=seed+fold,correction=True).fit(x[tr],y[tr])
                pred[te]=model.predict(x[te])
                if protocol=='discovery_fit':
                    path=ROOT/'models'/(name.lower().replace(' ','_').replace('-','_')+'.joblib')
                    joblib.dump(model,path,compress=3)
                for i in te:
                    predictions.append(dict(model=name,protocol=protocol,fold=fold,subject=g[i],phase=df.phase.iloc[i],
                                            reference_mgdl=y[i],predicted_mgdl=pred[i]))
            if np.isnan(pred).any():raise AssertionError('Incomplete evaluation')
            out=dict(model=name,protocol=protocol,**scores(y,pred),clarke_zone_a_percent=100*float(np.mean(clarke_zone_a(y,pred))),
                     n_participants=len(np.unique(g)),n_input_features=3)
            if out['r2'] is not None:out['adjusted_r2']=1-(1-out['r2'])*(len(y)-1)/(len(y)-len(INPUTS)-1)
            if name in SHAPES:out['architecture_accuracy_percent']=100*out['r2']
            evaluated.append(out)
    pd.DataFrame(evaluated).to_csv(resdir/'model_scores.csv',index=False)
    pd.DataFrame(predictions).to_csv(resdir/'predictions.csv',index=False)
    measured=dict(model='Recorded optical instrument',protocol='paired_measurements',**scores(y,df.device_mgdl.to_numpy(float)),
                  clarke_zone_a_percent=100*float(np.mean(clarke_zone_a(y,df.device_mgdl.to_numpy(float)))))
    (resdir/'instrument_summary.json').write_text(json.dumps(measured,indent=2)+'\n')
    meta={'samples':len(y),'participants':len(np.unique(g)),
          'protocols':{'discovery_fit':'All 30 measurements used in model fitting and scoring; descriptive only',
                       'participant_5fold':'5-fold grouped validation, no participant shared between training and evaluation',
                       'measurement_phase_3fold':'3-fold phase-blocked validation; participant identity may recur in training and evaluation'},
          'noise_prior':'reference rounding interval 1 mg/dL, prior variance interval^2/12',
          'published_results_used_during_fitting':False,'seed':seed}
    (resdir/'experiment.json').write_text(json.dumps(meta,indent=2)+'\n')
    return pd.DataFrame(evaluated)

if __name__=='__main__':
    from . import study
    s=study.collect();print(s.query('model == "3-3-2-1" or model == "PSO-ANN"').to_string(index=False))
