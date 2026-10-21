"""Participant-exclusive nested calibration; no published summary statistics are loaded."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, HuberRegressor, BayesianRidge
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

ROOT=Path(__file__).resolve().parents[1] / 'results'
D=pd.read_csv(Path(__file__).resolve().parents[1] / 'data' / 'observed_30.csv')
y=D.reference_mgdl.to_numpy(float);base=D.device_mgdl.to_numpy(float)
group=D.subject.to_numpy()
features = {
 'instrument': ['device_mgdl'],
 'instrument+voltage': ['device_mgdl','voltage_mv'],
 'instrument+physiology': ['device_mgdl','age_years','bmi'],
 'instrument+optical': ['device_mgdl','voltage_mv','age_years','bmi'],
 'instrument+phase':['device_mgdl','voltage_mv','age_years','bmi','phase'],
}
D['phase']=pd.Categorical(D.phase,categories=['before','after','random']).codes.astype(float)

def predict(tr,te,feature,method):
 x=D[features[feature]].to_numpy(float)
 t=y[tr]-base[tr]
 if method=='none':return base[te]
 if method=='mean':return base[te]+t.mean()
 if method=='median':return base[te]+np.median(t)
 xs=x[tr]; xt=x[te]
 opts={
 'ridge2':make_pipeline(StandardScaler(),Ridge(alpha=2.)),
 'ridge10':make_pipeline(StandardScaler(),Ridge(alpha=10.)),
 'ridge50':make_pipeline(StandardScaler(),Ridge(alpha=50.)),
 'ridge200':make_pipeline(StandardScaler(),Ridge(alpha=200.)),
 'huber':make_pipeline(StandardScaler(),HuberRegressor(epsilon=1.6,alpha=3.,max_iter=500)),
 'bayes':make_pipeline(StandardScaler(),BayesianRidge()),
 'poly20':make_pipeline(StandardScaler(),PolynomialFeatures(degree=2,include_bias=False),Ridge(alpha=20.)),
 'poly100':make_pipeline(StandardScaler(),PolynomialFeatures(degree=2,include_bias=False),Ridge(alpha=100.)),
 'svr2':make_pipeline(StandardScaler(),SVR(C=2.,epsilon=.5,gamma=.05)),
 'svr10':make_pipeline(StandardScaler(),SVR(C=10.,epsilon=.5,gamma=.05)),
 'krr2':make_pipeline(StandardScaler(),KernelRidge(kernel='rbf',gamma=.1,alpha=2.)),
 'krr10':make_pipeline(StandardScaler(),KernelRidge(kernel='rbf',gamma=.1,alpha=10.)),
 'knn4':make_pipeline(StandardScaler(),KNeighborsRegressor(n_neighbors=4,weights='distance')),
 'extra':ExtraTreesRegressor(n_estimators=50,min_samples_leaf=3,random_state=19),
 'rf':RandomForestRegressor(n_estimators=50,min_samples_leaf=3,random_state=19),
 }
 model=opts[method];model.fit(xs,t)
 return base[te]+model.predict(xt)

candidates=[(f,m) for f in features for m in ['none','mean','median','ridge2','ridge10','ridge50','ridge200','huber','bayes','poly20','poly100','svr2','svr10','krr2','krr10','knn4','extra','rf']]
outer=GroupKFold(n_splits=10)
record=[];out=np.empty(len(y))
for fold,(tr,te) in enumerate(outer.split(y,groups=group)):
 inner=GroupKFold(n_splits=3)
 errors=np.zeros((len(candidates),len(tr)))
 for ins,iv in inner.split(tr,groups=group[tr]):
  a=tr[ins];b=tr[iv]
  for j,(feature,method) in enumerate(candidates):
   try:errors[j,iv]=predict(a,b,feature,method)-y[b]
   except Exception as e:errors[j,iv]=1e6
 ranks=(errors**2).mean(axis=1)
 best=np.argmin(ranks)
 f,m=candidates[best]
 pred=predict(tr,te,f,m);out[te]=pred
 record.append({'fold':fold,'train_subjects':len(np.unique(group[tr])),'test_subjects':len(np.unique(group[te])),'features':f,'model':m,'inner_rmse':np.sqrt(ranks[best]),'holdout_rmse':np.sqrt(np.mean((pred-y[te])**2))})
 print('fold',fold,'best',f,m,'inner',round(np.sqrt(ranks[best]),3),'holdout',round(np.sqrt(np.mean((pred-y[te])**2)),3),flush=True)

pd.DataFrame(record).to_csv(ROOT/'nested_search_lopo.csv',index=False)
pd.DataFrame({'subject':group,'phase':pd.read_csv(Path(__file__).resolve().parents[1] / 'data' / 'observed_30.csv').phase,'reference_mgdl':y,'device_mgdl':base,'nested_estimate_mgdl':out}).to_csv(ROOT/'nested_predictions_lopo.csv',index=False)
for name,p in [('recorded',base),('nested',out)]:
 err=p-y
 print(name, json.dumps({'mae':round(np.abs(err).mean(),5),'mse':round(np.mean(err**2),5),'rmse':round(np.sqrt(np.mean(err**2)),5),'r2':round(1-sum(err**2)/sum((y-y.mean())**2),5)}),flush=True)
