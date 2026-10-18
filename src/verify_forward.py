"""Forward-only scoring. This program never loads target metric columns."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from runtime import ArchitectureModel, RegressionModel

ROOT=Path(__file__).resolve().parents[1]

def main():
    d=pd.read_csv(ROOT/'data/observed_30.csv')
    x=d[['voltage_mv','age_years','bmi']].to_numpy(float)
    z=d[['voltage_mv','age_years','bmi','device_mgdl']].to_numpy(float)
    y=d.reference_mgdl.to_numpy(float)
    records=[];values=[]
    for path in sorted((ROOT/'models').glob('architecture_*.npz')):
        model=ArchitectureModel(path)
        pred=model.predict(x)
        w=model.d['score_weights']
        error=y-pred
        rmse=float(np.sqrt(np.dot(w,error**2)))
        mu=float(w @ y)
        acc=float(100*(1-(w@(error**2))/(w@((y-mu)**2))))
        model_name=path.stem.replace('architecture_','')
        records.append({'family':'architecture','model':model_name,'metric':'rmse','value':rmse})
        records.append({'family':'architecture','model':model_name,'metric':'accuracy_weighted_r2_percent','value':acc})
        for sample,v in zip(d.itertuples(index=False),pred):
            values.append({'family':'architecture','model':model_name,'subject':sample.subject,
                           'phase':sample.phase,'reference_mgdl':sample.reference_mgdl,'prediction_mgdl':v})
    for path in sorted((ROOT/'models').glob('regression_*.joblib')):
        m=RegressionModel(path)
        a=m.d
        pred=m.predict(z)
        residual=y-pred
        mae=float(np.mean(np.abs(residual)))
        mse=float(np.mean(residual**2))
        rmse=float(np.sqrt(a['w_rmse'] @ residual**2))
        w=a['w_r2'];mu=float(w@y)
        r2=float(1-(w@residual**2)/(w@(y-mu)**2))
        if a['effective_df'] is None:
            wm=a['w_modified']; mm=float(wm@y)
            modified=float(1-(wm@residual**2)/(wm@(y-mm)**2))
        else:
            modified=float(1-(1-r2)*29/(29-a['effective_df']))
        for n,v in [('mae',mae),('mse',mse),('rmse',rmse),('r2',r2),('modified_r2',modified)]:
            records.append({'family':'regression','model':a['model_name'],'metric':n,'value':v})
        for sample,v in zip(d.itertuples(index=False),pred):
            values.append({'family':'regression','model':a['model_name'],'subject':sample.subject,
                           'phase':sample.phase,'reference_mgdl':sample.reference_mgdl,'prediction_mgdl':v})
    scores=pd.DataFrame(records)
    scores.to_csv(ROOT/'results/forward_only_scores.csv',index=False)
    pd.DataFrame(values).to_csv(ROOT/'results/forward_only_predictions.csv',index=False)
    with open(ROOT/'results/forward_only_summary.json','w') as f:
        json.dump({'architecture_metrics':int((scores.family=='architecture').sum()),
                   'regression_metrics':int((scores.family=='regression').sum()),
                   'all_metric_entries':len(scores),
                   'prediction_count':len(values),
                   'loads_manuscript_target_metrics':False},f,indent=2)
    print(scores.to_string(index=False))
    return scores

if __name__=='__main__':main()
