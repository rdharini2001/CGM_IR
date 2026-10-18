"""Train each fold from measured data and test on entirely unseen participants.

This control reuses the inverse-inferred 3-3-2-1 kernel penalty, but no paper
scores are loaded when fitting the held-out models.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics.pairwise import rbf_kernel
from pso import PSOANNRegressor

ROOT=Path(__file__).resolve().parents[1]

def main():
    d=pd.read_csv(ROOT/'data/observed_30.csv')
    x=d[['voltage_mv','age_years','bmi']].to_numpy(float)
    y=d.reference_mgdl.to_numpy(float)
    groups=d.subject.values
    with np.load(ROOT/'models/architecture_3-3-2-1.npz') as artifact:
        alpha=float(artifact['kernel_alpha'])
        gamma=float(artifact['gamma'])
    rows=[]
    for fold,(tr,te) in enumerate(GroupKFold(n_splits=5).split(x,y,groups)):
        net=PSOANNRegressor(hidden=(3,2),n_particles=25,n_iter=70,seed=2026+fold)
        net.fit(x[tr],y[tr])
        train_base=net.predict(x[tr]); test_base=net.predict(x[te])
        train_scaled=net.x_scaler_.transform(x[tr]);test_scaled=net.x_scaler_.transform(x[te])
        k=rbf_kernel(train_scaled,gamma=gamma)
        dual=np.linalg.solve(k+alpha*np.eye(len(tr)),y[tr]-train_base)
        correction=rbf_kernel(test_scaled,train_scaled,gamma=gamma)@dual
        for i,bias in zip(te,correction):
            rows.append({'fold':fold,'subject':groups[i],'phase':d.phase.iloc[i],
                         'reference_mgdl':y[i],'nn_prediction':test_base[np.where(te==i)[0][0]],
                         'calibration_prediction':test_base[np.where(te==i)[0][0]]+bias,
                         'uses_target_metrics_during_fold_fit':False,
                         'regularization_preselected_from_inverse_full_data':True})
    p=pd.DataFrame(rows).sort_values(['subject','phase']).reset_index(drop=True)
    p.to_csv(ROOT/'results/heldout_control_predictions.csv',index=False)
    scores={
        'n_predictions':len(p),'n_subjects':len(set(groups)),
        'base_rmse_mgdl':float(np.sqrt(np.mean((p.nn_prediction-p.reference_mgdl)**2))),
        'calibrated_rmse_mgdl':float(np.sqrt(np.mean((p.calibration_prediction-p.reference_mgdl)**2))),
        'selection':'5-fold GroupKFold; frozen full-data inverse alpha, unweighted scores',
        'note':'The frozen alpha was originally selected using full-data targets, so this is a sensitivity diagnostic, not fully independent hyperparameter selection.'}
    (ROOT/'results/heldout_control_summary.json').write_text(json.dumps(scores,indent=2))
    print(json.dumps(scores,indent=2))
    return scores
if __name__=='__main__':main()
