"""Compute observed 70-step optimization trajectories for a matched 17-weight ANN.

Adam learning rates are a fixed diagnostic sweep; differences in number of
objective function evaluations per step mean the curves are NOT normalized for
optimizer computational cost.
"""
import sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pso import PSOANNRegressor

def main():
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    d=pd.read_csv(ROOT/'data/observed_30.csv')
    x=StandardScaler().fit_transform(d[['voltage_mv','age_years','bmi']].values)
    y=StandardScaler().fit_transform(d[['reference_mgdl']].values).ravel()
    xx=torch.tensor(x,dtype=torch.float64)
    yy=torch.tensor(y,dtype=torch.float64)
    net=PSOANNRegressor(hidden=(3,2),n_particles=25,n_iter=70,seed=2026)
    net.fit(d[['voltage_mv','age_years','bmi']].values,d.reference_mgdl.values)
    baseline=np.array(net.history_[1:71],float)
    rows=[];slopes=[]
    def curve(optimizer,learning_rate):
        torch.manual_seed(2026)
        w1=torch.nn.Parameter(torch.empty((3,3),dtype=torch.float64).uniform_(-.5,.5))
        w2=torch.nn.Parameter(torch.empty((3,2),dtype=torch.float64).uniform_(-.5,.5))
        w3=torch.nn.Parameter(torch.empty((2,),dtype=torch.float64).uniform_(-.5,.5))
        args=[w1,w2,w3]
        opt=(torch.optim.Adam(args,lr=learning_rate) if optimizer=='Adam'
             else torch.optim.SGD(args,lr=learning_rate))
        history=[]
        for _ in range(70):
            opt.zero_grad()
            a=torch.sigmoid(xx@w1)
            b=torch.sigmoid(a@w2)
            pred=b@w3
            loss=torch.mean((pred-yy)**2)
            loss.backward()
            opt.step()
            with torch.no_grad():
                p=torch.sigmoid(torch.sigmoid(xx@w1)@w2)@w3
                history.append(float(torch.mean((p-yy)**2)))
        return np.array(history)
    all_curves=[('PSO',None,baseline)]
    for opt in ('Adam','SGD'):
        for lr in ((.001,.003,.01,.03,.1) if opt=='Adam' else (.01,.03,.1,.3,1.)):
            all_curves.append((opt,lr,curve(opt,lr)))
    # Inverse-design a gradient optimizer learning rate against the manuscript's
    # 28.7% training-slope endpoint. This is not a target-blind comparison.
    pso_slope=float(np.polyfit(np.arange(70),baseline,1)[0])
    target_bp_magnitude=abs(pso_slope)/(1-0.287)
    def bp_slope(lr):
        return float(np.polyfit(np.arange(70),curve('Adam',lr),1)[0])
    tuned_lr=brentq(lambda lr:abs(bp_slope(lr))-target_bp_magnitude,.01,.03,xtol=1e-11)
    tuned_curve=curve('Adam',tuned_lr)
    all_curves.append(('Adam_target_inverse',tuned_lr,tuned_curve))
    for opt,lr,vals in all_curves:
        slope=float(np.polyfit(np.arange(70),vals,1)[0])
        slopes.append({'optimizer':opt,'learning_rate':lr,'training_loss_slope_per_iteration':slope,
                       'starting_normalized_mse':vals[0],'ending_normalized_mse':vals[-1]})
        rows.extend({'optimizer':opt,'learning_rate':lr,'iteration':i+1,'normalized_mse':v}
                    for i,v in enumerate(vals))
    p=pd.DataFrame(slopes)
    pso_slope=p[p.optimizer=='PSO'].iloc[0].training_loss_slope_per_iteration
    p['slope_magnitude_change_vs_PSO_pct']=(1-abs(p.training_loss_slope_per_iteration)/abs(pso_slope))*100
    p['pso_slope_smaller_than_optimizer_pct']=(1-abs(pso_slope)/abs(p.training_loss_slope_per_iteration))*100
    p.to_csv(ROOT/'results/optimizer_slope_sweep.csv',index=False)
    pd.DataFrame(rows).to_csv(ROOT/'results/optimizer_learning_histories.csv',index=False)
    print(p.to_string(index=False))
if __name__=='__main__':main()
