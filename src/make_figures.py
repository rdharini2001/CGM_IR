from __future__ import annotations

def main():
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    R=Path(__file__).resolve().parents[1]
    F=R/'figures';F.mkdir(exist_ok=True)
    a=pd.read_csv(R/'results/architecture_parameter_realization.csv')
    b=pd.read_csv(R/'results/regression_parameter_realization.csv')
    x=np.arange(len(a))
    fig,ax=plt.subplots(figsize=(11,5.6))
    ax.plot(x,a.target_rmse,'o-',label='Manuscript endpoint')
    ax.plot(x,a.forward_rmse,'x--',label='Forward fitted model')
    ax.set_xticks(x,a.architecture,rotation=35,ha='right')
    ax.set_ylabel('Weighted calibration-fit RMSE (mg/dL)')
    ax.set_title('Eight PSO–ANN architectures · forward parameter realization')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(F/'architecture_forward_matches.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5.6))
    ax.scatter(a.kernel_alpha,a.weighted_effective_n,s=60)
    for r in a.itertuples():
        ax.annotate(r.architecture,(r.kernel_alpha,r.weighted_effective_n),fontsize=8,xytext=(4,4),textcoords='offset points')
    ax.set_xscale('log');ax.set_xlabel('Inferred kernel regularization α (log scale)');ax.set_ylabel('Effective weighted calibration population size')
    ax.set_title('Inverse-inferred parameters for the forward predictors');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(F/'inferred_parameter_landscape.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,6))
    y=np.arange(len(b))
    ax.barh(y-.2,b.target_rmse,height=.4,label='Manuscript RMSE')
    ax.barh(y+.2,b.forward_rmse,height=.4,label='Forward metric')
    ax.set_yticks(y,b.model);ax.invert_yaxis();ax.set_xlabel('Target-conditioned, reweighted RMSE')
    ax.set_title('Seven fitted regression calibrators · separate scoring distributions')
    ax.legend();fig.tight_layout();fig.savefig(F/'regression_forward_matches.png',dpi=170);plt.close(fig)
    p=pd.read_csv(R/'results/heldout_control_predictions.csv')
    fig,ax=plt.subplots(figsize=(7,6))
    ax.scatter(p.reference_mgdl,p.nn_prediction,label='PSO–ANN base',alpha=.85)
    ax.scatter(p.reference_mgdl,p.calibration_prediction,label='Inverse-calibrated kernel',alpha=.7)
    lo=min(p.reference_mgdl.min(),p.nn_prediction.min(),p.calibration_prediction.min());hi=max(p.reference_mgdl.max(),p.nn_prediction.max(),p.calibration_prediction.max())
    ax.plot([lo,hi],[lo,hi],color='black',linewidth=1,linestyle='--');ax.set_xlabel('Measured reference (mg/dL)');ax.set_ylabel('Prediction (mg/dL)')
    ax.set_title('Patient-held-out control: calibration penalty instability');ax.legend();fig.tight_layout();fig.savefig(F/'heldout_parameter_transfer.png',dpi=170);plt.close(fig)
    with np.load(R/'models/architecture_3-3-2-1.npz') as d:
        w=d['nn_weights']
    fig,ax=plt.subplots(figsize=(8,2.7))
    ax.bar(np.arange(1,18),w)
    ax.set_xticks(np.arange(1,18));ax.set_xlabel('PSO–ANN weight index');ax.set_ylabel('Weight value')
    ax.set_title('17 trained connection weights of the 3–3–2–1 network')
    fig.tight_layout();fig.savefig(F/'pso_neural_weights.png',dpi=170);plt.close(fig)
    print('Generated five figures.')

if __name__ == "__main__":
    main()
