"""Visual record of the deterministic weight-replay and seed-sensitivity run."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]


def main():
    df=pd.read_csv(ROOT/'results/seed_sensitivity.csv')
    names=df.architecture.drop_duplicates().to_list()
    fig,ax=plt.subplots(figsize=(12,5.5))
    for i,name in enumerate(names):
        d=df[df.architecture==name]
        alt=d[d.seed_offset.ne(0)].neural_fitted_rmse_mgdl.to_numpy()
        base=float(d[d.seed_offset.eq(0)].neural_fitted_rmse_mgdl.iloc[0])
        ax.plot([i]*len(alt),alt,marker='o',linestyle='none',alpha=.65)
        ax.plot(i,base,marker='D',markersize=9,markeredgecolor='black',markerfacecolor='none')
    ax.set_xticks(range(len(names)),names,rotation=30,ha='right')
    ax.set_ylabel('Neural-only in-sample RMSE (mg/dL)')
    ax.set_xlabel('PSO–ANN architecture')
    ax.set_title('Five deterministic seeds per architecture: diamond = saved recipe seed')
    ax.grid(axis='y',alpha=.2)
    fig.tight_layout()
    (ROOT/'figures').mkdir(exist_ok=True)
    fig.savefig(ROOT/'figures/seed_training_variation.png',dpi=160)
    plt.close(fig)
    print('Wrote figures/seed_training_variation.png')

if __name__=='__main__':main()
