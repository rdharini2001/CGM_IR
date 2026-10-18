"""Report match status only after forward-only prediction/scoring has finished."""
from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
    f=pd.read_csv(ROOT/'results/forward_only_scores.csv')
    a=pd.read_csv(ROOT/'data/paper_architectures.csv')
    b=pd.read_csv(ROOT/'data/paper_algorithm_comparison.csv')
    targets=[]
    for t in a.itertuples(index=False):
        targets.extend([('architecture',t.architecture,'rmse',float(t.validation_rmse)),
                        ('architecture',t.architecture,'accuracy_weighted_r2_percent',float(t.accuracy_percent))])
    for t in b.itertuples(index=False):
        targets.extend([('regression',t.model,n,float(getattr(t,n))) for n in ['mae','mse','rmse','r2','modified_r2']])
    t=pd.DataFrame(targets,columns=['family','model','metric','target'])
    tab=t.merge(f,on=['family','model','metric'],validate='one_to_one')
    tab['absolute_difference']=abs(tab.target-tab.value)
    tab['matched_at_published_precision']=np.where(tab.metric=='accuracy_weighted_r2_percent',
        np.round(tab.value,1)==np.round(tab.target,1),
        np.round(tab.value,3)==np.round(tab.target,3))
    tab.to_csv(ROOT/'results/forward_target_audit.csv',index=False)
    print(f'Matched {tab.matched_at_published_precision.sum()}/{len(tab)} endpoints at printed precision')
    assert bool(tab.matched_at_published_precision.all())

if __name__=='__main__':main()
