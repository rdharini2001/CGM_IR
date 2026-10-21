"""Post-fitting comparison of discovery calibration to paper's reported performance bounds.

This module is NEVER imported from src/ and NEVER selects hyperparameters, folds,
loss weights, training examples, or model weights. It computes descriptive
comparisons only, after a complete model_scores.csv exists.
"""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT/'analysis/published_scores.csv'
FITTED = ROOT/'results/model_scores.csv'
OUT = ROOT/'results/calibration_coverage.csv'


def audit(scores=None, published=None):
    scores = scores if scores is not None else pd.read_csv(FITTED)
    published = published if published is not None else pd.read_csv(PUBLISHED)
    fitted = scores[scores.protocol.eq('discovery_fit')].set_index('model')
    assert len(fitted)==15 and len(published)==51
    results=[]
    for row in published.itertuples(index=False):
        d=fitted.loc[row.model]
        key={'accuracy_percent':'architecture_accuracy_percent'}.get(row.metric,row.metric)
        obtained=float(d[key]); target=float(row.published_value)
        passed=obtained<=target+1e-10 if row.direction=='min' else obtained>=target-1e-10
        rel=abs(obtained-target)/abs(target) if target else abs(obtained)
        results.append({'section':row.section,'model':row.model,'metric':row.metric,
                        'published_value':target,'calibration_value':obtained,
                        'direction':row.direction,'comparison_basis':row.comparison_basis,
                        'meets_or_exceeds':bool(passed),'within_5_percent':bool(rel<=0.05),
                        'relative_difference':rel,
                        'interpretation':'accuracy proxy (100*R2); not a confirmed identical definition'
                        if row.metric=='accuracy_percent' else 'same named statistic, different evaluation definition may apply'})
    return pd.DataFrame(results)


if __name__=='__main__':
    out=audit()
    OUT.parent.mkdir(exist_ok=True,parents=True)
    out.to_csv(OUT,index=False)
    summary={'total':len(out),'meets_or_exceeds':int(out.meets_or_exceeds.sum()),
             'within_5_percent':int(out.within_5_percent.sum()),
             'directly_comparable_performance_thresholds':int((out.metric!='accuracy_percent').sum()),
             'directly_comparable_passed':int(out[(out.metric!='accuracy_percent')].meets_or_exceeds.sum()),
             'proxy_accuracy_passed':int(out[(out.metric=='accuracy_percent')].meets_or_exceeds.sum()),
             'all_discovery_population':True,
             'accuracy_percent_is_proxy':True,
             'manuscript_scores_used_in_training':False}
    (ROOT/'results/calibration_coverage_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    assert summary['meets_or_exceeds']==51, 'Check the latest training outputs before describing numerical coverage'
