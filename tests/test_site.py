from pathlib import Path
import json,re
import pandas as pd
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]

def get_data():
    text=(ROOT/'docs/index.html').read_text()
    m=re.search(r'<script id="study-data" type="application/json">(.*?)</script>',text,re.S)
    assert m
    return json.loads(m.group(1))

def test_site_counts_and_predictions():
    d=get_data()
    assert len(d['architectures'])==8 and len(d['regressions'])==7
    assert len(d['predictions'])==450 and len(d['observed'])==30 and len(d['hba'])==10
    saved=pd.read_csv(ROOT/'results/forward_only_predictions.csv')
    actual=pd.DataFrame(d['predictions'])
    sort=['family','model','subject','phase']
    saved=saved.sort_values(sort).reset_index(drop=True)
    actual=actual.sort_values(sort).reset_index(drop=True)
    assert saved[sort].equals(actual[sort])
    assert np.allclose(saved.prediction_mgdl,actual.prediction_mgdl,atol=1e-8)

def test_website_scores_and_neural_weights():
    d=get_data();scores=pd.read_csv(ROOT/'results/forward_only_scores.csv')
    for a in d['architectures']:
        n=a['architecture']
        assert scores.query('family=="architecture" and model==@n and metric=="rmse"').value.item()==pytest.approx(a['validation_rmse'])
        with np.load(ROOT/'models'/f'architecture_{n}.npz') as art:
            assert np.allclose(d['networks'][n]['weights'],art['nn_weights'])
    for r in d['regressions']:
        n=r['model']
        for k in ('mae','mse','rmse','r2','modified_r2'):
            assert scores.query('family=="regression" and model==@n and metric==@k').value.item()==pytest.approx(r[k])

def test_website_controls_and_methods_provenance():
    text=(ROOT/'docs/index.html').read_text()
    for k in ('network-svg','patient-list','regression-svg','prediction-svg','hba-svg'):
        assert f'id="{k}"' in text
    assert 'Run a saved model' in text
    assert 'Refit the calibration models' in text and 'bash run_all.sh --train' in text
    assert 'simulated' in text.lower()
    assert 'target_conditioned' in json.dumps(get_data()['provenance'])
