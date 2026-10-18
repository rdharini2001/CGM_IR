"""Provenance-separated PSO recipe and end-to-end saved forward evaluation."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trace_forward import trace
from runtime import ArchitectureModel
from weight_recipe import train_network


def test_recipe_replays_all_eight_saved_neural_arrays():
    df=pd.read_csv(ROOT/'data/observed_30.csv')
    x=df[['voltage_mv','age_years','bmi']].to_numpy(float)
    y=df.reference_mgdl.to_numpy(float)
    names=pd.read_csv(ROOT/'data/paper_architectures.csv').architecture.to_list()
    assert len(names)==8
    for i,n in enumerate(names):
        net=train_network(n,2026+i,x,y)
        with np.load(ROOT/'models'/f'architecture_{n}.npz') as expected:
            assert np.allclose(net.weights_,expected['nn_weights'],atol=1e-9,rtol=0)
            assert np.allclose(net.x_scaler_.mean_,expected['x_mean'],atol=1e-9,rtol=0)
            assert np.allclose(net.x_scaler_.scale_,expected['x_scale'],atol=1e-9,rtol=0)
        assert len(net.history_)==71


def test_seed_sweep_documents_initialization_nonuniqueness():
    data=pd.read_csv(ROOT/'results/seed_sensitivity.csv')
    assert len(data)==40
    assert set(data.seed_offset)=={0,1,7,31,113}
    assert int(data[data.seed_offset==0].exact_weight_replay.sum())==8
    assert int(data[data.seed_offset!=0].exact_weight_replay.sum())==0
    assert np.isfinite(data.neural_fitted_rmse_mgdl).all()
    assert data.weight_distance_l2.max()>0
    summary=json.loads((ROOT/'results/weight_recipe_summary.json').read_text())
    assert not summary['training_reads_published_metric_targets']
    assert summary['downstream_kernel_and_scorer_fit_use_manuscript_targets']


def test_forward_trace_separates_neural_and_residual_parts():
    inputs=(380.,51.,22.2)
    obj=trace('3-3-2-1',*inputs)
    assert len(obj['first_hidden_activations'])==3
    assert len(obj['second_hidden_activations'])==2
    assert len(obj['kernel_residual_contributions_mgdl'])==30
    assert obj['final_prediction_mgdl']==pytest_approx(
        obj['neural_prediction_mgdl']+obj['calibration_correction_mgdl'])
    assert obj['final_prediction_mgdl']==pytest_approx(ArchitectureModel(ROOT/'models/architecture_3-3-2-1.npz').predict(inputs)[0])
    assert obj['neural_prediction_mgdl']==pytest_approx(96.84404714523852)
    assert obj['final_prediction_mgdl']==pytest_approx(86.99977273346241)


def pytest_approx(x):
    import pytest
    return pytest.approx(x,abs=1e-8)


def test_site_embeds_full_forward_coefficients_and_recipe():
    from test_site import get_data
    d=get_data()
    assert d['training_recipe']['exact_weight_replays']==8
    assert len(d['training_seed_sensitivity'])==40
    assert len(d['networks']['3-3-2-1']['anchors_scaled'])==30
    assert len(d['networks']['3-3-2-1']['kernel_dual'])==30
    t=(ROOT/'docs/index.html').read_text()
    assert 'function forwardInspector()' in t
    assert 'function seedLaboratory()' in t
    assert 'WEIGHT_RECIPE.md' in t


def test_javascript_forward_matches_python_from_saved_artifact():
    """Evaluate the exact website forward function in Node when installed."""
    import shutil, subprocess, re
    import pytest
    node=shutil.which('node')
    if not node:
        pytest.skip('Node unavailable; Python forward checks still run')
    html=(ROOT/'docs/index.html').read_text()
    data=json.loads(re.search(r'<script id="study-data" type="application/json">(.*?)</script>',html,re.S).group(1))
    js=html[html.index('function forwardInspector(){'):html.index('architecture();signal();patientList();')]
    model=json.dumps(data['networks']['3-3-2-1'])
    driver=f"""const DATA={{networks:{{'3-3-2-1':{model}}}}};
const fmt=(v,n=2)=>Number(v).toFixed(n);
const E={{}};
const qs=s=>E[s]||(E[s]={{value:({{'#trace-voltage':'380','#trace-age':'51','#trace-bmi':'22.2'}})[s],textContent:'',addEventListener:()=>{{}}}});
{js}
forwardInspector();
const one=+qs('#trace-final').textContent;
qs('#trace-voltage').value='550';forwardInspector();
const two=+qs('#trace-final').textContent;
console.log(JSON.stringify([one,two]));"""
    process=subprocess.run([node,'-'],input=driver,text=True,capture_output=True,check=True,timeout=15)
    got=json.loads(process.stdout.strip())
    for v,n in zip((380.,550.),got):
        expected=trace('3-3-2-1',v,51.,22.2)['final_prediction_mgdl']
        assert n==pytest.approx(expected,abs=0.006)  # webpage displays two decimals
