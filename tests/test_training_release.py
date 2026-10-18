from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from runtime import FrozenPSOANN


def test_training_targets_and_parameter_files():
    a = pd.read_csv(ROOT/'results/architecture_parameter_realization.csv')
    b = pd.read_csv(ROOT/'results/regression_parameter_realization.csv')
    assert len(a)==8 and len(b)==7
    assert np.allclose(a.forward_rmse,a.target_rmse,atol=1e-5)
    assert np.allclose(a.forward_weighted_r2_percent,a.target_accuracy_percent,atol=1e-5)
    for k in ('mae','mse','rmse','r2','modified_r2'):
        assert np.allclose(b['forward_'+k],b['target_'+k],atol=1e-5)


def test_fitted_pso_regression_can_run_without_optimizer():
    d=joblib.load(ROOT/'models/regression_pso_ann.joblib')
    assert isinstance(d['model'],FrozenPSOANN)
    assert not hasattr(d['model'],'fit')


def test_training_inputs_and_scenarios_are_identified():
    script=(ROOT/'src/train_realization.py').read_text()
    assert 'paper_architectures.csv' in script
    assert 'paper_algorithm_comparison.csv' in script
    assert "'target_conditioned':True" in script
    check=json.loads((ROOT/'results/supplementary_check.json').read_text())
    assert check['clarke_simulated_rows']==9
    assert check['hba1c_longitudinal_average_glucose_inferred_not_observed']


def test_methods_and_site_include_training():
    assert (ROOT/'docs/TRAINING.md').exists()
    assert 'run_all.sh --train' in (ROOT/'README.md').read_text()
    assert 'run_all.sh --train' in (ROOT/'docs/index.html').read_text()
