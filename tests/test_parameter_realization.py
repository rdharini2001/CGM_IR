from pathlib import Path
import json
import subprocess
import sys
import numpy as np
import pandas as pd
import joblib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from runtime import ArchitectureModel, RegressionModel


def test_forward_exact_printed_endpoints():
    t=pd.read_csv(ROOT/'results/forward_target_audit.csv')
    assert len(t)==51
    assert t.matched_at_published_precision.all()
    assert t.absolute_difference.max()<1e-6


def test_forward_recomputed_from_artifacts():
    obs=pd.read_csv(ROOT/'data/observed_30.csv')
    x=obs[['voltage_mv','age_years','bmi']].values
    z=obs[['voltage_mv','age_years','bmi','device_mgdl']].values
    p=pd.read_csv(ROOT/'results/forward_only_predictions.csv')
    for a in sorted((ROOT/'models').glob('architecture_*.npz')):
        model=a.stem.replace('architecture_','')
        s=p[(p.family=='architecture')&(p.model==model)]
        assert len(s)==30
        assert np.allclose(ArchitectureModel(a).predict(x),s.prediction_mgdl)
    for a in sorted((ROOT/'models').glob('regression_*.joblib')):
        model=RegressionModel(a)
        s=p[(p.family=='regression')&(p.model==model.d['model_name'])]
        assert len(s)==30
        assert np.allclose(model.predict(z),s.prediction_mgdl)


def test_network_parameter_structure():
    arch=pd.read_csv(ROOT/'results/architecture_parameter_realization.csv')
    assert len(arch)==8
    for r in arch.itertuples(index=False):
        m=ArchitectureModel(ROOT/r.model_path)
        arr=m.d
        expected=3*int(arr['h1'])+int(arr['h1'])*int(arr['h2'])+int(arr['h2'])
        assert len(arr['nn_weights'])==expected==r.n_neural_weights
        assert len(arr['kernel_dual'])==30
        assert arr['kernel_alpha']>0
        assert abs(float(arr['score_weights'].sum())-1)<1e-10
        assert (arr['score_weights']>0).all()
    assert arch.loc[arch.architecture=='3-3-2-1','n_neural_weights'].item()==17


def test_regression_populations_are_valid():
    d=pd.read_csv(ROOT/'results/regression_parameter_realization.csv')
    assert len(d)==7
    for r in d.itertuples(index=False):
        a=joblib.load(ROOT/r.model_path)
        assert np.isclose(a['w_rmse'].sum(),1)
        assert np.isclose(a['w_r2'].sum(),1)
        assert a['rmse_high_mass']>=0
        assert (a['w_rmse']>=0).all()


def test_forward_predictions_when_targets_temporarily_unavailable(tmp_path):
    """Forward predictions work in an isolated directory with NO target files."""
    import shutil
    (tmp_path/'src').mkdir()
    (tmp_path/'models').mkdir()
    shutil.copy(ROOT/'src/runtime.py',tmp_path/'src/runtime.py')
    shutil.copy(ROOT/'models/architecture_3-3-2-1.npz',tmp_path/'models/architecture_3-3-2-1.npz')
    program="""import sys; sys.path.insert(0,'src'); from runtime import ArchitectureModel
print(ArchitectureModel('models/architecture_3-3-2-1.npz').predict([[380,51,22.2]])[0])"""
    p=subprocess.run([sys.executable,'-c',program],cwd=tmp_path,capture_output=True,text=True,check=True)
    assert np.isfinite(float(p.stdout.strip()))
    assert not (tmp_path/'data').exists()


def test_participant_disjoint_heldout():
    p=pd.read_csv(ROOT/'results/heldout_control_predictions.csv')
    assert len(p)==30
    assert p.subject.nunique()==10
    assert not p.duplicated(['subject','phase']).any()
    assert p.groupby('subject').fold.nunique().max()==1
    a=json.loads((ROOT/'results/heldout_control_summary.json').read_text())
    assert a['calibrated_rmse_mgdl']>a['base_rmse_mgdl']


def test_model_output_on_new_input_is_finite():
    a=ArchitectureModel(ROOT/'models/architecture_3-3-2-1.npz')
    p=a.predict([[425,42,24]])
    assert p.shape==(1,)
    assert np.isfinite(p[0])
    b=RegressionModel(ROOT/'models/regression_pso_ann.joblib')
    q=b.predict([[425,42,24,110]])
    assert q.shape==(1,)
    assert np.isfinite(q[0])


def test_supplementary_scenarios_provenance():
    p=json.loads((ROOT/'results/supplementary_check.json').read_text())
    assert p['clarke_observed_rows']==40
    assert p['clarke_simulated_rows']==9
    assert p['clarke_synthetic_rows_target_conditioned'] is True
    assert p['hba1c_longitudinal_average_glucose_inferred_not_observed'] is True
    assert abs(p['optimizer_slope_percent']-28.7)<1e-6
