from pathlib import Path
import json
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from runtime import ArchitectureModel, RegressionModel


def test_architectures_return_saved_predictions():
    data=pd.read_csv(ROOT/'data/observed_30.csv')
    saved=pd.read_csv(ROOT/'results/forward_only_predictions.csv')
    x=data[['voltage_mv','age_years','bmi']].to_numpy(float)
    names=list((ROOT/'models').glob('architecture_*.npz'))
    assert len(names)==8
    for path in names:
        m=ArchitectureModel(path)
        p=m.predict(x)
        s=saved[(saved.family=='architecture')&(saved.model==path.stem.removeprefix('architecture_'))]
        assert len(s)==30
        assert np.allclose(p,s.prediction_mgdl,atol=1e-9)
        assert m.d['score_weights'].sum()==pytest.approx(1)


def test_regressions_return_saved_predictions():
    data=pd.read_csv(ROOT/'data/observed_30.csv')
    x=data[['voltage_mv','age_years','bmi','device_mgdl']].to_numpy(float)
    saved=pd.read_csv(ROOT/'results/forward_only_predictions.csv')
    names=list((ROOT/'models').glob('regression_*.joblib'))
    assert len(names)==7
    for path in names:
        m=RegressionModel(path)
        s=saved[(saved.family=='regression')&(saved.model==m.d['model_name'])]
        assert len(s)==30
        assert np.allclose(m.predict(x),s.prediction_mgdl,atol=1e-9)


def test_51_metrics_recomputed_from_predictions():
    s=pd.read_csv(ROOT/'results/forward_only_scores.csv')
    assert len(s)==51
    assert sum(s.family=='architecture')==16
    assert sum(s.family=='regression')==35
    data=pd.read_csv(ROOT/'data/observed_30.csv')
    y=data.reference_mgdl.to_numpy(float)
    for p in (ROOT/'models').glob('architecture_*.npz'):
        m=ArchitectureModel(p)
        pred=m.predict(data[['voltage_mv','age_years','bmi']].to_numpy(float))
        w=m.d['score_weights'];avg=w@y
        rmse=np.sqrt(w@((y-pred)**2))
        accuracy=100*(1-(w@((y-pred)**2))/(w@((y-avg)**2)))
        n=p.stem.removeprefix('architecture_')
        assert s.query('family=="architecture" and model==@n and metric=="rmse"').value.item()==pytest.approx(rmse,abs=1e-8)
        assert s.query('family=="architecture" and model==@n and metric=="accuracy_weighted_r2_percent"').value.item()==pytest.approx(accuracy,abs=1e-8)


def test_saved_models_do_not_need_training_code(tmp_path):
    import shutil
    (tmp_path/'src').mkdir();(tmp_path/'models').mkdir()
    shutil.copy(ROOT/'src/runtime.py',tmp_path/'src/runtime.py')
    shutil.copy(ROOT/'models/architecture_3-3-2-1.npz',tmp_path/'models/architecture_3-3-2-1.npz')
    command='import sys;sys.path.insert(0,"src");from runtime import ArchitectureModel;print(ArchitectureModel("models/architecture_3-3-2-1.npz").predict([380,51,22.2])[0])'
    r=subprocess.run([sys.executable,'-c',command],cwd=tmp_path,check=True,capture_output=True,text=True)
    assert np.isfinite(float(r.stdout.strip()))
    assert not (tmp_path/'src/train_realization.py').exists()


def test_both_training_and_inference_are_available():
    for relative in ('src/train_realization.py', 'src/pso.py',
                     'src/verify_forward.py', 'src/audit_targets.py',
                     'models/architecture_3-3-2-1.npz', 'run_all.sh'):
        assert (ROOT / relative).exists()
    runner = (ROOT/'run_all.sh').read_text()
    assert '--train' in runner and 'train_realization.py' in runner
    assert 'verify_forward.py' in runner


def test_saved_regression_pso_is_inference_serializable():
    import joblib
    m = joblib.load(ROOT/'models/regression_pso_ann.joblib')
    from runtime import FrozenPSOANN
    assert isinstance(m['model'], FrozenPSOANN)
    assert not hasattr(m['model'], 'fit')


def test_prediction_cli():
    cmd=[sys.executable,'predict.py','--model','3-3-2-1','--voltage','380','--age','51','--bmi','22.2']
    r=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,check=True)
    assert 'mg/dL' in r.stdout
    r=subprocess.run([sys.executable,'predict.py','--list-models'],cwd=ROOT,capture_output=True,text=True,check=True)
    assert '3-3-2-1' in r.stdout and 'pso_ann' in r.stdout
