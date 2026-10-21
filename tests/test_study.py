import unittest
from pathlib import Path
import numpy as np,pandas as pd,joblib
from src.study import ROOT, INPUTS, SHAPES, REGRESSORS, ResidualKernel,GlucoseModel
from src.physics import scores
from src.sensitivity import noise_from_identical_inputs

class TestReproduction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=pd.read_csv(ROOT/'data'/'observed_30.csv')
        cls.eval=pd.read_csv(ROOT/'results'/'model_scores.csv')
    def test_measured_population(self):
        self.assertEqual(len(self.data),30)
        self.assertEqual(self.data.subject.nunique(),10)
        self.assertEqual(self.data[INPUTS].drop_duplicates().shape[0],29)
    def test_model_count(self):
        self.assertEqual(len(self.eval),15*3)
        self.assertEqual(len(list((ROOT/'models').glob('*.joblib'))),15)
    def test_common_metrics(self):
        for _,r in self.eval.iterrows():
            self.assertAlmostEqual(r.rmse**2,r.mse,places=8)
            self.assertGreaterEqual(r.mae,0)
    def test_fold_exclusivity(self):
        p=pd.read_csv(ROOT/'results'/'predictions.csv')
        self.assertEqual(len(p),15*3*30)
        for (_,name),z in p[p.protocol.eq('participant_5fold')].groupby(['model','subject']):
            self.assertEqual(z.fold.nunique(),1)
    def test_forward_replay(self):
        x=self.data[INPUTS].to_numpy(float)
        rec=pd.read_csv(ROOT/'results'/'predictions.csv')
        for name in SHAPES+REGRESSORS:
            m=joblib.load(ROOT/'models'/(name.lower().replace(' ','_').replace('-','_')+'.joblib'))
            saved=rec[(rec.model==name)&rec.protocol.eq('discovery_fit')].predicted_mgdl.to_numpy()
            np.testing.assert_allclose(m.predict(x),saved,atol=1e-8)
    def test_noise_pair(self):
        r=noise_from_identical_inputs(self.data)
        self.assertEqual(r['duplicate_groups'],1)
        self.assertAlmostEqual(r['within_group_mle_variance'],1.0)
        self.assertAlmostEqual(r['within_group_unbiased_variance'],2.0)
    def test_scoring_identity(self):
        r=scores(np.array([100.,101.]),np.array([101.,102.]))
        self.assertAlmostEqual(r['mse'],r['rmse']**2)
        self.assertEqual(r['mae'],1)
    def test_endpoint_isolation(self):
        source=(ROOT/'src'/'study.py').read_text()+(ROOT/'src'/'sensitivity.py').read_text()
        self.assertNotIn('published_architectures.csv',source)
        self.assertNotIn('published_algorithms.csv',source)

if __name__=='__main__':unittest.main()

class TestWebsite(unittest.TestCase):
    def test_site_and_supplement(self):
        self.assertTrue((ROOT/'docs'/'SUPPLEMENTARY_METHODS.pdf').stat().st_size>20000)
        h=(ROOT/'docs'/'index.html').read_text()
        self.assertIn('Light captured.',h)
        self.assertIn('predictionChart',h)
        self.assertIn('OPTIC_DATA=',h)
        self.assertNotIn('/*__OPTIC_DATA__*/',h)
        import json
        data=json.loads((ROOT/'docs'/'site_data.json').read_text())
        self.assertEqual(len(data['scores']),45)
        self.assertEqual(len(data['predictions']),1350)
