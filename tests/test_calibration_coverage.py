import unittest
from pathlib import Path
from analysis.calibration_coverage import audit

ROOT=Path(__file__).resolve().parents[1]
class CalibrationCoverageTests(unittest.TestCase):
    def test_post_training_comparison_is_complete(self):
        a=audit()
        self.assertEqual(len(a),51)
        self.assertEqual(int(a.meets_or_exceeds.sum()),51)
        self.assertEqual(int(a.within_5_percent.sum()),5)
    def test_proxy_accuracy_is_labeled(self):
        a=audit()
        self.assertEqual(int((a.metric=='accuracy_percent').sum()),8)
        self.assertTrue(a[a.metric=='accuracy_percent'].interpretation.str.contains('proxy').all())
    def test_article_scores_are_not_fitting_dependencies(self):
        files=list((ROOT/'src').glob('*.py'))
        text='\n'.join(x.read_text() for x in files)
        self.assertNotIn('published_scores.csv',text)
        self.assertNotIn('paper_algorithm_comparison',text)
    def test_data_has_30_observations_in_10_groups(self):
        import pandas as pd
        x=pd.read_csv(ROOT/'data/observed_30.csv')
        self.assertEqual(len(x),30)
        self.assertEqual(x.subject.nunique(),10)
        self.assertTrue((x.groupby('subject').size()==3).all())
