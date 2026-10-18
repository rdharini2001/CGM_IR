"""Equation-10 inverse reconstruction of latent mean glucose (Table 4)."""
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def run():
    T=pd.read_csv(ROOT/'data/hba1c.csv')
    # The reference source gives estimated HbA1c but not prior 3-month mean G.
    # Invert the published equation to obtain the implied (NOT measured) G.
    T['implied_three_month_mean_glucose_mgdl']=28.7*T.reported_estimated_hba1c_percent-46.7
    T['computed_hba1c_from_implied_mean']=(T.implied_three_month_mean_glucose_mgdl+46.7)/28.7
    T['absolute_hba1c_difference']=abs(T.reported_estimated_hba1c_percent-T.actual_hba1c_percent)
    T['inference_type']='algebraic_inverse_of_published_HbA1c_not_measured_three_month_glucose'
    assert np.allclose(T.computed_hba1c_from_implied_mean,T.reported_estimated_hba1c_percent)
    T.to_csv(ROOT/'results/hba1c_implied_mean.csv',index=False)
    print(f"Table 4: {len(T)} exact equation outputs, mean absolute HbA1c difference {T.absolute_hba1c_difference.mean():.3f} p.p.")
if __name__=='__main__':run()
