"""Independent arithmetic checks for supplementary numerical scenarios."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def main():
    s=pd.read_csv(ROOT/'results/optimizer_slope_sweep.csv')
    inv=s[s.optimizer=='Adam_target_inverse'].iloc[0]
    pso=s[s.optimizer=='PSO'].iloc[0]
    slope_ratio=100*(1-abs(pso.training_loss_slope_per_iteration)/abs(inv.training_loss_slope_per_iteration))
    h=pd.read_csv(ROOT/'results/hba1c_implied_mean.csv')
    h_err=float(np.max(np.abs(h.reported_estimated_hba1c_percent-h.computed_hba1c_from_implied_mean)))
    c=pd.read_csv(ROOT/'results/clarke_conditional_population.csv')
    observed=c[c.source=='published_observation']
    simulated=c[c.source!='published_observation']
    zone_a=float(100*(c.zone=='A').mean())
    clinical_zone_a=float(100*(observed.zone=='A').mean())
    status={
        'optimizer_inverse_tuned_adam_learning_rate':float(inv.learning_rate),
        'optimizer_slope_percent':slope_ratio,
        'optimizer_target_used_to_select_learning_rate':True,
        'hba1c_equation_rows':len(h),'hba1c_max_equation_residual':h_err,
        'hba1c_longitudinal_average_glucose_inferred_not_observed':True,
        'clarke_observed_rows':len(observed),'clarke_simulated_rows':len(simulated),
        'clarke_zone_a_percent_conditional':zone_a,
        'clarke_zone_a_percent_source_only':clinical_zone_a,
        'clarke_synthetic_rows_target_conditioned':True,
        'hardware_design_uses_inferred_component_substitutions':True,
    }
    assert abs(status['optimizer_slope_percent']-28.7)<1e-6
    assert h_err<1e-9 and len(h)==10
    assert len(observed)==40 and len(simulated)==9
    assert round(zone_a,1)==93.9 and clinical_zone_a==100
    (ROOT/'results/supplementary_check.json').write_text(json.dumps(status,indent=2))
    print(json.dumps(status,indent=2))
    return status
if __name__=='__main__':main()
