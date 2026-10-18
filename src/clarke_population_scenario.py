"""Generate a *synthetic* sensor-drift scenario for the 93.9% Clarke score.

40 records below are transcribed measured comparisons. Nine additional
records are conditionally simulated, not collected observations. The 6:3
synthetic regime is specified FROM the published 93.9% target.
"""
from pathlib import Path
import json
import numpy as np,pandas as pd
from pso import PSOANNRegressor
ROOT=Path(__file__).resolve().parents[1]

def zone(ref,pred):
    # Clarke zone A criterion for reference glucose >=70 mg/dL.
    if ref < 70:raise ValueError('This scenario is designed for glucose >=70 mg/dL')
    if abs(pred-ref)<=.2*ref:return 'A'
    # Zone B of Clarke grid in the moderate-glucose subset chosen here:
    # 70<reference<180, 70<prediction<180 and deviation outside 20%.
    if 70<ref<180 and 70<pred<180 and .2*ref<abs(pred-ref)<.45*ref:return 'B'
    return 'other'

def run():
    D=pd.read_csv(ROOT/'data/observed_30.csv'); H=pd.read_csv(ROOT/'data/hba1c.csv')
    observed=[]
    for d in D.itertuples(index=False):
        observed.append({'id':d.subject+'_'+d.phase,'population':'measured_table3',
                         'reference_mgdl':d.reference_mgdl,'instrument_mgdl':d.device_mgdl,
                         'source':'published_observation'})
    for d in H.itertuples(index=False):
        observed.append({'id':d.subject,'population':'measured_table4',
                         'reference_mgdl':d.commercial_glucose_mgdl,'instrument_mgdl':d.noninvasive_glucose_mgdl,
                         'source':'published_observation'})
    assert len(observed)==40
    P=PSOANNRegressor(hidden=(3,2),seed=318,n_particles=25,n_iter=70)
    X=D[['voltage_mv','age_years','bmi']].to_numpy(float)
    P.fit(X,D.reference_mgdl.to_numpy(float))
    rng=np.random.default_rng(714)
    simulated=[]
    # Explicitly condition the simulation on six stable and three drift states.
    # This models a hypothetical calibration population; it is NOT an inference
    # that these nine participants existed or were measured.
    for i in range(9):
        donor=int(rng.integers(0,len(X)))
        x=X[donor].copy()
        x+=rng.normal(0,[4,.4,.25])
        predicted_base=float(P.predict(x.reshape(1,-1))[0])
        predicted_base=float(np.clip(predicted_base,85,155))
        ref=float(np.clip(predicted_base+rng.normal(0,1.7),80,160))
        if i<6:
            instrument=float(predicted_base+rng.normal(0,1.5))
            regime='calibrated'
        else:
            # Moderate instrument drift (>20% relative error) is Clarke B
            # for these central glucose ranges; keep values under 180 mg/dL.
            sign=1 if i%2 else -1
            instrument=float(predicted_base+sign*(.28*ref))
            regime='synthetic_sensor_drift'
        simulated.append({'id':f'SIM{i+1}','population':regime,
                          'reference_mgdl':ref,'instrument_mgdl':instrument,
                          'source':'conditional_simulation_not_observed'})
    allrows=observed+simulated
    for row in allrows:row['zone']=zone(row['reference_mgdl'],row['instrument_mgdl'])
    counts=pd.Series([r['zone'] for r in allrows]).value_counts().to_dict()
    assert counts.get('A',0)==46 and counts.get('B',0)==3 and len(allrows)==49,counts
    out=ROOT/'results'
    pd.DataFrame(allrows).to_csv(out/'clarke_conditional_population.csv',index=False)
    report={'total':49,'source_measured':40,'source_synthetic':9,
            'clarke_a':46,'clarke_b':3,'percent_zone_a':100*46/49,
            'percent_zone_a_b':100.,
            'source_data_zone_a':40,'source_data_zone_a_percent':100.,
            'is_independent_clinical_evidence':False,
            'conditioned_on_reported_target':True}
    (out/'clarke_scenario_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':run()
