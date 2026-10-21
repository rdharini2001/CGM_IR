from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
result=pd.read_csv(ROOT/'results'/'model_scores.csv')
pred=pd.read_csv(ROOT/'results'/'predictions.csv')
observed=pd.read_csv(ROOT/'data'/'observed_30.csv')
instrument=json.loads((ROOT/'results'/'instrument_summary.json').read_text())
hba1c=pd.read_csv(ROOT/'data'/'hba1c.csv')
coverage=json.loads((ROOT/'results'/'calibration_coverage_summary.json').read_text())
payload={'scores':result.astype(object).where(result.notna(),None).to_dict(orient='records'),
         'predictions':pred.to_dict(orient='records'),
         'observed':observed.to_dict(orient='records'),'hba1c':hba1c.to_dict(orient='records'),'instrument':instrument,'coverage':coverage}
src=(ROOT/'docs'/'template.html').read_text()
assert src.count('/*__OPTIC_DATA__*/')==1
html=src.replace('/*__OPTIC_DATA__*/','window.OPTIC_DATA='+json.dumps(payload,default=lambda x: None if pd.isna(x) else x,allow_nan=False).replace('</','<\\/')+';')
(ROOT/'docs'/'index.html').write_text(html)
(ROOT/'docs'/'site_data.json').write_text(json.dumps(payload,indent=1,allow_nan=False))
print(f'Built static site with {len(result)} model score rows and {len(pred)} predictions')
