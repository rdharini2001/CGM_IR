"""Research inference from a fitted OPTIC calibration model."""
import argparse
from pathlib import Path
import joblib
import numpy as np
from src.study import ROOT,SHAPES,REGRESSORS

def main():
    p=argparse.ArgumentParser(description='Explore a fitted PSO-ANN or regression calibration function. Not for clinical use.')
    p.add_argument('--model',default='3-3-2-1',choices=SHAPES+REGRESSORS)
    p.add_argument('--voltage-mv',type=float,required=True)
    p.add_argument('--age',type=float,required=True)
    p.add_argument('--bmi',type=float,required=True)
    a=p.parse_args()
    path=ROOT/'models'/(a.model.lower().replace(' ','_').replace('-','_')+'.joblib')
    m=joblib.load(path)
    pred=m.predict(np.asarray([[a.voltage_mv,a.age,a.bmi]],float))[0]
    print(f'{pred:.4f} mg/dL  [research calibration estimate; not for diagnosis or treatment]')
if __name__=='__main__':main()
