"""Predict with the saved PSO–ANN and regression calibration models."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from runtime import ArchitectureModel, RegressionModel


def main():
    parser=argparse.ArgumentParser(description='Forward glucose prediction with saved OPTIC model parameters.')
    parser.add_argument('--list-models',action='store_true',help='List bundled model names and exit')
    parser.add_argument('--model',default='3-3-2-1',help='ANN architecture such as 3-3-2-1, or regression model slug such as pso_ann')
    parser.add_argument('--voltage',type=float)
    parser.add_argument('--age',type=float)
    parser.add_argument('--bmi',type=float)
    parser.add_argument('--device-glucose',type=float)
    args=parser.parse_args()
    available_ann={p.stem.removeprefix('architecture_'):p for p in (ROOT/'models').glob('architecture_*.npz')}
    available_reg={p.stem.removeprefix('regression_'):p for p in (ROOT/'models').glob('regression_*.joblib')}
    if args.list_models:
        print('ANN architectures (voltage, age, BMI):')
        for key in sorted(available_ann):print('  '+key)
        print('Regression calibrators (also require device glucose):')
        for key in sorted(available_reg):print('  '+key)
        return
    if any(getattr(args,x) is None for x in ('voltage','age','bmi')):
        parser.error('--voltage, --age and --bmi are required for prediction')
    values=[args.voltage,args.age,args.bmi]
    if args.model in available_ann:
        estimate=ArchitectureModel(available_ann[args.model]).predict(values)
    else:
        slug=args.model.lower().replace(' ','_').replace('-','_')
        if slug not in available_reg:
            parser.error('Unknown model. Use --list-models to see the available names.')
        if args.device_glucose is None:
            parser.error('Regression model requires --device-glucose')
        estimate=RegressionModel(available_reg[slug]).predict(values+[args.device_glucose])
    print(f'{float(estimate[0]):.3f} mg/dL (research calibration estimate; not clinically validated)')

if __name__=='__main__': main()
