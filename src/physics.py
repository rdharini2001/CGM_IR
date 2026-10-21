"""Optical representation and conventional calibration metrics."""
from dataclasses import dataclass
import numpy as np

@dataclass
class OpticalTransform:
    voltage_reference: float
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x):
        a=np.asarray(x,float);v0=float(a[:,0].max()+1.)
        z=np.column_stack((np.log(v0/(a[:,0]+1.)),a[:,1:]))
        return cls(v0,z.mean(0),np.maximum(z.std(0),1e-8))

    def transform(self,x):
        a=np.atleast_2d(np.asarray(x,float))
        if np.any(a[:,0]<0):raise ValueError('Voltage must be nonnegative')
        z=np.column_stack((np.log(self.voltage_reference/(a[:,0]+1.)),a[:,1:]))
        return (z-self.mean)/self.scale


def scores(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float);e=p-y
    mse=float(np.mean(e**2));var=float(np.sum((y-y.mean())**2))
    return dict(n=len(y),mae=float(np.mean(abs(e))),mse=mse,rmse=float(np.sqrt(mse)),r2=float(1-np.sum(e**2)/var) if var>0 else None)


def clarke_zone_a(r,p):
    r=np.asarray(r,float);p=np.asarray(p,float)
    return np.where(r<70.,p<70.,np.abs(p-r)<=.2*r)


def iso15_fraction(r,p):
    r=np.asarray(r,float);p=np.asarray(p,float)
    return float(np.mean(abs(p-r)<=np.where(r<100.,15.,.15*r)))
