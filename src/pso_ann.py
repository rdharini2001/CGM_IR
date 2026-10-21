"""Optical PSO-ANN: training data select neural weights and smoothness."""
import numpy as np
from .physics import OpticalTransform

def sigmoid(x):return 1./(1.+np.exp(-np.clip(x,-40.,40.)))

class OpticalPSO:
    def __init__(self,hidden=(3,2),particles=25,iterations=70,seed=2023,penalty=.001,jitter_weight=.01,
                 inertia=1.25,cognitive=2.76,social=2.34):
        self.hidden=tuple(hidden);self.particles=particles;self.iterations=iterations
        self.seed=seed;self.penalty=penalty;self.jitter_weight=jitter_weight
        self.inertia=inertia;self.cognitive=cognitive;self.social=social

    def num_weights(self):
        h,k=self.hidden;return 3*h+h*k+k

    def _run(self,z,weights):
        w=np.atleast_2d(weights);n=len(w);h,k=self.hidden;i=3*h;j=i+h*k
        w1=w[:,:i].reshape(n,3,h);w2=w[:,i:j].reshape(n,h,k);w3=w[:,j:].reshape(n,k)
        a=sigmoid(np.einsum('sf,pfh->psh',z,w1));b=sigmoid(np.einsum('psh,phk->psk',a,w2))
        return np.einsum('psk,pk->ps',b,w3)

    def fit(self,x,y):
        x=np.asarray(x,float);y=np.asarray(y,float)
        self.transform_=OpticalTransform.fit(x)
        self.ymean_=float(y.mean());self.yscale_=max(float(y.std()),1e-8)
        z=self.transform_.transform(x);t=(y-self.ymean_)/self.yscale_
        # Robustness to a ±0.5 mV perturbation in displayed integer-mV values.
        xp=x.copy();xm=x.copy();xp[:,0]+=.5;xm[:,0]-=.5
        zp=self.transform_.transform(xp);zm=self.transform_.transform(xm)
        rng=np.random.default_rng(self.seed);d=self.num_weights()
        pos=rng.uniform(-1,1,(self.particles,d));vel=rng.uniform(-.1,.1,pos.shape)
        def obj(w):
            dat=np.mean((self._run(z,w)-t)**2,axis=1)
            reg=np.mean(w*w,axis=1)
            smooth=np.mean((self._run(zp,w)-self._run(zm,w))**2,axis=1)
            return dat+self.penalty*reg+self.jitter_weight*smooth
        personal=pos.copy();personal_loss=obj(pos);i=int(personal_loss.argmin());best=personal[i].copy();loss=float(personal_loss[i]);self.history_=[loss]
        for _ in range(self.iterations):
            vel=np.clip(self.inertia*vel+self.cognitive*rng.random(pos.shape)*(personal-pos)+self.social*rng.random(pos.shape)*(best-pos),-.75,.75)
            pos=np.clip(pos+vel,-5,5);now=obj(pos)
            better=now<personal_loss;personal[better]=pos[better];personal_loss[better]=now[better]
            i=int(personal_loss.argmin())
            if personal_loss[i]<loss:best=personal[i].copy();loss=float(personal_loss[i])
            self.history_.append(loss)
        self.weights_=best;return self

    def predict(self,x):
        return self.ymean_+self.yscale_*self._run(self.transform_.transform(x),self.weights_)[0]

    def save(self,path):
        np.savez_compressed(path,weights=self.weights_,hidden=self.hidden,voltage_reference=self.transform_.voltage_reference,
            mean=self.transform_.mean,scale=self.transform_.scale,ymean=self.ymean_,yscale=self.yscale_,penalty=self.penalty,jitter_weight=self.jitter_weight)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as f:
            m=cls(hidden=tuple(f['hidden']),penalty=float(f['penalty']),jitter_weight=float(f['jitter_weight']))
            m.weights_=f['weights'];m.transform_=OpticalTransform(float(f['voltage_reference']),f['mean'],f['scale']);m.ymean_=float(f['ymean']);m.yscale_=float(f['yscale']);return m
