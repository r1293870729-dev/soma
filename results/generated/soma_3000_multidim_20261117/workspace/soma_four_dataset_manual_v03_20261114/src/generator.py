"""Compactly supported independent branches with model-free moment compensation."""
from pathlib import Path
import hashlib,json,math
import numpy as np

VERTICES=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],dtype=np.float64)


def mm(a,b):
    # Exact small-feature contraction, avoiding platform BLAS floating-status flags.
    return np.einsum('ij,jk->ik',a,b,optimize=False)


def renderer_product(u,v):
    return np.stack([np.cos(u),np.sin(u),np.cos(v),np.sin(v),np.cos(u+v),np.sin(u+v),np.cos(u-v),np.sin(u-v)],axis=-1)/np.sqrt(2)


def renderer_mixed(u,v):
    return np.stack([np.cos(u),np.sin(u),np.cos(v),np.sin(v),np.cos(2*u+v),np.sin(2*u+v),np.cos(u+2*v),np.sin(u+2*v)],axis=-1)/np.sqrt(2)


def seed_for(*values):
    return int.from_bytes(hashlib.sha256(json.dumps(values,separators=(',',':')).encode()).digest()[:8],'little')%(2**63-1)


def hash_array(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape,str(a.dtype))).encode()+a.tobytes()).hexdigest()


class Generator:
    def __init__(self,family,plan):
        self.family,self.plan=family,plan
        self.render=renderer_product if family=='product_fourier' else renderer_mixed
        tangent_norm=np.sqrt(1.5) if family=='product_fourier' else np.sqrt(3.)
        self.join_width=3*plan['noise_std']/tangent_norm
        embedding_rng=np.random.default_rng(seed_for('embedding',family))
        self.embedding=np.linalg.qr(embedding_rng.normal(size=(64,8)))[0]
        n=plan['moment_fit_grid'];u,v=self.quadrature(n);y=self.render(u,v);rho=self.bump(u,v);h=rho**2
        self.center=np.sum(h[:,None]*y,axis=0)/h.sum()
        centered=y-self.center
        S=np.einsum('i,ij,ik->jk',h,centered,centered)/len(y)
        R=np.einsum('i,ij,ik->jk',h*h,centered,centered)/len(y)
        vals,vecs=np.linalg.eigh(S)
        if vals[0]<=0 or vals[-1]/vals[0]>1e12:raise ValueError('Generator moment system is ill-conditioned')
        self.Q=vecs[:,-3:]
        # Deterministic eigenvector signs, independent of all model results.
        for j in range(3):
            if self.Q[np.abs(self.Q[:,j]).argmax(),j]<0:self.Q[:,j]*=-1
        self.Sinv=np.linalg.inv(S)
        Rq=mm(mm(mm(mm(self.Q.T,self.Sinv),R),self.Sinv),self.Q)
        self.m=1/np.linalg.eigvalsh(Rq)[-1]
        D=2*self.m*np.eye(3)-self.m**2*Rq
        eig,U=np.linalg.eigh(D/h.mean())
        if eig[0]<=0:raise ValueError('Nonpositive branch transfer covariance')
        self.B=mm(U*np.sqrt(eig),U.T)
        self.contraction=self.m*mm(mm(self.Q,self.Q.T),self.Sinv)
        self.metadata={'family':family,'S_condition':float(vals[-1]/vals[0]),'m':float(self.m),
            'center':self.center.tolist(),'Q':self.Q.tolist(),'B':self.B.tolist(),
            'contraction':self.contraction.tolist(),'embedding_sha256':hash_array(self.embedding),
            'quadrature_rows':len(y),'region_probability':.25,'branch_independent_of_angles':True,
            'join_width_radians':float(self.join_width),'join_width_rule':'3 observational noise standard deviations in tangent-angle units'}

    def quadrature(self,n):
        a=(np.arange(n)+.5)/n*np.pi-np.pi/2
        du,dv=np.meshgrid(a,a,indexing='ij')
        return (np.pi+du.ravel())%(2*np.pi),dv.ravel()%(2*np.pi)

    def bump(self,u,v):
        du=(u-np.pi+np.pi)%(2*np.pi)-np.pi
        dv=(v+np.pi)%(2*np.pi)-np.pi
        inside=(np.abs(du)<np.pi/2)&(np.abs(dv)<np.pi/2)
        def taper(distance):
            ratio=np.clip((np.abs(distance)-(np.pi/2-self.join_width))/self.join_width,0,1)
            return np.cos(np.pi/2*ratio)**2
        return np.where(inside,taper(du)*taper(dv),0.)

    def coordinates(self,u,v,g):
        base=self.render(u,v);rho=self.bump(u,v);active=np.asarray(g)>0
        vectors=VERTICES[np.maximum(g-1,0)]
        difference=-(rho*rho)[:,None]*mm(base-self.center,self.contraction.T)+rho[:,None]*mm(mm(vectors,self.B.T),self.Q.T)
        return base+active[:,None]*difference

    def role(self,seed,role,phase,count,branch_only=False):
        rng=np.random.default_rng(seed_for('role',self.family,seed,role,phase))
        if branch_only:
            u=(np.pi+rng.uniform(-np.pi/2,np.pi/2,count))%(2*np.pi)
            v=rng.uniform(-np.pi/2,np.pi/2,count)%(2*np.pi)
        else:
            u=rng.uniform(0,2*np.pi,count)
            v=np.full(count,self.plan['v0']) if phase==0 else rng.uniform(0,2*np.pi,count)
        g=np.zeros(count,dtype=np.int64)
        inside=self.bump(u,v)>0
        if phase==2:
            if branch_only:
                g=np.resize(np.arange(1,5),count);rng.shuffle(g)
            else:g[inside]=rng.integers(1,5,int(inside.sum()))
        coords=self.coordinates(u,v,g)
        observations=(mm(coords,self.embedding.T)+self.plan['noise_std']*rng.normal(size=(count,64))).astype(np.float32)
        labels=np.stack([np.floor(u*8/(2*np.pi)).astype(np.int64),np.floor(v*8/(2*np.pi)).astype(np.int64),g],axis=1)
        ids=[f'{self.family}:{seed}:{role}:{phase}:{i}' for i in range(count)]
        return {'x':observations,'labels':labels,'u':u,'v':v,'g':g,'rho':self.bump(u,v),'ids':ids,
                'observation_sha256':hash_array(observations),'labels_sha256':hash_array(labels)}

    def moment_check(self,n):
        u,v=self.quadrature(n);base=self.render(u,v)
        mu=base.mean(0);C=mm((base-mu).T,base-mu)/len(base)
        means=[];seconds=[]
        for g in range(1,5):
            y=self.coordinates(u,v,np.full(len(u),g,dtype=np.int64))
            means.append(y.mean(0));seconds.append(mm(y.T,y)/len(y))
        split_mean=np.mean(means,axis=0);split_C=np.mean(seconds,axis=0)-np.outer(split_mean,split_mean)
        return {'mean_relative_error':float(np.linalg.norm(split_mean-mu)/np.sqrt(np.trace(C))),
                'covariance_relative_error':float(np.linalg.norm(split_C-C)/np.linalg.norm(C)),
                'region_unsplit_covariance_trace':float(np.trace(C)),
                'region_split_covariance_trace':float(np.trace(split_C)),
                'validation_quadrature_rows':len(u),'all_four_labels_at_each_coordinate':True}
