#!/usr/bin/env python3
"""Codex Sol firewall review of Occurrence Theory II (numpy + released data)."""
from pathlib import Path
import numpy as np

TOL=1e-10; FAIL=[]
def cert(name,e,tol=TOL):
    ok=bool(e<tol); print(f"[{'C' if ok else 'FAIL'}] {name}: {e:.3e}")
    if not ok: FAIL.append(name)
def exact(name,g,w):
    ok=g==w; print(f"[{'C' if ok else 'FAIL'}] {name}: {g} (want {w})")
    if not ok: FAIL.append(name)
def comps(adj):
    unseen=set(adj); out=[]
    while unseen:
        q=[unseen.pop()]; p=[]
        while q:
            v=q.pop(); p.append(v)
            for w in adj[v]:
                if w in unseen: unseen.remove(w); q.append(w)
        out.append(p)
    return out
def sector_basis(n,sym):
    out=[]
    for i in range(n):
        if sym:
            a=np.zeros((n,n)); a[i,i]=1; out.append(a)
        for j in range(i+1,n):
            a=np.zeros((n,n)); a[i,j]=1/np.sqrt(2); a[j,i]=(1 if sym else -1)/np.sqrt(2); out.append(a)
    return out
def main():
    d=np.load(Path(__file__).parents[1]/'data/kraus84.npz'); K=d['K']; mu=d['mu']; n=16
    exact('K shape',K.shape,(84,16,16)); exact('mu shape',mu.shape,(84,)); cert('weights sum',abs(mu.sum()-1)); cert('finite data',float(not(np.isfinite(K).all() and np.isfinite(mu).all())),.5)
    cert('antisymmetric Kraus family',max(np.linalg.norm(k+k.T) for k in K))
    cert('CPTP/unital balance',np.linalg.norm(sum(w*k.T@k for w,k in zip(mu,K))-np.eye(n)))
    Z=K[:,:,0]; cert('unit events',max(abs(np.linalg.norm(z)-1) for z in Z)); exact('rank profile',sorted(set(np.linalg.matrix_rank(k,tol=1e-9) for k in K)),[12])
    P=np.diag([0]+[1]*7+[0]+[1]*7); cert('pencil second moment',np.linalg.norm(np.einsum('a,ai,aj->ij',mu,Z,Z)-P/14))
    def channel(x): return sum(w*k.T@x@k for w,k in zip(mu,K))
    spectra={}
    for label,sym in [('symmetric',True),('antisymmetric',False)]:
        B=sector_basis(n,sym); A=np.array([[np.sum(x*channel(y)) for y in B] for x in B]); v=np.linalg.eigvalsh(A); spectra[label]=v
    ts=np.array([-1,-3/7,-2*np.sqrt(3)/7,-1/7,0,1/7,2*np.sqrt(3)/7,3/7,1])
    cert('nine spectral levels',max(min(abs(x-ts)) for x in np.r_[*spectra.values()]))
    def mult(v,t): return int(np.sum(abs(v-t)<1e-8))
    exact('symmetric multiplicities',[mult(spectra['symmetric'],t) for t in [1,3/7,0,-1/7,-3/7]],[1,7,72,42,14])
    exact('antisymmetric multiplicities',[mult(spectra['antisymmetric'],t) for t in [2*np.sqrt(3)/7,3/7,1/7,0,-3/7,-2*np.sqrt(3)/7,-1]],[14,14,42,28,7,14,1])
    B=sector_basis(n,False); A=np.array([[np.sum(x*channel(y)) for y in B] for x in B]); v,q=np.linalg.eigh(A); J=sum(c*b for c,b in zip(q[:,np.argmin(abs(v+1))],B)); J*=4/np.linalg.norm(J)
    cert('minus-one eigenmode',np.linalg.norm(channel(J)+J)); cert('complex structure',np.linalg.norm(J@J+np.eye(n)))
    adj={i:[j for j in range(84) if i!=j and np.linalg.norm(K[i]@Z[j])<TOL] for i in range(84)}; exact('lattice degree',sorted(set(map(len,adj.values()))),[4]); exact('lattice cells',sorted(map(len,comps(adj))),[12]*7)
    rng=np.random.default_rng(20260711); worst_mean=worst_born=0.
    e0=np.eye(n)[0]
    for _ in range(300):
        x=rng.normal(size=n); x/=np.linalg.norm(x); costs=np.array([np.linalg.norm(k@x)**2 for k in K]); worst_mean=max(worst_mean,abs(mu@(costs-1)))
        a=rng.integers(84); y=K[a]@x; cost=np.dot(y,y)
        if cost>1e-12:
            sp=(np.dot(y,e0)**2+np.dot(y,J@e0)**2)/cost; overlap=np.dot(Z[a],x)**2+np.dot(J@Z[a],x)**2
            worst_born=max(worst_born,abs(sp*cost-overlap))
    cert('pointwise mean strain',worst_mean); cert('Born transport identity',worst_born)
    if FAIL: print('FAILED: '+', '.join(FAIL)); return 1
    print('PASSED: independent Paper II firewall ledger'); return 0
if __name__=='__main__': raise SystemExit(main())
