#!/usr/bin/env python3
"""Independent Codex Sol verification of Occurrence Theory I.

No project modules are imported.  The sedenion product is reconstructed from
the Cayley--Dickson recursion and every reported certificate can fail.
"""
from collections import deque
import numpy as np

TOL = 1e-10
FAIL = []

def cert(name, error, tol=TOL):
    good = bool(error < tol)
    print(f"[{'C' if good else 'FAIL'}] {name}: {error:.3e} (tol {tol:.0e})")
    if not good: FAIL.append(name)

def exact(name, got, want):
    good = got == want
    print(f"[{'C' if good else 'FAIL'}] {name}: {got} (want {want})")
    if not good: FAIL.append(name)

def conj(x):
    if len(x) == 1: return x.copy()
    h = len(x)//2
    return np.r_[conj(x[:h]), -x[h:]]

def mul(x, y):
    if len(x) == 1: return x*y
    h = len(x)//2; a,b=x[:h],x[h:]; c,d=y[:h],y[h:]
    return np.r_[mul(a,c)-mul(conj(d),b), mul(d,a)+mul(b,conj(c))]

def left(x):
    return np.column_stack([mul(x, np.eye(len(x))[j]) for j in range(len(x))])

def right(x):
    return np.column_stack([mul(np.eye(len(x))[j], x) for j in range(len(x))])

def crack():
    E=np.eye(16); return np.array([(E[i]+s*E[8+j])/np.sqrt(2)
        for i in range(1,8) for j in range(1,8) if i != j for s in (-1,1)])

def components(adj):
    unseen=set(adj); out=[]
    while unseen:
        q=[unseen.pop()]; part=[]
        while q:
            v=q.pop(); part.append(v)
            for w in adj[v]:
                if w in unseen: unseen.remove(w); q.append(w)
        out.append(part)
    return out

def closure_dimension(gens, lie=False, tol=1e-9):
    basis=[]; flat=[]
    def add(a):
        v=a.reshape(-1).copy()
        for q in flat: v -= q*np.dot(q,v)
        n=np.linalg.norm(v)
        if n < tol: return False
        flat.append(v/n); basis.append(a); return True
    for g in gens: add(g)
    frontier=list(basis)
    while frontier:
        a=frontier.pop(0)
        current=list(basis)
        for b in current:
            c=a@b-b@a if lie else a@b
            if add(c): frontier.append(c)
    return len(basis)

def main():
    rng=np.random.default_rng(20260711); E8=np.eye(8); E16=np.eye(16)
    # Mandatory gates, independently in the appropriate dimensions.
    worst=[0.,0.,0.,0.]
    for _ in range(40):
        x=rng.normal(size=8); y=rng.normal(size=8); a=rng.normal(size=8)
        b=rng.normal(size=8); c=rng.normal(size=8)
        worst[0]=max(worst[0],abs(np.linalg.norm(mul(x,y))-np.linalg.norm(x)*np.linalg.norm(y)))
        x[0]=0; worst[1]=max(worst[1],np.linalg.norm(left(x)+left(x).T))
        worst[2]=max(worst[2],np.linalg.norm(mul(x,x)+np.dot(x,x)*E8[0]))
        worst[3]=max(worst[3],np.linalg.norm(mul(mul(a,b),mul(c,a))-mul(mul(a,mul(b,c)),a)))
    for n,e in zip(('G1 composition','G2 antisymmetry','G3 quadratic','G4 Moufang'),worst): cert(n,e)
    for _ in range(20):
        x=rng.normal(size=16); x[0]=0
        cert('S quadratic identity',np.linalg.norm(mul(x,x)+np.dot(x,x)*E16[0]))

    Z=crack(); L=np.array([left(z) for z in Z]); M=np.array([k.T@k for k in L])
    exact('basic crack cardinality',len(Z),84)
    cert('zero-divisor kernels',max(abs(np.linalg.matrix_rank(k,tol=1e-9)-12) for k in L),.5)
    want=np.array([0]*4+[1]*8+[2]*4)
    cert('metric spectra {0^4,1^8,2^4}',max(np.linalg.norm(np.linalg.eigvalsh(m)-want,np.inf) for m in M))
    cert('forced equilibrium',np.linalg.norm(M.mean(0)-E16))
    P=np.diag([0]+[1]*7+[0]+[1]*7)
    cert('84-design second moment',np.linalg.norm(np.einsum('ai,aj->ij',Z,Z)/84-P/14))
    J=right(E16[8]); cert('J squared',np.linalg.norm(J@J+E16)); cert('J antisymmetric',np.linalg.norm(J+J.T))
    cert('axis left/right commute',np.linalg.norm(left(E16[8])@J-J@left(E16[8])))
    cert('event/axis anticommutation',max(np.linalg.norm(left(E16[8])@k+k@left(E16[8])) for k in L))
    Phi=sum(np.kron(k.T,k.T) for k in L)/84
    vals=np.linalg.eigvalsh(Phi); targets=np.array([-1,-3/7,-2*np.sqrt(3)/7,-1/7,0,1/7,2*np.sqrt(3)/7,3/7,1])
    cert('nine-level channel spectrum',max(min(abs(v-targets)) for v in vals))
    exact('channel trace',round(float(np.trace(Phi)),10),0.0)
    adj={i:[j for j in range(84) if i!=j and np.linalg.norm(L[i]@Z[j])<TOL] for i in range(84)}
    exact('annihilation degrees',sorted(set(map(len,adj.values()))),[4])
    parts=components(adj); exact('annihilation component sizes',sorted(map(len,parts)),[12]*7)
    gens8=[left(E8[i]) for i in range(1,8)]; gens16=[left(E16[i]) for i in range(1,16)]
    exact('octonion associative envelope dimension',closure_dimension(gens8),64)
    exact('sedenion Lie closure dimension',closure_dimension(gens16,lie=True),120)
    if FAIL: print(f"FAILED: {len(FAIL)} certificate(s): {', '.join(FAIL)}"); return 1
    print('PASSED: independent Paper I certificate ledger')
    return 0

if __name__ == '__main__': raise SystemExit(main())
