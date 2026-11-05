"""Retry curve (retry vs retry_frozen, run.sh here): success and damage within n grasps, paired by case, object-bootstrap
intervals; the subset whose first menu gives the best row P(success) >= 0.2; prediction quality after a failure.
Run: python analysis.py (from this folder)."""
import collections, gzip, json, os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IRR={"dropped_high","rolled_off","crushed","bruised","broken","spilled"}
R={}
for p in ("retry","retry_frozen"):
    d=json.load(gzip.open(os.path.join(HERE, f"{p}_k1_A10_s1_seed12.json.gz"))); rows=d["rows"]
    ok=[r for r in rows if "error" not in r and not r.get("artifact")]
    R[p]={(r["object"],r["idx"]):r for r in ok}
    print(p,"rows",len(rows),"ok",len(ok),"err",sum("error" in r for r in rows),"art",sum(bool(r.get("artifact")) for r in rows),
          dict(collections.Counter(r.get("ended") for r in ok)),"secs",d.get("seconds"),d["code"]["revision"][:7],d["code"].get("changed_during_run"))
keys=sorted(set(R["retry"])&set(R["retry_frozen"]))
objs=sorted({k[0] for k in keys}); oi={o:i for i,o in enumerate(objs)}
def stats(sub):
    out={}
    for p in R:
        att=[R[p][k].get("attempts") or [] for k in sub]
        S=np.array([[any(t["label"]=="placed_stable" for t in x[:n]) for n in range(1,6)] for x in att],float)
        D=np.array([[any(t["label"] in IRR for t in x[:n]) for n in range(1,6)] for x in att],float)
        out[p]=(S,D)
    return out
rng=np.random.default_rng(0)
def report(name,sub):
    st=stats(sub); g=np.array([oi[k[0]] for k in sub])
    print(f"\n== {name}: {len(sub)} paired cases, {len(set(g))} objects")
    for p in st:
        S,D=st[p]; print(f"  {p:13s} success<=n "+" ".join(f"{100*v:5.1f}" for v in S.mean(0))+" | damage<=n "+" ".join(f"{100*v:4.1f}" for v in D.mean(0)))
    dS=st["retry"][0]-st["retry_frozen"][0]; dD=st["retry"][1]-st["retry_frozen"][1]
    uo=np.unique(g); idx={o:np.where(g==o)[0] for o in uo}
    bs=[]; bd=[]
    for _ in range(2000):
        pick=np.concatenate([idx[o] for o in rng.choice(uo,len(uo))]); bs.append(dS[pick].mean(0)); bd.append(dD[pick].mean(0))
    bs=np.array(bs); bd=np.array(bd)
    for n in (2,3,5):
        print(f"  updated-frozen n={n}: success {100*dS.mean(0)[n-1]:+.1f} [{100*np.percentile(bs[:,n-1],2.5):+.1f}, {100*np.percentile(bs[:,n-1],97.5):+.1f}]  damage {100*dD.mean(0)[n-1]:+.1f} [{100*np.percentile(bd[:,n-1],2.5):+.1f}, {100*np.percentile(bd[:,n-1],97.5):+.1f}]")
    for p in R:
        att=[R[p][k].get("attempts") or [] for k in sub]
        print("   ",p,"per attempt: "+" | ".join(f"k{k}: n{len(a)} S{100*np.mean([t['label']=='placed_stable' for t in a]):.0f}% D{100*np.mean([t['label'] in IRR for t in a]):.1f}%" for k in range(1,6) for a in [[x[k-1] for x in att if len(x)>=k]] if a))
report("all cases",keys)
feas=[k for k in keys if R["retry"][k].get("attempts") and R["retry"][k]["attempts"][0]["declared"]["p"]["placed_stable"]>=0.2]
report("first menu best P(success) >= 0.2",feas)
print("\n== attempts k>=2: mean predicted P(success) vs observed, Brier")
for p in R:
    pr=[];ob=[]
    for k in keys:
        for t in (R[p][k].get("attempts") or [])[1:]:
            pr.append(t["declared"]["p"]["placed_stable"]); ob.append(t["label"]=="placed_stable")
    pr=np.array(pr); ob=np.array(ob,float)
    print(f"  {p:13s} n={len(pr)} predicted {pr.mean():.3f} observed {ob.mean():.3f} Brier {np.mean((pr-ob)**2):.4f}")
