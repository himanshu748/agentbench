"""Paired-task analysis. No scores for absent/unevaluated observations."""
import argparse, itertools, json, random, statistics
from pathlib import Path
H=('pi','opencode','mini')
def analyze(rows, seed=20261126, bootstrap=10000):
    seen=set(); by={}; missing=0
    for r in rows:
        k=(r['task_id'],r['harness'],r['attempt'])
        if k in seen: raise ValueError('duplicate attempt')
        seen.add(k)
        if r.get('status')!='evaluated' or type(r.get('reward')) is not int or r['reward'] not in (0,1) or not r.get('model_calls'):
            missing+=1; continue
        by[k]=r['reward']
    if len({r.get('model_id') for r in rows})>1: raise ValueError('analyze one exact model at a time')
    def mean(x): return statistics.mean(x) if x else None
    result={'evaluated_attempts':len(by),'unevaluated_attempts':missing,'harnesses':{},'paired':[]}
    for h in H:
        v=[r for (t,g,a),r in by.items() if g==h]; f=[r for (t,g,a),r in by.items() if g==h and a==1]
        result['harnesses'][h]={'evaluated':len(v),'resolved_rate':mean(v),'pass_at_1_first_attempt':mean(f),'first_attempt_count':len(f)}
    for h,g in itertools.combinations(H,2):
        ds={}
        for (t,x,a),v in by.items():
            if x==h and (t,g,a) in by: ds.setdefault(t,[]).append(v-by[t,g,a])
        d=[statistics.mean(v) for t,v in sorted(ds.items())]; rng=random.Random(seed)
        boots=sorted(statistics.mean(rng.choices(d,k=len(d))) for _ in range(bootstrap)) if len(d)>=2 else []
        p=None
        if 1<=len(d)<=20:
            obs=abs(statistics.mean(d)); p=sum(abs(statistics.mean(x*s for x,s in zip(d,signs)))>=obs-1e-12 for signs in itertools.product((-1,1),repeat=len(d)))/(2**len(d))
        result['paired'].append({'a':h,'b':g,'matched_tasks':len(d),'matched_attempts':sum(map(len,ds.values())),'difference_a_minus_b':mean(d),'bootstrap_95_percentile_ci':[boots[int(.025*bootstrap)],boots[min(bootstrap-1,int(.975*bootstrap))]] if boots else None,'sign_flip_p_unadjusted':p})
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('results',type=Path); a=p.parse_args(); print(json.dumps(analyze(json.loads(a.results.read_text())),indent=2))
