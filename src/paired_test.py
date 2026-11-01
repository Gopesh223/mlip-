import json, sys, numpy as np
from scipy.stats import wilcoxon

mp = {r["material_id"]: r for r in json.load(open("data/mp_alloys.json"))}
split = json.load(open("data/split.json"))
def load(tag):
    d = {}
    for l in open(f"results/{tag}_zeroshot.jsonl"):
        r = json.loads(l)
        if "error" not in r:
            d[r["material_id"]] = r
    return d

base, ft = load("mace"), load(sys.argv[1])
rng = np.random.default_rng(0)
for name in ["val", "test_chemsys", "test_hard"]:
    ids = [i for i in split[name] if i in base and i in ft]
    for p in ["K", "G"]:
        ref = np.array([mp[i][f"{p}_mp"] for i in ids])
        e0 = np.abs(np.array([base[i][f"{p}_ml"] for i in ids]) - ref)
        e1 = np.abs(np.array([ft[i][f"{p}_ml"] for i in ids]) - ref)
        diff = e1 - e0                      # negative = fine-tuned is better
        boots = [rng.choice(diff, len(diff)).mean() for _ in range(5000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        pval = wilcoxon(diff).pvalue
        print(f"{name:13s} {p}: mean change in |error| = {diff.mean():+6.2f} GPa "
              f"[95% CI {lo:+.2f}, {hi:+.2f}]  Wilcoxon p={pval:.3g}  (n={len(ids)})")
