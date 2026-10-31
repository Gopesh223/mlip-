import json, sys, numpy as np
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

def n_unstable(d, ids):
    n = 0
    for i in ids:
        C = np.array(d[i]["C"])
        n += np.linalg.eigvalsh(0.5 * (C + C.T)).min() <= 0
    return n

for name in ["val", "test_chemsys", "test_hard"]:
    ids = [i for i in split[name] if i in base and i in ft]
    if not ids:
        continue
    for p in ["K", "G"]:
        ref = np.array([mp[i][f"{p}_mp"] for i in ids])
        for tag, d in [("zero-shot", base), ("fine-tuned", ft)]:
            ml = np.array([d[i][f"{p}_ml"] for i in ids])
            print(f"{name:13s} n={len(ids):4d} {p} {tag:10s} "
                  f"MAE={np.abs(ml - ref).mean():6.2f} GPa  "
                  f"median ratio={np.median(ml / ref):.3f}")
    print(f"{name:13s} unstable: zero-shot {n_unstable(base, ids)}, "
          f"fine-tuned {n_unstable(ft, ids)} of {len(ids)}")
