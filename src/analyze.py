import json, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

mp = {r["material_id"]: r for r in json.load(open("data/mp_alloys.json"))}

def load(name):
    rows = []
    for l in open(f"results/{name}_zeroshot.jsonl"):
        r = json.loads(l)
        if "error" in r: continue
        m = mp[r["material_id"]]
        C = np.array(r["C"])
        rows.append(dict(material_id=r["material_id"], formula=m["formula"],
                         nelements=m["nelements"], is_magnetic=m["is_magnetic"],
                         K_mp=m["K_mp"], G_mp=m["G_mp"],
                         K_ml=r["K_ml"], G_ml=r["G_ml"],
                         vol_ratio=r["vol_ratio"],
                         min_eig=np.linalg.eigvalsh(0.5*(C+C.T)).min()))
    return pd.DataFrame(rows)

models = {n: load(n) for n in sys.argv[1:]}
fig, ax = plt.subplots(len(models), 2, figsize=(9, 4.2*len(models)), squeeze=False)
for i, (n, df) in enumerate(models.items()):
    for j, p in enumerate(["K", "G"]):
        x, y = df[f"{p}_mp"], df[f"{p}_ml"]
        lim = [min(x.min(), y.min(), 0), max(x.max(), y.max())*1.05]
        ax[i][j].scatter(x, y, s=6, alpha=.5)
        ax[i][j].plot(lim, lim, "k--", lw=1)
        mae = (y-x).abs().mean()
        ax[i][j].set(title=f"{n}: {p} (MAE {mae:.1f} GPa, n={len(df)})",
                     xlabel=f"{p} MP (GPa)", ylabel=f"{p} ML (GPa)")
    unstable = (df.min_eig <= 0).sum()
    print(f"{n}: n={len(df)}, median K ratio {(df.K_ml/df.K_mp).median():.3f}, "
          f"median G ratio {(df.G_ml/df.G_mp).median():.3f}, "
          f"mechanically unstable {unstable} ({100*unstable/len(df):.1f}%)")
plt.tight_layout(); plt.savefig("results/parity.png", dpi=150)
for n, df in models.items(): df.to_csv(f"results/{n}_table.csv", index=False)
