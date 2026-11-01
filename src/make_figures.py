import json, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Figure 1: shear parity, MACE vs CHGNet, all 3278 structures
fig, ax = plt.subplots(1, 2, figsize=(10, 4.6))
for a, m, name in zip(ax, ["mace", "chgnet"], ["MACE-MP-0", "CHGNet"]):
    d = pd.read_csv(f"results/{m}_table.csv")
    a.scatter(d.G_mp, d.G_ml, s=4, alpha=0.35)
    a.plot([-50, 400], [-50, 400], "k--", lw=1)
    a.set(xlim=(0, 260), ylim=(-60, 360), xlabel="Shear modulus, Materials Project (GPa)",
          ylabel="Shear modulus, ML potential (GPa)",
          title=f"{name}: median ratio {np.median(d.G_ml / d.G_mp):.2f}")
plt.tight_layout()
plt.savefig("results/fig_shear_parity.png", dpi=200)

# Figure 2: zero-shot vs fine-tuned, by evaluation set
mp = {r["material_id"]: r for r in json.load(open("data/mp_alloys.json"))}
split = json.load(open("data/split.json"))
def load(t):
    d = {}
    for l in open(f"results/{t}_zeroshot.jsonl"):
        r = json.loads(l)
        if "error" not in r:
            d[r["material_id"]] = r
    return d
base, ft = load("mace"), load("mace_ft2")
sets = ["val", "test_chemsys", "test_hard"]
labels = ["val (used to pick epoch)", "test (new chem. systems)", "test (Pd, unseen element)"]
res = {"G ratio": ([], []), "K MAE": ([], [])}
for s in sets:
    ids = [i for i in split[s] if i in base and i in ft]
    for p, key, fn in [("G", "G ratio", lambda m, r: np.median(m / r)),
                       ("K", "K MAE", lambda m, r: np.abs(m - r).mean())]:
        ref = np.array([mp[i][f"{p}_mp"] for i in ids])
        for k, d in enumerate([base, ft]):
            res[key][k].append(fn(np.array([d[i][f"{p}_ml"] for i in ids]), ref))
x = np.arange(len(sets)); w = 0.38
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
for a, key, ylab in zip(ax, ["G ratio", "K MAE"],
                        ["Median shear ratio, ML / MP (1 = perfect)", "Bulk modulus MAE (GPa)"]):
    a.bar(x - w / 2, res[key][0], w, label="MACE-MP-0 (zero-shot)")
    a.bar(x + w / 2, res[key][1], w, label="fine-tuned")
    a.set_xticks(x); a.set_xticklabels(labels, rotation=12, ha="right"); a.set_ylabel(ylab)
ax[0].axhline(1, color="k", ls="--", lw=1); ax[0].set_ylim(0, 1.35); ax[0].legend(loc="upper left")
plt.tight_layout()
plt.savefig("results/fig_finetune.png", dpi=200)
print("saved results/fig_shear_parity.png and results/fig_finetune.png")
