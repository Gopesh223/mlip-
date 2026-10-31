import sys, json
import numpy as np, pandas as pd
from pymatgen.core import Composition
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

model = sys.argv[1]
mp = {r["material_id"]: r for r in json.load(open("data/mp_alloys.json"))}
df = pd.read_csv(f"results/{model}_features.csv")   # made by error_model.py
ml = [json.loads(l) for l in open(f"results/{model}_zeroshot.jsonl")]
ml = [r for r in ml if "error" not in r][:len(df)]
assert all(mp[r["material_id"]]["formula"] == f for r, f in zip(ml, df["formula"])), \
    "row mismatch: rerun error_model.py first"
df["K_abs"] = np.clip([r["K_ml"] - mp[r["material_id"]]["K_mp"] for r in ml], -200, 200)
df["G_abs"] = np.clip([r["G_ml"] - mp[r["material_id"]]["G_mp"] for r in ml], -200, 200)

meta = ["K_rel_err", "G_rel_err", "K_abs", "G_abs", "unstable", "chemsys", "formula"]
DFT_FEATS = ["e_form", "e_hull", "total_mag", "is_magnetic"]
allcols = [c for c in df.columns if c not in meta]
featsets = {"all": allcols, "composition_only": [c for c in allcols if c not in DFT_FEATS]}
targets = ["K_rel_err", "G_rel_err", "K_abs", "G_abs"]

df["els"] = [set(e.symbol for e in Composition(f).elements) for f in df["formula"]]
freq = pd.Series([s for S in df["els"] for s in S]).value_counts()
top_elems = list(freq.index[:12])
print("Leave-one-element-out on:", top_elems)

def make(kind):
    if kind == "rf":
        return RandomForestRegressor(200, min_samples_leaf=3, n_jobs=8, random_state=0)
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12)))

out = []
groups = df["chemsys"].values
gkf = GroupKFold(n_splits=5)
for fs_name, cols in featsets.items():
    X = df[cols].copy(); X = X.fillna(X.median())
    for t in targets:
        y = df[t].values
        for kind in ["rf", "ridge"]:
            # scheme 1: group by exact chemical system
            pred = np.zeros(len(y))
            for tr, te in gkf.split(X, y, groups):
                pred[te] = make(kind).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
            out.append(dict(featset=fs_name, target=t, model=kind,
                            scheme="group_chemsys", R2=r2_score(y, pred), n_folds=5))
            # scheme 2: leave-one-element-out (train WITHOUT element, test WITH it)
            r2s = []
            for E in top_elems:
                has = df["els"].apply(lambda s: E in s).values
                if has.sum() < 30 or (~has).sum() < 200: continue
                p = make(kind).fit(X[~has], y[~has]).predict(X[has])
                r2s.append(r2_score(y[has], p))
            out.append(dict(featset=fs_name, target=t, model=kind,
                            scheme="leave_one_element_out_mean", R2=np.mean(r2s), n_folds=len(r2s)))
            out.append(dict(featset=fs_name, target=t, model=kind,
                            scheme="leave_one_element_out_median", R2=np.median(r2s), n_folds=len(r2s)))
            print(fs_name, t, kind, "done", flush=True)

res = pd.DataFrame(out)
res.to_csv(f"results/ablation_{model}.csv", index=False)
pd.set_option("display.width", 200)
print(res.pivot_table(index=["featset", "target", "model"], columns="scheme", values="R2").round(3))
