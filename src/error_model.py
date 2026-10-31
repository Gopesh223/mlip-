import json, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pymatgen.core import Composition
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, roc_auc_score
from scipy.stats import spearmanr

model = sys.argv[1]  # "mace" or "chgnet"
mp = {r["material_id"]: r for r in json.load(open("data/mp_alloys.json"))}

PROPS = ["X", "atomic_radius", "atomic_mass", "row", "group", "melting_point", "molar_volume", "Z"]
MAGNETIC = {"Fe", "Co", "Ni", "Mn", "Cr"}
D_BLOCK = set(range(21, 31)) | set(range(39, 49)) | set(range(72, 81))
F_BLOCK = set(range(57, 72))                       # lanthanides La-Lu
ALKALI = {3, 11, 19, 37, 55, 4, 12, 20, 38, 56}    # alkali + alkaline earth

def eprop(el, name):
    try:
        v = getattr(el, name)
        return float(v) if v is not None else np.nan
    except Exception:
        return np.nan

def comp_features(formula):
    comp = Composition(formula)
    els = list(comp.elements)
    w = np.array([comp.get_atomic_fraction(e) for e in els])
    f = {}
    for p in PROPS:
        v = np.array([eprop(e, p) for e in els])
        ok = ~np.isnan(v)
        if ok.sum() == 0:
            f[f"{p}_mean"] = f[f"{p}_range"] = f[f"{p}_std"] = np.nan
            continue
        ww, vv = w[ok] / w[ok].sum(), v[ok]
        m = (ww * vv).sum()
        f[f"{p}_mean"] = m
        f[f"{p}_range"] = vv.max() - vv.min()
        f[f"{p}_std"] = np.sqrt((ww * (vv - m) ** 2).sum())
    f["frac_d"] = sum(wi for e, wi in zip(els, w) if e.Z in D_BLOCK)
    f["frac_f"] = sum(wi for e, wi in zip(els, w) if e.Z in F_BLOCK)
    f["frac_alkali"] = sum(wi for e, wi in zip(els, w) if e.Z in ALKALI)
    f["frac_magnetic_el"] = sum(wi for e, wi in zip(els, w) if e.symbol in MAGNETIC)
    return f

rows = []
for l in open(f"results/{model}_zeroshot.jsonl"):
    r = json.loads(l)
    if "error" in r:
        continue
    m = mp[r["material_id"]]
    C = np.array(r["C"])
    feat = comp_features(m["formula"])
    feat.update(nelements=m["nelements"], nsites=m["nsites"],
                e_form=m["e_form"], e_hull=m["e_hull"],
                is_magnetic=float(m["is_magnetic"]), total_mag=m["total_mag"])
    feat["K_rel_err"] = np.clip((r["K_ml"] - m["K_mp"]) / m["K_mp"], -2, 2)
    feat["G_rel_err"] = np.clip((r["G_ml"] - m["G_mp"]) / m["G_mp"], -2, 2)
    feat["unstable"] = int(np.linalg.eigvalsh(0.5 * (C + C.T)).min() <= 0)
    feat["chemsys"] = "-".join(sorted(e.symbol for e in Composition(m["formula"]).elements))
    feat["formula"] = m["formula"]
    rows.append(feat)

df = pd.DataFrame(rows)
meta = ["K_rel_err", "G_rel_err", "unstable", "chemsys", "formula"]
X = df.drop(columns=meta)
X = X.fillna(X.median())
groups = df["chemsys"].values
df.to_csv(f"results/{model}_features.csv", index=False)
print(f"{model}: n={len(df)}, unique chemical systems={len(set(groups))}, features={X.shape[1]}")

gkf = GroupKFold(n_splits=5)

def cv_regress(target):
    y = df[target].values
    pred = np.zeros(len(y))
    imp = np.zeros(X.shape[1])
    for tr, te in gkf.split(X, y, groups):
        rf = RandomForestRegressor(300, min_samples_leaf=3, n_jobs=-1, random_state=0).fit(X.iloc[tr], y[tr])
        pred[te] = rf.predict(X.iloc[te])
        pi = permutation_importance(rf, X.iloc[te], y[te], n_repeats=5, random_state=0, n_jobs=-1)
        imp += pi.importances_mean / 5
    print(f"  {target}: CV R2 = {r2_score(y, pred):.3f}, Spearman = {spearmanr(y, pred)[0]:.3f}")
    return imp

print("Predicting signed relative error (CV grouped by chemical system):")
impK = cv_regress("K_rel_err")
impG = cv_regress("G_rel_err")

y = df["unstable"].values
if y.sum() >= 10 and (len(y) - y.sum()) >= 10:
    prob = np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups):
        clf = RandomForestClassifier(300, min_samples_leaf=3, class_weight="balanced",
                                     n_jobs=-1, random_state=0).fit(X.iloc[tr], y[tr])
        prob[te] = clf.predict_proba(X.iloc[te])[:, 1]
    print(f"  mechanical instability: ROC-AUC = {roc_auc_score(y, prob):.3f} "
          f"({y.sum()} unstable of {len(y)})")
else:
    print(f"  not enough unstable/stable examples yet ({y.sum()} unstable of {len(y)})")

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for a, imp, t in zip(ax, [impK, impG], ["K error", "G error"]):
    s = pd.Series(imp, index=X.columns).sort_values().tail(12)
    a.barh(s.index, s.values)
    a.set_title(f"{model}: what predicts {t}?")
    a.set_xlabel("permutation importance (held-out chemistries)")
plt.tight_layout()
plt.savefig(f"results/{model}_error_importance.png", dpi=150)
print("saved results/%s_error_importance.png" % model)