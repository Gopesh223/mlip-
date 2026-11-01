import numpy as np, pandas as pd
edges = [0, 20, 40, 60, 80, 120, 400]
for m in ["mace", "chgnet"]:
    d = pd.read_csv(f"results/{m}_table.csv")
    d["bin"] = pd.cut(d.G_mp, edges)
    g = d.groupby("bin", observed=True)
    out = pd.DataFrame({
        "n": g.size(),
        "median_G_mp": g.G_mp.median().round(1),
        "median_G_ml": g.G_ml.median().round(1),
        "median_ratio": g.apply(lambda x: (x.G_ml / x.G_mp).median(), include_groups=False).round(2),
    })
    print(m); print(out.to_string()); print()
