import json, numpy as np
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from mace.calculators import mace_mp

GPA = 160.21766208  # GPa per eV/A^3
d = json.load(open("data/mptrj_subset.json"))
frames = [(m, k, f) for m, fr in d.items() for k, f in fr.items()]
frames.sort(key=lambda x: -np.abs(np.array(x[2]["stress"])).max())
top = frames[:300]
rng = np.random.default_rng(0)
pick = [top[i] for i in rng.choice(len(top), 40, replace=False)]
print("largest |raw stress| in subset:", round(np.abs(np.array(frames[0][2]["stress"])).max(), 1), "kBar")

calc = mace_mp(model="medium", device="cuda", default_dtype="float64")
dots, norms, de, df = 0.0, 0.0, [], []
for mid, k, f in pick:
    s = Structure.from_dict(f["structure"])
    if "magmom" in s.site_properties:
        s.remove_site_property("magmom")
    at = AseAtomsAdaptor.get_atoms(s)
    at.calc = calc
    ref = np.array(f["stress"]) * 0.1 / GPA          # kBar -> eV/A^3, raw sign
    pred = at.get_stress(voigt=False)                 # ASE convention, eV/A^3
    dots += (pred * ref).sum(); norms += (ref * ref).sum()
    de.append(abs(at.get_potential_energy() - f["corrected_total_energy"]) / len(at))
    df.append(np.abs(at.get_forces() - np.array(f["force"])).mean())
slope = dots / norms
print(f"stress slope (MACE vs +raw*0.1/160.2): {slope:.2f}   (near +1 -> use +raw, near -1 -> use -raw)")
print(f"energy MAE vs corrected_total_energy: {np.mean(de):.3f} eV/atom (median {np.median(de):.3f})")
print(f"force MAE vs MPtrj force: {np.mean(df):.3f} eV/A")
