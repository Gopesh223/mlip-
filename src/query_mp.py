import os, json
from mp_api.client import MPRester

BAD = {"H","He","B","C","N","O","F","Ne","Si","P","S","Cl","Ar","Ge","As","Se",
       "Br","Kr","Te","I","Xe","At","Rn"}
def ok(elements):
    return all(e.symbol not in BAD and e.Z <= 83 for e in elements)

def vrh(x):
    """bulk_modulus / shear_modulus come back as dict-like or object with .vrh"""
    if x is None: return None
    if isinstance(x, dict): return x.get("vrh")
    return getattr(x, "vrh", None)

fields = ["material_id", "formula_pretty", "structure", "nsites", "nelements",
          "elements", "energy_above_hull", "formation_energy_per_atom",
          "is_magnetic", "total_magnetization", "bulk_modulus", "shear_modulus",
          "volume"]

with MPRester(os.environ["MP_API_KEY"]) as mpr:
    docs = mpr.materials.summary.search(
        num_elements=(1, 3), num_sites=(1, 16),
        energy_above_hull=(0, 0.05), is_metal=True,
        has_props=["elasticity"], fields=fields)

records = []
for d in docs:
    if not ok(d.elements): continue
    K, G = vrh(d.bulk_modulus), vrh(d.shear_modulus)
    if K is None or G is None: continue
    if not (0 < K < 500 and 0 < G < 400): continue  # drop DFT outliers
    records.append({
        "material_id": str(d.material_id), "formula": d.formula_pretty,
        "nelements": d.nelements, "nsites": d.nsites,
        "e_hull": d.energy_above_hull, "e_form": d.formation_energy_per_atom,
        "is_magnetic": bool(d.is_magnetic), "total_mag": d.total_magnetization,
        "K_mp": K, "G_mp": G,
        "structure": d.structure.as_dict()})

import random; random.Random(0).shuffle(records)
json.dump(records, open("data/mp_alloys.json", "w"))
print(len(docs), "fetched ->", len(records), "kept")
