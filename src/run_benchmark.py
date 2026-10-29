import json, sys, os, time
import numpy as np
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from elastic import relax_full, elastic_tensor, vrh_moduli

model_name = sys.argv[1]            # "mace" or "chgnet"
limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
out = f"results/{model_name}_zeroshot.jsonl"

if model_name == "mace":
    from mace.calculators import mace_mp
    calc = mace_mp(model="medium", device="cuda", default_dtype="float64")
else:
    from chgnet.model.dynamics import CHGNetCalculator
    calc = CHGNetCalculator(use_device="cuda")

data = json.load(open("data/mp_alloys.json"))[:limit]
done = set()
if os.path.exists(out):
    done = {json.loads(l)["material_id"] for l in open(out)}

with open(out, "a") as f:
    for r in data:
        if r["material_id"] in done: continue
        t0 = time.time()
        try:
            s = Structure.from_dict(r["structure"])
            s.remove_site_property("magmom") if "magmom" in s.site_properties else None
            atoms = AseAtomsAdaptor.get_atoms(s)
            atoms.calc = calc
            relax_full(atoms)
            C = elastic_tensor(atoms)
            K, G = vrh_moduli(C)
            vol_ratio = atoms.get_volume() / s.volume
            res = {"material_id": r["material_id"], "K_ml": K, "G_ml": G,
                   "vol_ratio": vol_ratio, "C": C.tolist(),
                   "time_s": time.time() - t0}
        except Exception as e:
            res = {"material_id": r["material_id"], "error": str(e)}
        f.write(json.dumps(res) + "\n"); f.flush()
        print(r["formula"], res.get("K_ml"), res.get("G_ml"))
