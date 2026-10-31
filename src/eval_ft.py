import json, sys, time
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from mace.calculators import MACECalculator
from elastic import relax_full, elastic_tensor, vrh_moduli

model_path, tag = sys.argv[1], sys.argv[2]
calc = MACECalculator(model_paths=model_path, device="cuda", default_dtype="float64")
split = json.load(open("data/split.json"))
ids = set(split["test_chemsys"]) | set(split["test_hard"]) | set(split["val"])
data = [r for r in json.load(open("data/mp_alloys.json")) if r["material_id"] in ids]
out = f"results/{tag}_zeroshot.jsonl"
done = set()
try:
    done = {json.loads(l)["material_id"] for l in open(out)}
except FileNotFoundError:
    pass
with open(out, "a") as f:
    for r in data:
        if r["material_id"] in done:
            continue
        t0 = time.time()
        try:
            s = Structure.from_dict(r["structure"])
            if "magmom" in s.site_properties:
                s.remove_site_property("magmom")
            atoms = AseAtomsAdaptor.get_atoms(s)
            atoms.calc = calc
            relax_full(atoms)
            C = elastic_tensor(atoms)
            K, G = vrh_moduli(C)
            res = dict(material_id=r["material_id"], K_ml=K, G_ml=G,
                       vol_ratio=atoms.get_volume() / s.volume,
                       C=C.tolist(), time_s=time.time() - t0)
        except Exception as e:
            res = dict(material_id=r["material_id"], error=str(e))
        f.write(json.dumps(res) + "\n")
        f.flush()
