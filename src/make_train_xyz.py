import json, random
import numpy as np
from ase.io import write
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

GPA = 160.21766208
MAX_FRAMES, FMAX = 10, 50.0
d = json.load(open("data/mptrj_subset.json"))
split = json.load(open("data/split.json"))
rng = random.Random(0)

def convert(ids, path):
    out = []
    for mid in ids:
        if mid not in d:
            continue
        keys = sorted(d[mid].keys())
        rng.shuffle(keys)
        n = 0
        for k in keys:
            f = d[mid][k]
            if np.abs(np.array(f["force"])).max() > FMAX:
                continue
            s = Structure.from_dict(f["structure"])
            if "magmom" in s.site_properties:
                s.remove_site_property("magmom")
            at = AseAtomsAdaptor.get_atoms(s)
            at.info["REF_energy"] = float(f["corrected_total_energy"])
            at.info["REF_stress"] = (-np.array(f["stress"]) * 0.1 / GPA).flatten()
            at.arrays["REF_forces"] = np.array(f["force"])
            at.info["mp_id"] = mid
            out.append(at)
            n += 1
            if n >= MAX_FRAMES:
                break
    write(path, out)
    print(path, ":", len(out), "frames from", len({a.info['mp_id'] for a in out}), "materials")

convert(split["train"], "data/train.xyz")
convert(split["val"], "data/valid.xyz")
