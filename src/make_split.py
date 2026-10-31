import json, random
from pymatgen.core import Composition

HARD_ELEMENT = "Pd"
data = json.load(open("data/mp_alloys.json"))
info = {}
for r in data:
    els = {e.symbol for e in Composition(r["formula"]).elements}
    info[r["material_id"]] = (els, "-".join(sorted(els)))

hard = [i for i, (els, _) in info.items() if HARD_ELEMENT in els]
rest = [i for i in info if i not in set(hard)]
chemsys = sorted({info[i][1] for i in rest})
random.Random(0).shuffle(chemsys)
n = len(chemsys)
test_cs = set(chemsys[: n // 10]); val_cs = set(chemsys[n // 10 : n // 5])
split = {"train": [], "val": [], "test_chemsys": [], "test_hard": hard}
for i in rest:
    cs = info[i][1]
    key = "test_chemsys" if cs in test_cs else "val" if cs in val_cs else "train"
    split[key].append(i)
json.dump(split, open("data/split.json", "w"))
print({k: len(v) for k, v in split.items()})
