import json, ijson, os, sys

SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/mptrj/MPtrj_2022.9_full.json")
ids = {r["material_id"] for r in json.load(open("data/mp_alloys.json"))}
print("wanted materials:", len(ids))

subset, n_frames, seen = {}, 0, 0
with open(SRC, "rb") as f:
    # top level is assumed to be {mp_id: {frame_id: {...}}}
    for mid, frames in ijson.kvitems(f, "", use_float=True):
        seen += 1
        if seen == 1:
            fk = next(iter(frames))
            print("first material:", mid, "| first frame:", fk)
            print("frame keys:", list(frames[fk].keys()))
        if mid in ids:
            subset[mid] = frames
            n_frames += len(frames)
        if seen % 5000 == 0:
            print(f"scanned {seen} materials, kept {len(subset)} ({n_frames} frames)", flush=True)

json.dump(subset, open("data/mptrj_subset.json", "w"))
print(f"done: {len(subset)}/{len(ids)} materials found, {n_frames} frames")
