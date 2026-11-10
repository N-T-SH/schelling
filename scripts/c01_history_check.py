"""C01 first task: check history masking on real Twin-2K-500 validation rows (15 persons)."""

import pandas as pd
from pathlib import Path
from schelling.data.loader import load_records
from schelling.eval.report import partition_frame, select_partition
from schelling.infer.history import history_for_frame

fr = load_records(Path("data/processed"))
parted = partition_frame(fr, "heldout_person", Path("splits"))
val = select_partition(parted, "val").reset_index(drop=True)
val = val[val["source"] == "twin2k"].reset_index(drop=True)
pids = sorted(val["person_id"].unique())[:15]
sub = val[val["person_id"].isin(pids)].reset_index(drop=True)
hist = history_for_frame(sub, k=50)
bad = 0
dup_text = 0
for (rid, q, item), h in zip(zip(sub["record_id"], sub["question"], sub["item_id"]), hist):
    if any(x.question == q for x in h):
        bad += 1
print("rows", len(sub), "persons", len(pids), "rows whose history contains own question text:", bad)
print("wave4 rows", int((sub.wave == 4).sum()), "mean history len", sum(map(len, hist)) / len(hist))
g = sub[sub.wave != 4].groupby(["person_id", "question"])["item_id"].nunique()
print("person-question pairs with >1 item id (duplicate text):", int((g > 1).sum()))
print("partitions in val frame:", sorted(val["wave"].unique()))
