"""C01 first task: check history masking on real Twin-2K-500 validation rows (15 persons)."""

from pathlib import Path

from schelling.data.loader import load_records
from schelling.eval.report import partition_frame, select_partition
from schelling.infer.history import history_for_frame

frame = load_records(Path("data/processed"))
parted = partition_frame(frame, "heldout_person", Path("splits"))
val = select_partition(parted, "val").reset_index(drop=True)
val = val[val["source"] == "twin2k"].reset_index(drop=True)
persons = sorted(val["person_id"].unique())[:15]
sub = val[val["person_id"].isin(persons)].reset_index(drop=True)
hist = history_for_frame(sub, k=50)
own_text = sum(any(x.question == q for x in h) for q, h in zip(sub["question"], hist, strict=True))
print(
    "rows", len(sub), "persons", len(persons), "rows with own question text in history:", own_text
)
print("wave-4 rows", int((sub["wave"] == 4).sum()))
pairs = sub[sub["wave"] != 4].groupby(["person_id", "question"])["item_id"].nunique()
print("person-question pairs with more than one item id:", int((pairs > 1).sum()))
