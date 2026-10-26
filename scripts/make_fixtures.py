"""Generate the synthetic fixtures under tests/fixtures/ (deterministic; committed)."""

import json
import random
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def psych101(rng: random.Random) -> None:
    rows = []
    for exp, keys in (("fixture2020odd/exp1.csv", "DPH"), ("fixture2021bandit/exp2.csv", "AB")):
        for pid in range(1, 9):
            lines = ["You will choose between keys " + ", ".join(keys) + " in each trial."]
            for t in range(15):
                lines.append(
                    f"Trial {t % 5}: item {rng.randint(0, 9)}. You press <<{rng.choice(keys)}>>."
                )
            rows.append({"text": "\n".join(lines), "experiment": exp, "participant": str(pid)})
    with (OUT / "psych101_rows.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def twin2k(rng: random.Random) -> None:
    d = OUT / "twin2k"
    d.mkdir(exist_ok=True)
    mc = lambda qid, text, opts, block: {  # noqa: E731
        "QuestionID": qid,
        "QuestionText": text,
        "QuestionType": "MC",
        "Options": opts,
        "Settings": {"Selector": "SAVR", "SubSelector": "TX"},
        "BlockName": block,
        "csv_columns": [qid],
    }
    cat = [
        mc(
            "QID11",
            "Which part of the United States do you currently live in?",
            ["Northeast", "Midwest", "South", "West"],
            "Demographics",
        ),
        mc(
            "QID12",
            "What is the sex that you were assigned at birth?",
            ["Male", "Female"],
            "Demographics",
        ),
        mc("QID13", "How old are you?", ["18-29", "30-49", "50-64", "65+"], "Demographics"),
        mc("QID14", "Highest schooling?", ["HS", "College", "Grad"], "Demographics"),
        {
            "QuestionID": "QID100",
            "QuestionText": "Intro",
            "QuestionType": "DB",
            "is_descriptive": True,
            "BlockName": "Personality",
        },
        mc(
            "QID157",
            "Disease program preference (gain frame)?",
            ["fav A", "slight A", "slight B", "fav B"],
            "Disease - gain",
        ),
        mc(
            "QID158",
            "Disease program preference (loss frame)?",
            ["fav A", "slight A", "slight B", "fav B"],
            "Disease-loss",
        ),
        mc(
            "QID101",
            "Would you buy the product at this price?",
            ["Yes", "No"],
            "Product Preferences - Pricing",
        ),
        mc(
            "QID102",
            "Which framing do you prefer for the sunk cost problem?",
            ["Keep going", "Stop", "Unsure", "Ask others"],
            "Sunk cost - yes",
        ),
        {
            "QuestionID": "QID25",
            "QuestionText": "I see myself as someone who",
            "QuestionType": "Matrix",
            "Rows": [f"trait {i}" for i in range(6)],
            "Columns": [
                "Disagree strongly",
                "Disagree a little",
                "Neither",
                "Agree a little",
                "Agree strongly",
            ],
            "Settings": {"Selector": "Likert", "SubSelector": "SingleAnswer"},
            "BlockName": "Personality",
            "csv_columns": [f"QID25_{i}" for i in range(1, 7)],
        },
        {
            "QuestionID": "QID84",
            "QuestionText": "Choose the left or right option on each line.",
            "QuestionType": "Matrix",
            "Rows": [
                "$6 in 6 weeks:$3 in 5 weeks",
                "$6 in 6 weeks:$4 in 5 weeks",
                "$6 in 6 weeks:$5 in 5 weeks",
            ],
            "Columns": ["1", "2"],
            "Settings": {"Selector": "Bipolar", "SubSelector": ""},
            "BlockName": "Economic preferences",
            "csv_columns": ["QID84_1", "QID84_2", "QID84_3"],
        },
        {
            "QuestionID": "QID156",
            "QuestionText": "Slide",
            "QuestionType": "Slider",
            "Range": {"Min": 0, "Max": 100},
            "Settings": {"Selector": "HSLIDER"},
            "BlockName": "Cognitive tests ",
            "csv_columns": ["QID156"],
        },
        {
            "QuestionID": "QID10",
            "QuestionText": "Words",
            "QuestionType": "TE",
            "Settings": {"Selector": "FORM"},
            "BlockName": "Forward Flow",
            "csv_columns": ["QID10_1", "QID10_2"],
        },
        {
            "QuestionID": "QID126",
            "QuestionText": "",
            "QuestionType": "MC",
            "Options": ["a", "b", "c", "d"],
            "Settings": {"Selector": "MAVR", "SubSelector": "TX"},
            "BlockName": "Cognitive tests ",
            "csv_columns": ["QID126_1", "QID126_2", "QID126_3", "QID126_4"],
        },
    ]
    (d / "question_catalog.json").write_text(json.dumps(cat, indent=1))
    n_opts = {"QID11": 4, "QID12": 2, "QID13": 4, "QID14": 3, "QID101": 2, "QID102": 4}
    opts = {
        q["QuestionID"]: q["Options"] for q in cat if q["QuestionType"] == "MC" and "Options" in q
    }
    num, lab = [], []
    rows4 = []
    for pid in range(1, 31):
        row = {"pid": pid}
        for qid, n in n_opts.items():
            row[qid] = rng.randint(1, n)
        # between-subject: each person sees one frame; the loss frame shifts answers towards B
        arm = random.Random(pid)
        row["QID157" if pid % 2 else "QID158"] = arm.randint(1, 3) if pid % 2 else arm.randint(2, 4)
        for i in range(1, 7):
            row[f"QID25_{i}"] = rng.randint(1, 5)
        for i in range(1, 4):
            row[f"QID84_{i}"] = rng.randint(1, 2)
        row["QID156"] = rng.randint(0, 100)
        row["QID10_1"] = "apple"
        row["QID10_2"] = "tree"
        for i in range(1, 5):
            row[f"QID126_{i}"] = rng.choice([1, None])
        num.append(row)
        lab.append(
            {
                **{k: (opts[k][v - 1] if k in opts else v) for k, v in row.items() if k != "pid"},
                "pid": pid,
            }
        )
        # wave 4 repeats three items for most persons; a few skip one (missing, like real data)
        r4 = {
            "pid": pid,
            "QID101": rng.randint(1, 2),
            "QID102": rng.randint(1, 4),
            "QID84_1": rng.randint(1, 2),
        }
        if pid % 7 == 0:
            r4["QID84_1"] = None
        rows4.append(r4)
    pd.DataFrame(num).to_csv(d / "wave1_3_response.csv", index=False)
    pd.DataFrame(lab).to_csv(d / "wave1_3_response_label.csv", index=False)
    pd.DataFrame(rows4).to_csv(d / "wave4_response.csv", index=False)


def tiny_vocab() -> None:
    """Character-level WordPiece vocab: every printable ASCII char, plain and as a ## suffix."""
    import string

    chars = [c for c in string.ascii_lowercase + string.digits + string.punctuation]
    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *chars, *[f"##{c}" for c in chars]]
    (OUT / "tiny_vocab.txt").write_text("\n".join(tokens) + "\n")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    psych101(random.Random(0))
    twin2k(random.Random(0))
    tiny_vocab()
    print(f"wrote fixtures to {OUT}")
