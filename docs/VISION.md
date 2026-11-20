# Vision: a decision model for studying society

**The problem.** Social outcomes come from many different people each making a choice, and the result is often one nobody intended (Schelling's point). The average person does not predict it; the spread of answers does. Schelling is a model that takes a typed question and a description of a person and returns a calibrated probability for each answer. Sample one answer per person across a realistic population and you get a synthetic society with real disagreements, minorities and tails.

**What it is for.** Asking "what if" at population scale, before spending on real people:

- preview the answer distribution of a new survey question;
- compare two policy designs and see which groups move, not just the average;
- drive agent-based simulations with learned decisions instead of hand-written rules.

It does not replace real data; it shows where to look. Its worth rests on measurable properties: calibration, human-like variance, and reproducing known experimental effects without inventing new ones.

**Why a decision model, not a role-playing language model.**

- *Variance:* a language model gives the plausible, typical answer, so its populations are too uniform and too confident.
- *Typed outputs:* probabilities over fixed answers can be trained, scored and calibrated directly against what people chose.
- *Stability:* no parsing, and no drift with option order or prompt wording.
- *Cost:* small and fast enough to simulate millions of decisions, reproducibly.

It can still borrow a language model's breadth on unseen questions, while human data has the final word on how people spread across the answers.
