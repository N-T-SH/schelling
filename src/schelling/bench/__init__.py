"""Fork D, campaign D01-lite: benchmark external models on Twin-2K-500 decision items.

Models are scored outside the harness: any model that writes a probability per option for each
prompt id can be compared. Metrics reuse `schelling.eval.metrics`.
"""
