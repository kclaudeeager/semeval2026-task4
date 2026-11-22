# Archived runs

Runs kept for the record but deliberately outside `reports/results/`, so
`scripts/audit.py` and `scripts/export_numbers.py` do not treat them as
live results.

## dev__l5_combined__over-L3n__IN-SAMPLE-NOT-CV.json

The 59.00% figure. Weights were selected on the development split and the
same split was then scored, so the number is optimistic by an unknown
amount. It predates the cross-validation path in `evaluate.py`.

Cite it only in Limitations, as an illustration of what in-sample fitting
buys, and never alongside the cross-validated layers in a results table.
Against the L2 backbone the gap is +0.50 points at p = 1.000.
