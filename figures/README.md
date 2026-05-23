# figures/

This directory is reserved for figures referenced from
[`README.md`](../README.md) or [`docs/paper.md`](../docs/paper.md).

## Tracking policy

- **Tracked**: only figures explicitly cited in `README.md` or `docs/paper.md`.
- **Not tracked**: experiment outputs, intermediate plots, exploratory
  rendering. These belong under [`results/`](../results/) (gitignored) or
  `figures/evaluation/` and `figures/comparison/` (both gitignored under
  `.gitignore`).

The repository previously tracked unreferenced figures from an unrelated
LDPC experiment; those are now removed from version control. Local copies
are kept in `~/morphling-figures-backup/<timestamp>/tracked/` for the
maintainer.

## Regenerating figures

Paper figures are produced by experiment scripts that are **not** shipped
in this open-source release (they targeted a private dataset and a
companion baselines repository). See [`docs/paper.md`](../docs/paper.md)
for the figure inventory and the data-availability statement.
