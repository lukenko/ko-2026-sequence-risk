# Replication package: A Mathematical Framework for Sequence Risk

Luken Ko, University of California, Berkeley

This repository contains the code and data that reproduce every table, figure, and
numerical result in the paper. Each script runs from a fixed random seed, so all
results regenerate deterministically.

## Quick start

```
pip install -r requirements.txt
python run_all.py
```

`run_all.py` runs every script in paper order. Each script's printed results are saved
to `output/<script>.txt`, and figures are saved to `output/` as PDF. To run selected
scripts only, name them, e.g. `python run_all.py realistic_grid fig_phase`.

The `output/` folder in this repository holds the results of a complete run, for
comparison.

Tested with Python 3.14, NumPy 2.3.5, and Matplotlib 3.11.0. A full run takes about an
hour on a laptop, most of it for Table 5 and the stability check that follows it.

## Map of results to scripts

| Paper result | Script | Runtime |
|---|---|---|
| Section 5: closed-form split, checked against enumeration of all orderings | `verify_split.py` | 40 s |
| Section 5.5: size of the sequencing share | `magnitude.py` | 1 s |
| Figure 1: sequencing share against the cash-flow rate | `fig_phase.py` | 10 s |
| Figure 2: sequencing share by horizon and withdrawal rate | `fig_order_contour.py` | 25 s |
| Table 1: sequencing share of depletion | `realistic_grid.py` | 1.5 min |
| Section 6.1: Student-t returns | `student_t.py` | 45 s |
| Figure 3: shortfall against a floor | `fig_floor.py` | 25 s |
| Table 2: sequencing share of goal attainment | `goal_grid.py` | 2 min |
| Figure 4: attainment as the goal varies | `fig_target.py` | 1 min |
| Section 7: summary statistics, 1966 cohort, Table 3 | `historical.py` | 5 s |
| Figure 5: historical reordering bands | `fig_historical.py` | 2 s |
| Table 4: volatility schedules, depletion and variance | `risk_timing.py` | 15 s |
| Section 8: accumulation counterpart of Table 4 | `accum_timing.py` | 5 s |
| Section 9: five-year variance ratio | `variance_ratio.py` | 1 s |
| Table 5: robustness to serial dependence | `dependence_grid.py` | 17 min |
| Section 9: stability of the importance-sampling estimates | `is_stability.py` | 34 min |
| Section 9: block bootstrap of historical returns | `dependence_realdata.py` | 20 s |

`figstyle.py` sets the figure style, and `verify_split.py` also supplies the closed-form
functions used by `magnitude.py` and the figure scripts. The worked examples in Sections 5
and 6 are small enough to check by hand.

## Data

`data/us_annual_returns_1928_2025.csv` holds the annual series used in Sections 7 and 9,
in percent:

- `sp500_total_return_pct` and `tnote10_total_return_pct`: nominal total returns on the
  S&P 500 and the ten-year U.S. Treasury note, from Damodaran, A., *Historical Returns on
  Stocks, Bonds and Bills*, Stern School of Business, New York University, updated
  January 5, 2026,
  https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histretSP.html
- `cpi_inflation_pct`: annual-average U.S. CPI inflation, U.S. Bureau of Labor Statistics.

The same series are written into `code/historical.py`, which the scripts use directly.
A portfolio's nominal return is the fixed-weight average of the two asset returns,
rebalanced annually, and its real return is (1 + nominal)/(1 + inflation) − 1.

All other results use simulated returns from the parameters stated in the paper.

## Citation

Ko, L. (2026). A Mathematical Framework for Sequence Risk. Working paper.

## License

Code: MIT (see `LICENSE`). The data are compiled from the public sources above and are
redistributed for replication with attribution.
