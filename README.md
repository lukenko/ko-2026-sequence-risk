# ko-2026-sequence-risk

Code and data for "A Mathematical Framework for Sequence Risk" (Luken Ko, 2026).

To reproduce everything:

    pip install -r requirements.txt
    python run_all.py

Each script's printout goes to `output/`, along with the five figures. The copy of `output/` in
the repo is from my own run, so you can compare. A full run takes about an hour, nearly all of
it Table 5 (`dependence_grid.py`) and the stability check after it (`is_stability.py`). To run
one script on its own, use e.g. `python run_all.py goal_grid`, or run it from inside `code/`.
All random draws use fixed seeds.

Scripts, in the order they appear in the paper:

- `verify_split.py`: checks the closed-form split against brute force on short plans
  (Section 5). The other scripts import their closed-form functions from here.
- `magnitude.py`: the variance shares quoted in Section 5.5.
- `fig_phase.py`, `fig_order_contour.py`: Figures 1 and 2.
- `realistic_grid.py`: Table 1.
- `student_t.py`: the t-distributed returns check in Section 6.1.
- `fig_floor.py`: Figure 3.
- `goal_grid.py`, `fig_target.py`: Table 2 and Figure 4.
- `historical.py`, `fig_historical.py`: Section 7, Table 3, and Figure 5.
- `risk_timing.py`, `accum_timing.py`: Table 4 and the accumulation case discussed after it.
- `variance_ratio.py`, `dependence_grid.py`, `is_stability.py`, `dependence_realdata.py`:
  Section 9 and Table 5.
- `figstyle.py`: plot settings shared by the figure scripts.

Data: `data/us_annual_returns_1928_2025.csv` has the annual S&P 500 and 10-year Treasury total
returns from Aswath Damodaran's NYU Stern data page (January 2026 update,
https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histretSP.html) and annual CPI
inflation from the BLS. The same numbers are typed into `historical.py`, which is what the
scripts actually use.

I ran this with Python 3.14, NumPy 2.3.5 and Matplotlib 3.11.0. Code is MIT licensed.
