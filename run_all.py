"""
Regenerate every table, figure, and in-text number of
"A Mathematical Framework for Sequence Risk" (Luken Ko).

Usage:  python run_all.py            # run everything
        python run_all.py fig_phase  # run selected scripts by name

Each script runs from code/ with a fixed random seed. Its printed output is saved to
output/<script>.txt and any figure it draws is moved to output/.
"""
import pathlib, shutil, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parent
CODE, OUT = ROOT / "code", ROOT / "output"

# (script, what it reproduces), in paper order
SCRIPTS = [
    ("verify_split",        "Section 5: closed-form split checked against enumeration"),
    ("magnitude",           "Section 5.5: size of the sequencing share"),
    ("fig_phase",           "Figure 1"),
    ("fig_order_contour",   "Figure 2"),
    ("realistic_grid",      "Table 1"),
    ("student_t",           "Section 6.1: Student-t returns"),
    ("fig_floor",           "Figure 3"),
    ("goal_grid",           "Table 2"),
    ("fig_target",          "Figure 4"),
    ("historical",          "Section 7 and Table 3"),
    ("fig_historical",      "Figure 5"),
    ("risk_timing",         "Table 4"),
    ("accum_timing",        "Section 8: accumulation counterpart of Table 4"),
    ("variance_ratio",      "Section 9: variance ratio"),
    ("dependence_grid",     "Table 5"),
    ("is_stability",        "Section 9: stability of the importance-sampling estimates"),
    ("dependence_realdata", "Section 9: block bootstrap"),
]

def main(selected):
    OUT.mkdir(exist_ok=True)
    todo = [s for s in SCRIPTS if not selected or s[0] in selected]
    for name, what in todo:
        t0 = time.time()
        print(f"[{name}] {what} ...", flush=True)
        res = subprocess.run([sys.executable, f"{name}.py"], cwd=CODE,
                             capture_output=True, text=True)
        (OUT / f"{name}.txt").write_text(res.stdout + res.stderr, encoding="utf-8")
        for pdf in CODE.glob("*.pdf"):
            shutil.move(str(pdf), OUT / pdf.name)
        status = "ok" if res.returncode == 0 else f"FAILED (exit {res.returncode})"
        print(f"    {status} in {time.time() - t0:.0f}s", flush=True)

if __name__ == "__main__":
    main(set(sys.argv[1:]))
