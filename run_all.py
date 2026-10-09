# Runs every script in paper order and saves the printouts and figures to output/.
# Pass script names to run only those, e.g. python run_all.py goal_grid fig_target
import os, shutil, subprocess, sys, glob

scripts = ["verify_split", "magnitude", "fig_phase", "fig_order_contour",
           "realistic_grid", "student_t", "fig_floor", "goal_grid", "fig_target",
           "historical", "fig_historical", "risk_timing", "accum_timing",
           "variance_ratio", "dependence_grid", "is_stability", "dependence_realdata"]

here = os.path.dirname(os.path.abspath(__file__))
code, out = os.path.join(here, "code"), os.path.join(here, "output")
os.makedirs(out, exist_ok=True)

for name in sys.argv[1:] or scripts:
    print("running", name, flush=True)
    res = subprocess.run([sys.executable, name + ".py"], cwd=code, capture_output=True, text=True)
    with open(os.path.join(out, name + ".txt"), "w") as fh:
        fh.write(res.stdout + res.stderr)
    for pdf in glob.glob(os.path.join(code, "*.pdf")):
        shutil.move(pdf, os.path.join(out, os.path.basename(pdf)))
    if res.returncode != 0:
        print("  failed, see output/" + name + ".txt")
