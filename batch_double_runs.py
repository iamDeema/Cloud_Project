# batch_double_runs.py
import subprocess
import time

POWERLOG = r"C:\Program Files\Intel\Power Gadget 3.6\PowerLog3.0.exe"

START_SAMPLES = 25600    # starting point
MAX_SAMPLES   = 204800  # stop when samples > this
PL_INTERVAL_MS = 500  # PowerLog sampling interval

samples = START_SAMPLES
last_duration = None

while samples <= MAX_SAMPLES:
    print(f"\n=== Running {samples} samples ===")
    t0 = time.time()
    subprocess.run([
        "python", "win_power_infer.py",
        "--samples", str(samples),
        "--pl-interval", str(PL_INTERVAL_MS),
        "--powerlog-exe", POWERLOG
    ], check=True)
    dt = time.time() - t0
    print(f"[DONE] {samples} samples in {dt:.1f}s")

    # Quick ETA guess for next run (doubling ≈ ~2× time)
    if last_duration is not None:
        eta = last_duration * 2
        print(f"[ETA] Next run (~{samples*2} samples) ≈ {eta:.1f}s")
    last_duration = dt

    samples *= 2  # double each run
