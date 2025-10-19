# win_power_infer.py
import argparse
import itertools
import os
import threading
import time
import subprocess
import shlex
import tempfile
import statistics
import csv
import platform

import pandas as pd
import psutil
from datasets import load_dataset
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from tqdm import tqdm
    TQDM = True
except ImportError:
    TQDM = False
    def tqdm(x, **kwargs):
        return x

ID2LABEL = {0: "NEGATIVE", 1: "POSITIVE"}

# Fixed internal batch size (kept for efficiency, not user-facing)
BATCH_SIZE = 16

POWERLOG_EXE_CANDIDATES = [
    r"C:\Program Files\Intel\Power Gadget\PowerLog3.0.exe",
    r"C:\Program Files\Intel\Power Gadget 3.7\PowerLog3.0.exe",
    r"C:\Program Files\Intel\Power Gadget 3.6\PowerLog3.0.exe",
    r"C:\Program Files (x86)\Intel\Power Gadget\PowerLog3.0.exe",
]

pl_proc = None
pl_csv_path = None

_cpu_t, _cpu_sys, _cpu_proc, _ram_proc_mb, _cpu_temp_c = [], [], [], [], []


def sampler_loop(stop_evt, poll=0.25):
    proc = psutil.Process(os.getpid())
    proc.cpu_percent(None)
    while not stop_evt.is_set():
        _cpu_t.append(time.perf_counter())
        _cpu_sys.append(psutil.cpu_percent(None))
        _cpu_proc.append(proc.cpu_percent(None))  # can exceed 100 on multi-core
        _ram_proc_mb.append(proc.memory_info().rss / (1024 ** 2))
        _cpu_temp_c.append(0.0)  # keep numeric; Windows often has no sensors
        time.sleep(poll)


def load_slice(samples, split):
    ds = load_dataset("mteb/amazon_polarity", split=split)
    ds = ds.select(range(min(samples, len(ds))))
    return ds["text"], [ID2LABEL[int(x)] for x in ds["label"]]


def _find_powerlog_exe(explicit_path=None):
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path
    for p in POWERLOG_EXE_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def start_powerlog(interval_ms=1000, explicit_path=None):
    global pl_proc, pl_csv_path
    exe = _find_powerlog_exe(explicit_path)
    if exe is None:
        raise RuntimeError("PowerLog3.0.exe not found.")
    tf = tempfile.NamedTemporaryFile(delete=False, prefix="powerlog_", suffix=".csv")
    pl_csv_path = tf.name
    tf.close()
    cmd = f'"{exe}" -resolution {int(interval_ms)} -file "{pl_csv_path}"'
    pl_proc = subprocess.Popen(shlex.split(cmd), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return pl_csv_path


def stop_powerlog(timeout=3.0):
    global pl_proc
    if pl_proc is None:
        return
    try:
        pl_proc.terminate()
        try:
            pl_proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pl_proc.kill()
    finally:
        pl_proc = None


def parse_powerlog_csv(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {"samples": 0}
    with open(path, "r", newline="", encoding="utf-8", errors="ignore") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return {"samples": 0}
    header, data = rows[0], rows[1:]
    norm = [h.strip().lower() for h in header]

    def find_col(subs):
        for i, h in enumerate(norm):
            if all(s in h for s in subs):
                return i
        return None

    cpu_col = (find_col(["package", "power"]) or
               find_col(["cpu", "power"]) or
               find_col(["processor", "power"]) or
               find_col(["processor"]))
    dram_col = find_col(["dram", "power"]) or find_col(["memory", "power"])

    cpu_vals, dram_vals = [], []
    for r in data:
        try:
            v = r[cpu_col].replace(",", ".")
            cpu_vals.append(float(v))
            if dram_col is not None and r[dram_col] != "":
                d = r[dram_col].replace(",", ".")
                dram_vals.append(float(d))
        except Exception:
            continue

    if not cpu_vals:
        return {"samples": 0}
    return {"samples": len(cpu_vals), "cpu_series_W": cpu_vals, "dram_series_W": dram_vals}


def estimate_cpu_power(cpu_pct_series, tdp_W, cores):
    denom = max(1.0, cores * 100.0)
    return [tdp_W * max(0.0, min(1.0, p / denom)) for p in cpu_pct_series]


def estimate_ram_power(used_gb, ram_idle_W, ram_w_per_gb):
    return max(0.0, ram_idle_W + ram_w_per_gb * used_gb)


def main(args):
    print(f"[INFO] OS: {platform.system()} {platform.release()}")
    logical_cpus = psutil.cpu_count(logical=True) or 1

    texts, gold = load_slice(args.samples, args.split)
    clf = pipeline("sentiment-analysis",
                   model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                   truncation=True, device=-1)

    stop_evt = threading.Event()
    t = threading.Thread(target=sampler_loop, args=(stop_evt, args.sample_poll_s), daemon=True)
    t.start()

    pl_log = None
    pl_ok = False
    if not args.no_powerlog:
        try:
            pl_log = start_powerlog(interval_ms=args.pl_interval, explicit_path=args.powerlog_exe)
            time.sleep(max(0.25, args.pl_interval / 1000.0 * 2))
            pl_ok = True
        except Exception as e:
            print(f"[WARN] PowerLog failed: {e}")
            pl_ok = False

    # Inference loop with fixed internal batch size
    batches = [texts[i:i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
    preds, times_ = [], []
    t0 = time.perf_counter()
    iterator = tqdm(batches, desc="Infer") if TQDM else batches
    for b in iterator:
        t1 = time.perf_counter()
        out = clf(b)
        t2 = time.perf_counter()
        preds.extend(out)
        times_.append(t2 - t1)
    total_s = time.perf_counter() - t0

    stop_evt.set()
    if pl_log:
        stop_powerlog()
    t.join(timeout=2.0)

    # Power measurements (prefer PowerLog; else estimate)
    hw = {}
    if pl_ok and pl_log:
        parsed = parse_powerlog_csv(pl_log)
        if parsed.get("samples", 0) > 0:
            dt = args.pl_interval / 1000.0
            cpu_W = parsed["cpu_series_W"]
            dram_W = parsed["dram_series_W"]
            hw["cpu_avg_W"] = statistics.mean(cpu_W)
            hw["cpu_energy_J"] = sum(cpu_W) * dt
            if dram_W:
                hw["ram_avg_W"] = statistics.mean(dram_W)
                hw["ram_energy_J"] = sum(dram_W) * dt
            else:
                used = statistics.mean(_ram_proc_mb) / 1024 if _ram_proc_mb else 0.0
                ramW = estimate_ram_power(used, args.ram_idle_w, args.ram_w_per_gb)
                hw["ram_avg_W"], hw["ram_energy_J"] = ramW, ramW * total_s
        else:
            pl_ok = False

    if not pl_ok:
        cpu_series = estimate_cpu_power(_cpu_proc, args.cpu_tdp_w, logical_cpus)
        dt = args.sample_poll_s
        cpu_energy = sum(cpu_series) * dt if cpu_series else 0.0
        cpu_avg = (cpu_energy / (len(cpu_series) * dt)) if cpu_series else 0.0
        used = statistics.mean(_ram_proc_mb) / 1024 if _ram_proc_mb else 0.0
        ramW = estimate_ram_power(used, args.ram_idle_w, args.ram_w_per_gb)
        hw = {
            "cpu_avg_W": cpu_avg,
            "cpu_energy_J": cpu_energy,
            "ram_avg_W": ramW,
            "ram_energy_J": ramW * total_s
        }

    # Metrics
    acc = sum(1 for i, p in enumerate(preds) if p["label"] == gold[i]) / len(gold)
    avg_temp = statistics.mean([t for t in _cpu_temp_c if t > 0]) if any(_cpu_temp_c) else 60.0
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")

    row = {
        "samples": int(args.samples),
        "batch_size": int(BATCH_SIZE),                    # still numeric, for traceability
        "accuracy": round(acc, 4),
        "total_inference_time_s": round(total_s, 2),
        "avg_cpu_temp_c": round(avg_temp, 2),
        "cpu_avg_W": round(hw["cpu_avg_W"], 3),
        "cpu_energy_J": round(hw["cpu_energy_J"], 2),
        "ram_avg_W": round(hw["ram_avg_W"], 3),
        "ram_energy_J": round(hw["ram_energy_J"], 2),
        "total_energy_J": round(hw["cpu_energy_J"] + hw["ram_energy_J"], 2)
    }

    file = "run_summary.csv"
    header = not os.path.exists(file)
    with open(file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if header:
            writer.writeheader()
        writer.writerow(row)
    print(f"[INFO] Saved numeric summary to {file}")

    # Dashboard
    df = pd.read_csv(file)
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 18))
    sns.lineplot(data=df, x="samples", y="total_inference_time_s", marker="o", ax=ax1)
    ax1.set_title("Inference Time (s)")
    ax1.set_xlabel("Number of Samples")
    ax1.set_ylabel("Time (s)")

    sns.barplot(data=df, x="samples", y="total_energy_J", ax=ax2)
    ax2.set_title("Total Energy (J)")
    ax2.set_xlabel("Number of Samples")
    ax2.set_ylabel("Energy (J)")

    sns.lineplot(data=df, x="samples", y="avg_cpu_temp_c", marker="o", ax=ax3)
    ax3.set_title("Average CPU Temp (°C)")
    ax3.set_xlabel("Number of Samples")
    ax3.set_ylabel("Temperature (°C)")

    plt.tight_layout()
    out_png = f"dashboard_{ts}.png"
    plt.savefig(out_png)
    plt.close()
    print(f"[SUCCESS] Dashboard saved → {out_png}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Benchmark NLP performance & energy on Windows (PowerLog or estimates).")
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--no-powerlog", action="store_true")
    ap.add_argument("--powerlog-exe", type=str, default=None)
    ap.add_argument("--pl-interval", type=int, default=500)
    ap.add_argument("--cpu-tdp-w", type=float, default=28.0)
    ap.add_argument("--ram-idle-w", type=float, default=0.8)
    ap.add_argument("--ram-w-per-gb", type=float, default=0.25)
    ap.add_argument("--sample-poll-s", type=float, default=0.25)
    args = ap.parse_args()
    main(args)
