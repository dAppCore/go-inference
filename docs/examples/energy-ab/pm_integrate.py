#!/usr/bin/env python3
"""Integrate powermetrics Combined Power over the marked experiment windows."""
import os
import re
from datetime import datetime

# Must match energy_turns.py's STATE_DIR and the `-o` path given to the
# manually-run `powermetrics` (see README) — NOT under /tmp, which is
# world-writable and subject to symlink/race attacks on a predictable path
# (sonar python:S5443).
STATE_DIR = os.path.join(os.path.expanduser("~"), "Lethean", "lem", "bench", "energy-ab")
PM = os.path.join(STATE_DIR, "pm.txt")
MARKERS = os.path.join(STATE_DIR, "pm_markers.txt")

samples = []  # (epoch_ts, watts, dt_seconds)
ts = None
elapsed = 0.5
for line in open(PM, errors="replace"):
    m = re.match(r"\*\*\* Sampled system activity \((.+?)\) \(([\d.]+)ms elapsed\)", line)
    if m:
        ts = datetime.strptime(m.group(1), "%a %b %d %H:%M:%S %Y %z").timestamp()
        elapsed = float(m.group(2)) / 1000.0
        continue
    m = re.match(r"Combined Power \(CPU \+ GPU \+ ANE\): (\d+) mW", line)
    if m and ts is not None:
        samples.append((ts, int(m.group(1)) / 1000.0, elapsed))

marks = {}
turn_marks = []
for line in open(MARKERS):
    t, label = line.split()
    t = float(t)
    marks[label] = t
    if label.startswith("TURN_"):
        turn_marks.append((label, t))


def integrate(t0, t1):
    e = 0.0
    dur = 0.0
    for ts_, w, dt in samples:
        if t0 <= ts_ <= t1:
            e += w * dt
            dur += dt
    return e, dur


idle_e, idle_d = integrate(marks["IDLE_START"], marks["IDLE_END"])
idle_w = idle_e / idle_d if idle_d else 0.0
print(f"idle baseline: {idle_w:.2f} W over {idle_d:.0f}s")

for arm in ("REPLAY", "STATE"):
    e, d = integrate(marks[f"ARM_{arm}_START"], marks[f"ARM_{arm}_END"])
    net = e - idle_w * d
    print(f"\narm {arm}: {d:.1f}s wall, {e:.1f} J gross, {net:.1f} J net ({net/10:.2f} J/turn net)")
    print(f"  per-turn net J: ", end="")
    for i in range(1, 11):
        te, td = integrate(marks[f"TURN_{arm}_{i}_START"], marks[f"TURN_{arm}_{i}_END"])
        tnet = te - idle_w * td
        print(f"{tnet:.1f}", end=" ")
    print()

er, dr = integrate(marks["ARM_REPLAY_START"], marks["ARM_REPLAY_END"])
es, ds = integrate(marks["ARM_STATE_START"], marks["ARM_STATE_END"])
nr, ns = er - idle_w * dr, es - idle_w * ds
print(f"\nDELTA (replay - state): {nr-ns:.1f} J net over 10 turns = {(nr-ns)/10:.2f} J/turn saved")
print(f"state uses {ns/nr*100:.0f}% of replay's net energy for identical output")
