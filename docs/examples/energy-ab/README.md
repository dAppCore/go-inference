# The energy A/B — replay vs `-state` continuity

Measures joules per conversation turn, same binary, same traffic, one flag.

1. In your own terminal (sudo needs a TTY):
   `sudo powermetrics --samplers cpu_power,gpu_power -i 500 -o /tmp/pm.txt`
2. 60s idle window (markers via the driver's clock).
3. Arm A: `lem serve --model <snapshot> -state-conversations=false`,
   then `LEM_PORT=<port> python3 energy_turns.py REPLAY`.
4. Arm B: serve with the default (continuity on), `energy_turns.py STATE`.
5. `python3 pm_integrate.py` — integrates Combined Power over the marked
   windows, subtracts idle, reports J/turn and the per-turn series.

Receipt (2026-07-13, M3 Ultra, gemma-4 E2B 4-bit, 10 turns, ~340 words/turn,
idle 1.02W): replay 1,439 J net (climbing 83→233 J/turn as history grows);
continuity 771 J net (flat ~77 J/turn). Identical output both arms
(3,370 vs 3,364 words). Continuity: 54% of the energy, gap widening
every turn.
