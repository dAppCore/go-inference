# SPDX-Licence-Identifier: EUPL-1.2
#
# e2b_mlx_reference.py — the bf16 REFERENCE-dump generator for
# TestForwardCaptureHiddensE2BVsOracle (train_e2b_capture_vs_oracle_test.go).
#
# Why a second reference beside e2b_mirror_oracle.py: the numpy oracle keeps every
# station in f32/f64, but ANY production forward stores the residual stream in bf16
# (gemma giant channels reach |h|≈100+, where one bf16 ULP is 0.5), and E2B's layer-8
# MLP + per-layer-input gates sit on gelu cliffs that chaotically amplify that
# correctly-rounded storage dust ~1000x in 1-cos terms for particular tokens (#49:
# token 6 of the parity ids). mlx-lm's own bf16 forward — the ecosystem reference the
# engine was validated against — diverges from the f64 oracle with the SAME signature
# (first bad layer 8, worst layer 11 at cosine ~0.83, recovery from 12). bf16 forwards
# form a tight equivalence class (engine-vs-mlx ≥ 0.9985 every layer, because their
# station rounding lands on the same bf16 grid), while the f64 trajectory leaves that
# class at the chaos layers. The correct ground-truth bar for a bf16 engine is
# therefore parity with THIS dump (direct cosine) plus the per-layer envelope against
# the oracle — which is exactly what the discriminator hard-gates.
#
# Usage:  uv run --with mlx-lm python e2b_mlx_reference.py <E2B_BF16_SNAPSHOT_DIR> <DUMP_DIR>
#         (DUMP_DIR = the same directory e2b_mirror_oracle.py dumped into)
# Dumps f32 little-endian: mlx_layer_out.f32 [NL,T,H] — mlx-lm's own bf16 per-layer
# output hiddens for the parity ids, captured by hooking DecoderLayer.__call__.
import sys, os
import numpy as np
import mlx.core as mx
from mlx_lm import load

SNAP = sys.argv[1]
OUT = sys.argv[2]
IDS = [1204, 2381, 977, 4102, 355, 2048, 613, 1777]  # the #42 harness's parity ids
T = len(IDS)

model, _tok = load(SNAP)
lm = model.language_model if hasattr(model, "language_model") else model
inner = lm.model if hasattr(lm, "model") else lm
layers = inner.layers
NL = len(layers)
DL = type(layers[0])
print(f"mlx model: {NL} x {DL.__name__}")

captured = []
orig = DL.__call__


def hooked(self, *a, **k):
    out = orig(self, *a, **k)
    h = out[0] if isinstance(out, tuple) else out
    captured.append(np.array(h.astype(mx.float32)))
    return out


DL.__call__ = hooked

out = lm(mx.array([IDS]))
mx.eval(out)
assert len(captured) >= NL, (len(captured), NL)

H = captured[0].reshape(T, -1).shape[1]
stack = np.stack([h.reshape(T, H) for h in captured[:NL]]).astype(np.float32)
os.makedirs(OUT, exist_ok=True)
stack.tofile(os.path.join(OUT, "mlx_layer_out.f32"))
print(f"dumped mlx_layer_out.f32 [{NL},{T},{H}] to {OUT}")
