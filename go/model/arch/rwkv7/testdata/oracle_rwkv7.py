#!/usr/bin/env python3
# SPDX-Licence-Identifier: EUPL-1.2
"""oracle_rwkv7.py -- the RWKV-7 "Goose" block-math oracle for the #42 gate.

A from-scratch, dependency-free (stdlib + numpy only -- no torch, no fla, no triton) transcription of the
RWKV-7 time-mix + channel-mix chain, equation-for-equation from the upstream reference
(github.com/fla-org/flash-linear-attention: fla/layers/rwkv7.py's RWKV7Attention.forward,
fla/models/rwkv7/modeling_rwkv7.py's RWKV7Block/RWKV7FeedForward/RWKV7Model, fla/ops/rwkv7/
fused_recurrent.py's kernel) -- read directly, not guessed. It loads a REAL checkpoint's safetensors
(a hand-rolled header/data reader -- no `safetensors` package dependency) and REAL config.json, runs the
block math over a short fixed prompt, and writes a small fixture (oracle_fixture.json) the Go test
(real_checkpoint_test.go) reads and compares its own host-Go forward against, via cosine similarity.

This script parses no untrusted/attacker-controlled input: config.json and *.safetensors are the
checkpoint files this port's Go loader (loader.go) also reads, and the file is run BY HAND, once, by a
developer who already chose to download and trust that checkpoint -- not invoked by any Go code, any
runtime path, or on any request path. It uses Python's stdlib `json` for config.json (a JSON object, not
Python source) and raw struct/byte parsing for the safetensors header -- no `eval`/`exec`/pickle anywhere.

Usage:
    python3 oracle_rwkv7.py <checkpoint-dir> [output.json]

Checkpoint: RWKV/RWKV7-Goose-World2.8-0.1B-HF (HF format, model_type "rwkv7", 12 layers, hidden_size 768).
"""
import json
import math
import struct
import sys

import numpy as np


# ---------------------------------------------------------------------------
# safetensors: a minimal header + tensor reader (no `safetensors` package dependency).
# ---------------------------------------------------------------------------

class SafeTensors:
    def __init__(self, path):
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        header = json.loads(self.f.read(n))
        self.base = 8 + n
        self.meta = header.pop("__metadata__", None)
        self.header = header

    def get(self, name):
        entry = self.header[name]
        dtype, shape, (start, end) = entry["dtype"], entry["shape"], entry["data_offsets"]
        self.f.seek(self.base + start)
        raw = self.f.read(end - start)
        if dtype == "BF16":
            u16 = np.frombuffer(raw, dtype="<u2")
            u32 = u16.astype(np.uint32) << 16
            arr = u32.view(np.float32).astype(np.float64)
        elif dtype == "F32":
            arr = np.frombuffer(raw, dtype="<f4").astype(np.float64)
        else:
            raise ValueError(f"unsupported dtype {dtype} for {name}")
        return arr.reshape(shape) if shape else arr

    def has(self, name):
        return name in self.header


# ---------------------------------------------------------------------------
# The block math -- float64 throughout (a higher-precision oracle than the f32 host port it gates).
# ---------------------------------------------------------------------------

NEG_EXP_HALF = -0.6065306597126334  # -exp(-0.5), fla's literal decay scale


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sqrelu(x):
    r = np.maximum(x, 0.0)
    return r * r


def token_shift_with_prior(x, prior):
    """delta[t] = prev(t) - x[t]; prev(0)=prior (zeros if None), prev(t>0)=x[t-1]. Returns (delta, newPrior)."""
    shifted = np.zeros_like(x)
    if prior is not None:
        shifted[0] = prior
    shifted[1:] = x[:-1]
    return shifted - x, x[-1].copy()


def addcmul(x, delta, mix):
    return x + delta * mix


def lora_forward(x, A, B, bias, act):
    """hidden = act(x @ A.T); out = hidden @ B.T (+ bias). A:[Low,In] B:[Out,Low] bias:[Out] or None."""
    hidden = x @ A.T
    if act is not None:
        hidden = act(hidden)
    out = hidden @ B.T
    if bias is not None:
        out = out + bias
    return out


def layernorm(x, w, b, eps):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    y = (x - mean) / np.sqrt(var + eps) * w
    return y + b if b is not None else y


def groupnorm_heads(x, w, b, H, V, eps):
    """x:[T,H*V] normalised per (row,head) over its own V channels, then per-full-channel affine."""
    T = x.shape[0]
    xr = x.reshape(T, H, V)
    mean = xr.mean(axis=-1, keepdims=True)
    var = ((xr - mean) ** 2).mean(axis=-1, keepdims=True)
    y = ((xr - mean) / np.sqrt(var + eps)).reshape(T, H * V) * w
    return y + b if b is not None else y


def l2norm_per_head(x, H, K, eps=1e-12):
    """PyTorch F.normalize(dim=-1,p=2) semantics: x / max(||x||_2, eps), per head."""
    T = x.shape[0]
    xr = x.reshape(T, H, K)
    norm = np.sqrt((xr * xr).sum(axis=-1, keepdims=True))
    denom = np.maximum(norm, eps)
    return (xr / denom).reshape(T, H * K)


def wkv7(r, w, k, v, a, b, H, K, V, prior):
    """The generalised delta-rule recurrence -- fla's fused_recurrent_rwkv7_fwd_kernel, verbatim:
    Sa = a . S_old; S = exp(w)*S_old + b(x)Sa + k(x)v; o = S_new . r. r/w/k/a:[T,H,K] v:[T,H,V]."""
    T = r.shape[0]
    state = np.zeros((H, K, V)) if prior is None else prior.copy()
    o = np.zeros((T, H, V))
    for t in range(T):
        for h in range(H):
            s_old = state[h]
            sa = a[t, h] @ s_old
            s_new = np.exp(w[t, h])[:, None] * s_old + np.outer(b[t, h], sa) + np.outer(k[t, h], v[t, h])
            state[h] = s_new
            o[t, h] = r[t, h] @ s_new
    return o, state


def time_mix(x, lw, layer_idx, v_first, prior_state, prior_shift, H, K, V, norm_eps):
    D, Dv = x.shape[1], H * V
    delta, new_shift = token_shift_with_prior(x, prior_shift)
    xr = addcmul(x, delta, lw["x_r"])
    xw = addcmul(x, delta, lw["x_w"])
    xk = addcmul(x, delta, lw["x_k"])
    xv = addcmul(x, delta, lw["x_v"])
    xa = addcmul(x, delta, lw["x_a"])
    xg = addcmul(x, delta, lw["x_g"])

    r = xr @ lw["r_proj"].T
    w_raw = lora_forward(xw, lw["w_lora_A"], lw["w_lora_B"], lw["w_lora_bias"], np.tanh)
    wdecay = NEG_EXP_HALF * sigmoid(w_raw)
    k_raw = xk @ lw["k_proj"].T
    v_raw = xv @ lw["v_proj"].T

    if layer_idx == 0:
        v, v_first_out = v_raw, v_raw.copy()
    else:
        vgate = sigmoid(lora_forward(xv, lw["v_lora_A"], lw["v_lora_B"], lw["v_lora_bias"], None))
        v = v_raw + vgate * (v_first - v_raw)
        v_first_out = v_first

    a_raw = lora_forward(xa, lw["a_lora_A"], lw["a_lora_B"], lw["a_lora_bias"], None)
    a = sigmoid(a_raw)
    g = lora_forward(xg, lw["g_lora_A"], lw["g_lora_B"], None, sigmoid)

    kk = l2norm_per_head(k_raw * lw["k_k"], H, K)
    kupd = k_raw * (1 + (a - 1) * lw["k_a"])

    T = x.shape[0]
    r_h, w_h, k_h = r.reshape(T, H, K), wdecay.reshape(T, H, K), kupd.reshape(T, H, K)
    a_h, b_h, v_h = (-kk).reshape(T, H, K), (kk * a).reshape(T, H, K), v.reshape(T, H, V)
    o, new_state = wkv7(r_h, w_h, k_h, v_h, a_h, b_h, H, K, V, prior_state)
    o = o.reshape(T, Dv)

    onorm = groupnorm_heads(o, lw["g_norm_w"], lw.get("g_norm_b"), H, V, K * norm_eps).reshape(T, H, V)
    corr = (r_h * k_h * lw["r_k"]).sum(axis=-1)  # [T,H]
    gated = ((onorm + corr[:, :, None] * v_h) * g.reshape(T, H, V)).reshape(T, Dv)
    out = gated @ lw["o_proj"].T
    return out, v_first_out, new_state, new_shift


def channel_mix(x, lw, prior_shift):
    delta, new_shift = token_shift_with_prior(x, prior_shift)
    xk = addcmul(x, delta, lw["ffn_x_k"])
    hidden = sqrelu(xk @ lw["ffn_key"].T)
    out = hidden @ lw["ffn_value"].T
    return out, new_shift


def layer_forward(x, lw, layer_idx, v_first, state, norm_eps):
    base0 = x
    if "pre_norm_w" in lw:
        base0 = layernorm(x, lw["pre_norm_w"], lw.get("pre_norm_b"), norm_eps)
    h1 = layernorm(base0, lw["attn_norm_w"], lw.get("attn_norm_b"), norm_eps)
    attn_out, v_first_out, new_wkv, new_shift1 = time_mix(
        h1, lw, layer_idx, v_first, state["wkv"], state["shift1"], lw["H"], lw["K"], lw["V"], norm_eps,
    )
    x1 = base0 + attn_out
    h2 = layernorm(x1, lw["ffn_norm_w"], lw.get("ffn_norm_b"), norm_eps)
    ffn_out, new_shift2 = channel_mix(h2, lw, state["shift2"])
    x2 = x1 + ffn_out
    return x2, v_first_out, {"wkv": new_wkv, "shift1": new_shift1, "shift2": new_shift2}


# ---------------------------------------------------------------------------
# Checkpoint loading.
# ---------------------------------------------------------------------------

def load_layer(st, li):
    p = f"model.layers.{li}."
    r_k = st.get(p + "attn.r_k")
    H, K = r_k.shape
    v_proj = st.get(p + "attn.v_proj.weight")
    Dv = v_proj.shape[0]
    V = Dv // H
    lw = {
        "H": H, "K": K, "V": V,
        "r_proj": st.get(p + "attn.r_proj.weight"),
        "k_proj": st.get(p + "attn.k_proj.weight"),
        "v_proj": v_proj,
        "o_proj": st.get(p + "attn.o_proj.weight"),
        "x_r": st.get(p + "attn.x_r").reshape(-1),
        "x_w": st.get(p + "attn.x_w").reshape(-1),
        "x_k": st.get(p + "attn.x_k").reshape(-1),
        "x_v": st.get(p + "attn.x_v").reshape(-1),
        "x_a": st.get(p + "attn.x_a").reshape(-1),
        "x_g": st.get(p + "attn.x_g").reshape(-1),
        "k_k": st.get(p + "attn.k_k"),
        "k_a": st.get(p + "attn.k_a"),
        "r_k": r_k,
        "g_norm_w": st.get(p + "attn.g_norm.weight"),
        "w_lora_A": st.get(p + "attn.w_lora.lora.0.weight"),
        "w_lora_B": st.get(p + "attn.w_lora.lora.2.weight"),
        "w_lora_bias": st.get(p + "attn.w_lora.lora.2.bias"),
        "a_lora_A": st.get(p + "attn.a_lora.lora.0.weight"),
        "a_lora_B": st.get(p + "attn.a_lora.lora.2.weight"),
        "a_lora_bias": st.get(p + "attn.a_lora.lora.2.bias"),
        "g_lora_A": st.get(p + "attn.g_lora.lora.0.weight"),
        "g_lora_B": st.get(p + "attn.g_lora.lora.2.weight"),
        "attn_norm_w": st.get(p + "attn_norm.weight"),
        "ffn_norm_w": st.get(p + "ffn_norm.weight"),
        "ffn_x_k": st.get(p + "ffn.x_k"),
        "ffn_key": st.get(p + "ffn.key.weight"),
        "ffn_value": st.get(p + "ffn.value.weight"),
    }
    if st.has(p + "attn.g_norm.bias"):
        lw["g_norm_b"] = st.get(p + "attn.g_norm.bias")
    if st.has(p + "attn_norm.bias"):
        lw["attn_norm_b"] = st.get(p + "attn_norm.bias")
    if st.has(p + "ffn_norm.bias"):
        lw["ffn_norm_b"] = st.get(p + "ffn_norm.bias")
    if li == 0:
        lw["pre_norm_w"] = st.get(p + "pre_norm.weight")
        if st.has(p + "pre_norm.bias"):
            lw["pre_norm_b"] = st.get(p + "pre_norm.bias")
    if li > 0:
        lw["v_lora_A"] = st.get(p + "attn.v_lora.lora.0.weight")
        lw["v_lora_B"] = st.get(p + "attn.v_lora.lora.2.weight")
        lw["v_lora_bias"] = st.get(p + "attn.v_lora.lora.2.bias")
    return lw


def main():
    ckpt_dir = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "oracle_fixture.json"

    with open(ckpt_dir + "/config.json") as f:
        cfg = json.load(f)
    norm_eps = cfg.get("norm_eps", 1e-5)
    num_layers = cfg["num_hidden_layers"]

    st = SafeTensors(ckpt_dir + "/model.safetensors")
    embed = st.get("model.embeddings.weight")
    norm_w = st.get("model.norm.weight")
    norm_b = st.get("model.norm.bias") if st.has("model.norm.bias") else None
    lm_head = st.get("lm_head.weight") if st.has("lm_head.weight") else embed

    layers = [load_layer(st, li) for li in range(num_layers)]

    # A short, fixed, arbitrary-but-in-range prompt -- the oracle validates the BLOCK MATH on real
    # weights, not tokenisation, so plain small ids (well within vocab_size) are sufficient; the
    # Go-side coherence check (real_checkpoint_test.go) is the one that needs real English tokenisation.
    prompt_ids = [500, 1000, 2000, 42, 7]
    capture_layers = sorted({0, num_layers // 3, (2 * num_layers) // 3, num_layers - 1})  # >=4 spread across depth

    x = embed[prompt_ids]
    v_first = None
    state = [{"wkv": None, "shift1": None, "shift2": None} for _ in range(num_layers)]
    captured = {}
    for li in range(num_layers):
        x, v_first, state[li] = layer_forward(x, layers[li], li, v_first, state[li], norm_eps)
        if li in capture_layers:
            captured[li] = x[-1].copy()  # last-token hidden after this layer

    final = layernorm(x, norm_w, norm_b, norm_eps)
    logits = final[-1] @ lm_head.T

    fixture = {
        "prompt_ids": prompt_ids,
        "capture_layers": capture_layers,
        "layer_hidden": {str(li): captured[li].tolist() for li in capture_layers},
        "logits": logits.tolist(),
        "top5_ids": [int(i) for i in np.argsort(-logits)[:5]],
        "top5_vals": [float(logits[i]) for i in np.argsort(-logits)[:5]],
    }
    with open(out_path, "w") as f:
        json.dump(fixture, f)
    print(f"wrote {out_path}: captured layers {capture_layers}, top5 ids {fixture['top5_ids']}")


if __name__ == "__main__":
    main()
