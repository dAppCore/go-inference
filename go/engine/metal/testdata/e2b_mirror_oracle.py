# SPDX-Licence-Identifier: EUPL-1.2
#
# e2b_mirror_oracle.py — the reference-dump generator for
# TestRealChainE2BMirrorVsReference_Good (train_real_globals_probe_test.go).
#
# A straight numpy/f32 CPU port of the mlx-lm gemma4_text reference semantics
# (mlx_lm/models/gemma4_text.py + rope_utils.ProportionalRoPE): embed·√hidden,
# per-layer inputs ((normed 1/√hidden-scaled projection + √pliDim per-layer
# embedding)·2^-0.5), per-layer-class attention geometry (sliding hd-256 full
# rotary theta 1e4; global hd-512 proportional partial-0.25 theta 1e6, the
# Inf-padded spectrum paired (j, j+headDim/2) over the whole head), QK-norm,
# no-scale value norm, SDPA scale 1.0, sandwich norms, the KV-shared tail
# (consumers attend the last owner of their class), PLE gate, layer scalar.
#
# Usage:  python3 e2b_mirror_oracle.py <E2B_BF16_SNAPSHOT_DIR> <DUMP_DIR>
# then:   e2b_mlx_reference.py into the SAME dump dir (the bf16 reference the #49
#         discriminator hard-gates against), and:
#         E2B_BF16_DIR=<snapshot> E2B_MIRROR_ORACLE_DIR=<DUMP_DIR> \
#         MLX_METALLIB_PATH=<lib> go test -run TestRealChainE2BMirrorVsReference_Good ...
# Dumps f32 little-endian: embeds_scaled.f32 [T,H], pli.f32 [T,NL,PLID],
# layer_out.f32 [NL,T,H], attn_res.f32 [NL,T,H] (residual after attention),
# mlp_res.f32 [NL,T,H] (residual after the MLP, before the PLE gate — the #49
# sublayer probe's station between attn_res and layer_out).
import json, struct, sys, glob, os
import numpy as np

SNAP = sys.argv[1]
OUT = sys.argv[2]
IDS = [1204, 2381, 977, 4102, 355, 2048, 613, 1777]

def load_tensors(d):
    out = {}
    for fn in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        with open(fn, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
            base = 8 + n
            for k, v in hdr.items():
                if k == "__metadata__":
                    continue
                if not k.startswith("language_model."):
                    continue
                assert v["dtype"] == "BF16", (k, v["dtype"])
                s, e = v["data_offsets"]
                f.seek(base + s)
                raw = np.frombuffer(f.read(e - s), dtype=np.uint16)
                arr = (raw.astype(np.uint32) << 16).view(np.float32).reshape(v["shape"]).copy()
                out[k[len("language_model."):]] = arr
    return out

cfg = json.load(open(os.path.join(SNAP, "config.json")))["text_config"]
T = len(IDS)
H = cfg["hidden_size"]              # 1536
NL = cfg["num_hidden_layers"]       # 35
NH = cfg["num_attention_heads"]     # 8
NKV = cfg["num_key_value_heads"]    # 1
HD = cfg["head_dim"]                # 256
GHD = cfg["global_head_dim"]        # 512
PLID = cfg["hidden_size_per_layer_input"]  # 256
EPS = cfg["rms_norm_eps"]
NSHARED = cfg["num_kv_shared_layers"]      # 20
LTYPES = cfg["layer_types"]
RP = cfg["rope_parameters"]
SW = cfg["sliding_window"]

W = load_tensors(SNAP)
print("tensors:", len(W))

def rms(x, w, eps=EPS):
    # plain w * x_hat over the last axis (mlx nn.RMSNorm / engine rmsNormForwardF32)
    inv = 1.0 / np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True) + eps)
    y = (x.astype(np.float64) * inv).astype(np.float32)
    return y * w if w is not None else y

def gelu_tanh(x):
    x64 = x.astype(np.float64)
    return (0.5 * x64 * (1.0 + np.tanh(0.7978845608028654 * (x64 + 0.044715 * x64 ** 3)))).astype(np.float32)

def rope_default(x, positions, dims, base):
    # mlx nn.RoPE non-traditional full rotary over `dims`: pairs (i, i+dims/2),
    # inv_freq[i] = base ** (-2i/dims). x: [T, nH, dims]
    half = dims // 2
    i = np.arange(half, dtype=np.float64)
    inv = base ** (-2.0 * i / dims)
    ang = positions[:, None].astype(np.float64) * inv[None, :]        # [T, half]
    c, s = np.cos(ang), np.sin(ang)
    x1 = x[..., :half].astype(np.float64)
    x2 = x[..., half:].astype(np.float64)
    out = np.empty_like(x, dtype=np.float64)
    out[..., :half] = x1 * c[:, None, :] - x2 * s[:, None, :]
    out[..., half:] = x1 * s[:, None, :] + x2 * c[:, None, :]
    return out.astype(np.float32)

def rope_proportional(x, positions, dims, rotated, base):
    # mlx-lm ProportionalRoPE: periods = base**(2i/dims) for i < rotated/2, Inf beyond
    # (inv_freq 0 -> identity); pairing (i, i+dims/2) over the WHOLE head.
    half = dims // 2
    inv = np.zeros(half, dtype=np.float64)
    for i in range(rotated // 2):
        inv[i] = base ** (-2.0 * i / dims)   # 1/period
    ang = positions[:, None].astype(np.float64) * inv[None, :]
    c, s = np.cos(ang), np.sin(ang)
    x1 = x[..., :half].astype(np.float64)
    x2 = x[..., half:].astype(np.float64)
    out = np.empty_like(x, dtype=np.float64)
    out[..., :half] = x1 * c[:, None, :] - x2 * s[:, None, :]
    out[..., half:] = x1 * s[:, None, :] + x2 * c[:, None, :]
    return out.astype(np.float32)

# --- embeds + per-layer inputs (the model preamble) ---
emb = W["model.embed_tokens.weight"][IDS]                      # [T,H] f32
h = emb * np.float32(np.sqrt(H))                               # embed scale
embeds_scaled = h.copy()

raw_pl = W["model.embed_tokens_per_layer.weight"][IDS] * np.float32(np.sqrt(PLID))  # [T, NL*PLID]
raw_pl = raw_pl.reshape(T, NL, PLID)
proj = h @ W["model.per_layer_model_projection.weight"].T      # [T, NL*PLID]
proj = proj * np.float32(1.0 / np.sqrt(H))
proj = proj.reshape(T, NL, PLID)
proj = rms(proj, W["model.per_layer_projection_norm.weight"])
pli = ((proj + raw_pl) * np.float32(2 ** -0.5)).astype(np.float32)  # [T, NL, PLID]

positions = np.arange(T)

# shared-KV owner map (mlx previous_kvs)
M = NL - NSHARED
prev = list(range(NL))
by_type = {}
for i in range(M):
    by_type[LTYPES[i]] = i
for j in range(M, NL):
    prev[j] = by_type[LTYPES[j]]

layer_out = np.zeros((NL, T, H), dtype=np.float32)
attn_res = np.zeros((NL, T, H), dtype=np.float32)
mlp_res = np.zeros((NL, T, H), dtype=np.float32)
kv_bank = {}

for li in range(NL):
    P = f"model.layers.{li}."
    is_sliding = LTYPES[li] == "sliding_attention"
    hd = HD if is_sliding else GHD
    nkv = NKV
    has_kv = li < M

    residual = h
    x = rms(h, W[P + "input_layernorm.weight"])                 # pre-attn norm

    q = (x @ W[P + "self_attn.q_proj.weight"].T).reshape(T, NH, hd)
    q = rms(q, W[P + "self_attn.q_norm.weight"])
    if has_kv:
        k = (x @ W[P + "self_attn.k_proj.weight"].T).reshape(T, nkv, hd)
        v = (x @ W[P + "self_attn.v_proj.weight"].T).reshape(T, nkv, hd)
        k = rms(k, W[P + "self_attn.k_norm.weight"])
        v = rms(v, None)                                        # no-scale value norm
        if is_sliding:
            k = rope_default(k, positions, hd, RP["sliding_attention"]["rope_theta"])
        else:
            k = rope_proportional(k, positions, hd, int(hd * RP["full_attention"]["partial_rotary_factor"]), RP["full_attention"]["rope_theta"])
        kv_bank[li] = (k, v)
    else:
        k, v = kv_bank[prev[li]]

    if is_sliding:
        q = rope_default(q, positions, hd, RP["sliding_attention"]["rope_theta"])
    else:
        q = rope_proportional(q, positions, hd, int(hd * RP["full_attention"]["partial_rotary_factor"]), RP["full_attention"]["rope_theta"])

    # SDPA, scale 1.0, causal (+ sliding window), GQA broadcast kv head
    o = np.zeros((T, NH, hd), dtype=np.float64)
    gqa = NH // nkv
    for hh in range(NH):
        hk = hh // gqa
        scores = q[:, hh, :].astype(np.float64) @ k[:, hk, :].astype(np.float64).T  # [T,T]
        mask = np.full((T, T), -np.inf)
        for i in range(T):
            lo = 0 if not is_sliding else max(0, i - SW + 1)
            mask[i, lo:i + 1] = 0.0
        scores = scores + mask
        p = np.exp(scores - scores.max(axis=-1, keepdims=True))
        p = p / p.sum(axis=-1, keepdims=True)
        o[:, hh, :] = p @ v[:, hk, :].astype(np.float64)
    o = o.reshape(T, NH * hd).astype(np.float32)
    attn = o @ W[P + "self_attn.o_proj.weight"].T
    attn = rms(attn, W[P + "post_attention_layernorm.weight"])  # sandwich norm on the branch
    h = residual + attn
    attn_res[li] = h

    residual = h
    x = rms(h, W[P + "pre_feedforward_layernorm.weight"])
    gate = x @ W[P + "mlp.gate_proj.weight"].T
    up = x @ W[P + "mlp.up_proj.weight"].T
    dn = (gelu_tanh(gate) * up) @ W[P + "mlp.down_proj.weight"].T
    dn = rms(dn, W[P + "post_feedforward_layernorm.weight"])
    h = residual + dn
    mlp_res[li] = h

    # PLE gate
    residual = h
    g = h @ W[P + "per_layer_input_gate.weight"].T              # [T, PLID]
    g = gelu_tanh(g) * pli[:, li, :]
    g = g @ W[P + "per_layer_projection.weight"].T              # [T, H]
    g = rms(g, W[P + "post_per_layer_input_norm.weight"])
    h = residual + g

    h = h * W[P + "layer_scalar"][0]
    layer_out[li] = h
    print(f"layer {li:2d} {'S' if is_sliding else 'G'}{'' if has_kv else ' shared->' + str(prev[li])}: |h| max {np.abs(h).max():.3f}")

os.makedirs(OUT, exist_ok=True)
embeds_scaled.tofile(os.path.join(OUT, "embeds_scaled.f32"))
pli.astype(np.float32).tofile(os.path.join(OUT, "pli.f32"))
layer_out.tofile(os.path.join(OUT, "layer_out.f32"))
attn_res.tofile(os.path.join(OUT, "attn_res.f32"))
mlp_res.tofile(os.path.join(OUT, "mlp_res.f32"))
json.dump({"ids": IDS, "T": T, "H": H, "NL": NL, "PLID": PLID}, open(os.path.join(OUT, "manifest.json"), "w"))
print("dumped to", OUT)
