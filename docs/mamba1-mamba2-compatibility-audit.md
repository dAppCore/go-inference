# Mamba1 / Mamba2 mixer compatibility audit

## Verdict

Mamba1 and Falcon-Mamba do **not** fit the existing `model/mamba2` mixer API.
They share a causal convolution and recurrent SSM at a high level, but the
checkpoint tensors and the state transition require different ranks and a
different projection pipeline. Treating `falcon_mamba` as a configuration
alias, reshaping its weights into `mamba2.BlockWeights`, or selecting special
Mamba2 head geometry would change the model rather than load it faithfully.

This is therefore an evidenced stop for roadmap rank 16. A later implementation
needs a Mamba1 selective-scan primitive and block, not conditionals in the
Mamba2 SSD primitive.

## Sources and scope

The reference is Hugging Face Transformers commit
[`63f32a8`](https://github.com/huggingface/transformers/blob/63f32a8782cb70da3365acab16f2b67947737985/src/transformers/models/mamba/modeling_mamba.py#L59-L105).
The real checkpoint evidence is Falcon-Mamba revision
[`080ad94`](https://huggingface.co/tiiuae/falcon-mamba-7b/tree/080ad94b3619e2c2d0afa59bafdc6113465b7006):
its
[`config.json`](https://huggingface.co/tiiuae/falcon-mamba-7b/blob/080ad94b3619e2c2d0afa59bafdc6113465b7006/config.json)
declares `model_type: falcon_mamba`, intermediate size 8192, state size 16,
time-step rank 256, and convolution kernel 4; its
[`model.safetensors.index.json`](https://huggingface.co/tiiuae/falcon-mamba-7b/blob/080ad94b3619e2c2d0afa59bafdc6113465b7006/model.safetensors.index.json)
lists separate `in_proj`, `x_proj`, `dt_proj`, `A_log`, `D`, `conv1d`, and
`out_proj` tensors for every mixer layer. No model packs were downloaded.

## Blocking differences

### 1. The input and selection projections have incompatible contracts

Mamba1 projects the residual stream to `2 * intermediate_size`, splits that
into `hidden_states | gate`, convolves only `hidden_states`, then applies a
separate `x_proj` to produce `time_step_rank | B | C` and a separate `dt_proj`
to expand the low-rank time step to every intermediate channel
([reference lines 277-321](https://github.com/huggingface/transformers/blob/63f32a8782cb70da3365acab16f2b67947737985/src/transformers/models/mamba/modeling_mamba.py#L277-L321)).

The existing Mamba2 block instead requires one fused input projection with
output width `2*dInner + 2*NumGroups*StateDim + NumHeads`
(`go/model/mamba2/block.go:20-42`). It splits that projection directly into
`z | xBC | dt`, and the convolution consumes `x`, `B`, and `C` together
(`go/model/mamba2/block.go:172-210`). `BlockWeights` has no `XProj` or
`DtProj` matrix (`go/model/mamba2/block.go:23-31`). Consequently the real
Falcon-Mamba `x_proj.weight` and `dt_proj.weight` have nowhere to go, and the
existing fused `InProj` expects columns which are absent from Mamba1's
`in_proj.weight`.

### 2. The convolution state has a different width

Mamba1 carries and convolves only the intermediate activations; its cache is
`[intermediate_size, conv_kernel]`
([reference lines 283-309](https://github.com/huggingface/transformers/blob/63f32a8782cb70da3365acab16f2b67947737985/src/transformers/models/mamba/modeling_mamba.py#L283-L309)).
Mamba2 defines the convolution width as
`dInner + 2*NumGroups*StateDim` (`go/model/mamba2/block.go:40-42`) because its
convolution also transforms B and C, and it threads a state of exactly that
width (`go/model/mamba2/block.go:187-210`). There is no Mamba2 configuration
which both accepts Mamba1's `conv1d.weight` and still supplies the B/C channels
the Mamba2 split requires.

### 3. The discretisation tensor ranks are incompatible

Mamba1 stores `A_log` as `[intermediate_size, state_size]`, computes a time step
for every intermediate channel, and therefore computes `discrete_A` as
`[intermediate_size, sequence, state_size]`
([reference lines 93-103](https://github.com/huggingface/transformers/blob/63f32a8782cb70da3365acab16f2b67947737985/src/transformers/models/mamba/modeling_mamba.py#L93-L103),
[323-327](https://github.com/huggingface/transformers/blob/63f32a8782cb70da3365acab16f2b67947737985/src/transformers/models/mamba/modeling_mamba.py#L323-L327)).

`SSDScanF32` accepts only scalar decay and step values per head: `A [H]` and
`dt [L,H]`; its single `dA` value is reused across every head-dimension/state
pair (`go/model/mamba2/scan.go:17-28,47-65`). The Mamba2 loader enforces a
rank-one `A_log` and derives `H` from it (`go/model/mamba2/loader.go:98-115`).
Flattening Mamba1 channels into `H` cannot repair this: Mamba2 would then need
`HeadDim=1`, but its B/C and convolution layout would still be the incompatible
Mamba2 layout described above.

### 4. The skip and gate stages are not the same operation

Mamba1's `D` is per intermediate channel, and the reference selective scan
receives the original gate so that it applies the Mamba1 `y * SiLU(gate)`
semantics before `out_proj`
([reference lines 236-269](https://github.com/huggingface/transformers/blob/63f32a8782cb70da3365acab16f2b67947737985/src/transformers/models/mamba/modeling_mamba.py#L236-L269)).
Mamba2's scan accepts `D [H]` and broadcasts it across `HeadDim`
(`go/model/mamba2/scan.go:25-37,62-64`), after which the block performs a
gated RMS normalisation with a separate norm weight
(`go/model/mamba2/block.go:237-264`). Falcon-Mamba has no mixer norm tensor in
its weight map. Omitting that weight does not remove the RMS normalisation, so
the current block would still change every output.

## Required neutral surface before a model lane

A faithful Mamba1 lane needs a separate neutral selective-scan contract with:

- channel-wise `A [dInner,stateDim]`, `dt [L,dInner]`, and `D [dInner]`;
- recurrent state `[dInner,stateDim]` and convolution state
  `[(kernel-1),dInner]`;
- explicit low-rank `x_proj` and `dt_proj` weights;
- convolution of `x` alone, with B/C selected after the convolution; and
- Mamba1's gate-without-mixer-RMSNorm output stage.

Those are new operator semantics and declared architecture parameters, not a
`falcon_mamba` alias. Until that primitive exists and has seeded-synthetic
prefill/decode parity against the pinned reference, rank 16 should remain
blocked rather than reuse `model/mamba2`.
