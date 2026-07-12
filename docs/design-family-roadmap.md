# Model-family roadmap

## Decision

Use Hugging Face Transformers as the definition authority, but dispatch ports by
implementation family rather than by every registry alias.  The first independent
lanes after the current MoE work should be **Phi**, **GPT-2/BigCode**, **OPT**,
**Falcon**, **BLOOM**, and **GPT-NeoX**.  They combine substantial public use with
bounded, text-only implementations.  Multimodal towers and genuinely new sequence
mixers remain explicitly behind their prerequisites.

This is a research queue, not an assertion that every old compatibility decoder is
worth shipping.  A lane starts only after choosing a real public checkpoint,
checking its exact Transformers configuration and tokenizer, recording golden
logits, and proving the Hugging Face weight-name map.

## Survey boundary and method

The survey was taken on **12 July 2026** from Transformers
**5.14.0.dev0**, commit
[`63f32a8782cb70da3365acab16f2b67947737985`](https://github.com/huggingface/transformers/tree/63f32a8782cb70da3365acab16f2b67947737985).
The enumerated authority is the 168 keys in
[`MODEL_FOR_CAUSAL_LM_MAPPING_NAMES`](https://github.com/huggingface/transformers/blob/63f32a8782cb70da3365acab16f2b67947737985/src/transformers/models/auto/modeling_auto.py#L665-L839),
cross-checked against the repository's
[model catalogue](https://github.com/huggingface/transformers/blob/63f32a8782cb70da3365acab16f2b67947737985/docs/source/en/models.md).
That is a reproducible meaning of “text-generation architecture”: models exposed
through `AutoModelForCausalLM`.  It includes compatibility decoders from BERT,
encoder-decoder, vision and audio families; it does not silently add
`AutoModelForSeq2SeqLM`, whose product task is text-to-text rather than continuation.
Tower-bearing entries are retained because Transformers deliberately exposes their
language decoder through the same causal-LM contract.

The classes mean:

- **DIRECT** — dense decoder attention/MLP can be expressed with the existing
  neutral tensor, attention, RoPE/position, norm and cache primitives; normally one
  model lane.
- **TEMPLATE-EXISTS** — implementation composes an existing or already-scheduled
  template: sparse MoE after Mixtral/DeepSeek, or recurrence/SSM through the current
  mixers.  Architecture-specific routing or state still needs parity tests.
- **TOWER-WORK** — language decoding is not useful without a vision, audio or other
  modality tower/projector and its processor contract.
- **EXOTIC** — a new neutral primitive is required.  The missing primitive is named
  in the inventory rather than hidden inside a model package.

## What is already covered

The repository already contains `go/model/gemma3`, `gemma4`, `llama`, `mistral`,
`qwen3`, `mamba2`, `rwkv7`, `deltanet`, `composed`, and `bert`.  In registry terms,
cross off:

- ~~`llama`; `mistral`, `ministral`, `ministral3`; `qwen3`; `mamba2`; `bert`~~;
- ~~`gemma3`, `gemma3_text`, `gemma4`, `gemma4_assistant`, `gemma4_text`,
  `gemma4_unified`, `gemma4_unified_assistant`, `gemma4_unified_text`~~, subject to
  each package's documented text/tower boundary; and
- the house-only ~~gated-delta, RWKV7, DeltaNet and composed~~ templates, which are
  capabilities rather than matching upstream registry keys (`rwkv` is RWKV4/5 and
  is therefore not crossed off).

~~`mixtral` and `deepseek_v2`, `deepseek_v3`, `deepseek_v32`, `deepseek_v4`~~ are
in flight on `lane/moe-family`.  Their result is a prerequisite, not an excuse to
duplicate router code in another lane.

## Usage signal

Hub “downloads last month” is a noisy but inspectable ecosystem proxy: downloads
are attached to a repository, fluctuate, and can include CI or derivative use.
These are representative checkpoint counts captured on **12 July 2026**, not sums
over every derivative and not forecasts.  The linked Hub API records are the
machine-readable evidence.

| Representative checkpoint | Family signal | Downloads/month |
|---|---:|---:|
| [`facebook/opt-125m`](https://huggingface.co/api/models/facebook/opt-125m) | OPT | 15,117,357 |
| [`openai-community/gpt2`](https://huggingface.co/api/models/openai-community/gpt2) | GPT-2 | 13,671,177 |
| [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/api/models/Qwen/Qwen2.5-1.5B-Instruct) | Qwen2 (adjacent to shipped Qwen3) | 12,162,068 |
| [`deepseek-ai/DeepSeek-V2-Lite-Chat`](https://huggingface.co/api/models/deepseek-ai/DeepSeek-V2-Lite-Chat) | DeepSeek (in flight) | 1,220,556 |
| [`tiiuae/falcon-7b`](https://huggingface.co/api/models/tiiuae/falcon-7b) | Falcon | 869,705 |
| [`microsoft/phi-2`](https://huggingface.co/api/models/microsoft/phi-2) | Phi | 851,363 |
| [`mistralai/Mixtral-8x7B-Instruct-v0.1`](https://huggingface.co/api/models/mistralai/Mixtral-8x7B-Instruct-v0.1) | Mixtral (in flight) | 697,157 |
| [`microsoft/Phi-3-mini-4k-instruct`](https://huggingface.co/api/models/microsoft/Phi-3-mini-4k-instruct) | Phi-3 | 605,920 |
| [`bigscience/bloom-560m`](https://huggingface.co/api/models/bigscience/bloom-560m) | BLOOM | 586,086 |
| [`EleutherAI/gpt-neox-20b`](https://huggingface.co/api/models/EleutherAI/gpt-neox-20b) | GPT-NeoX | 545,254 |
| [`bigcode/starcoder2-3b`](https://huggingface.co/api/models/bigcode/starcoder2-3b) | StarCoder2 | 180,212 |
| [`ibm-granite/granite-3.3-8b-instruct`](https://huggingface.co/api/models/ibm-granite/granite-3.3-8b-instruct) | Granite | 95,266 |
| [`allenai/OLMo-2-1124-7B-Instruct`](https://huggingface.co/api/models/allenai/OLMo-2-1124-7B-Instruct) | OLMo2 | 57,903 |

Counts favour small base checkpoints, so rank also considers ecosystem presence:
official instruction checkpoints, common quantisations/fine-tunes, inference-engine
support, and whether a port unlocks close relatives.  This is why Phi leads the
implementation queue even though two old, tiny checkpoints have larger automated
traffic.

## Dispatch queue

| Rank | Family | Class | Prerequisite | Estimated lane shape |
|---:|---|---|---|---|
| 1 | Phi / Phi-3 | DIRECT | none | One lane: Phi-2 and Phi-3 fixtures; partial RoPE and long-context scaling split behind config tests. |
| 2 | GPT-2 / GPT-SW3 / BigCode | DIRECT | learned absolute-position helper confirmed reusable | One lane with GPT-2 golden first, then Conv1D weight transpose and BigCode MQA variant. |
| 3 | OPT | DIRECT | none | One compact lane: pre/post-norm switch, learned positions and tied projection cases. |
| 4 | Falcon | DIRECT | ALiBi and multi-query attention parity | One lane covering old/new decoder architecture flags; do not include Falcon-H1/Mamba. |
| 5 | BLOOM | DIRECT | ALiBi | One lane: fused QKV layout, embedding layer norm and multilingual tokenizer fixture. |
| 6 | GPT-NeoX / GPT-J / GPT-Neo | DIRECT | rotary-dimension and parallel-residual hooks | One lane if hooks stay neutral; three golden checkpoints and separate weight maps. |
| 7 | Qwen2 | DIRECT | reuse Qwen3 only after a config/weight-map diff | One short compatibility lane; include Qwen2-MoE only after the MoE template. |
| 8 | StarCoder2 / CodeGen | DIRECT | sliding-window/cache parity | One code-model lane; tokenizer files remain fixtures, not model logic. |
| 9 | Granite dense | DIRECT | logit scaling and residual multiplier primitives | One lane; leave Granite MoE/hybrid variants to the template follow-up. |
| 10 | OLMo / OLMo2 / OLMo3 | DIRECT | explicit norm-placement strategy | One lane with per-generation config variants and golden residual-order checks. |
| 11 | Cohere / Cohere2 | DIRECT | QK normalisation and sliding-window parity | One lane; Cohere2-MoE is a later template variant. |
| 12 | Gemma / Gemma2 | DIRECT | audit reuse boundary with Gemma3 | One compatibility lane, not aliases in the Gemma3 loader; tied embeddings and scaling golden. |
| 13 | Llama4 text | TEMPLATE-EXISTS | `lane/moe-family` landed | One sparse text lane; vision is a separate TOWER-WORK lane. |
| 14 | DBRX / JetMoE / OLMoE / GraniteMoE / Qwen MoE | TEMPLATE-EXISTS | stable neutral MoE router/expert interface | Dispatch one family per lane; router score, top-k normalisation and shared experts are never assumed equal. |
| 15 | Jamba / Zamba / Falcon-H1 / Nemotron-H / OLMo-hybrid | TEMPLATE-EXISTS | mixer scheduling and recurrent-state ABI frozen | One hybrid family per lane, each with prefill/decode state goldens. |
| 16 | Mamba / Falcon-Mamba | TEMPLATE-EXISTS | [Mamba2 compatibility audit](mamba1-mamba2-compatibility-audit.md): incompatible | Mamba1 needs a channel-wise selective scan and x/dt projection primitive before a model lane. |
| 17 | RecurrentGemma | TEMPLATE-EXISTS | recurrent cache/state fixture format | One lane: recurrent gated linear unit plus local attention schedule. |
| 18 | GLM / GLM4 dense, Exaone4, Ernie4.5 dense, HunYuan dense | DIRECT | family-specific public fixture and tokenizer licences | Independent one-family lanes; demand justified by regional ecosystem presence, not presumed Llama aliases. |
| 19 | MPT / StableLM / Persimmon / SmolLM3 / Solar / Apertus / Arcee / Nemotron dense | DIRECT | none beyond quirks below | Opportunistic one-family lanes, ordered by a refreshed Hub query when a worker is free. |
| 20 | Llama4, Mllama, Gemma3n, Fuyu, Emu3, GIT, GOT-OCR2 | TOWER-WORK | neutral vision encoder/projector and processor contract | Tower lane first, then one integration lane per family. |
| 21 | Phi-4 multimodal, Moshi, MusicGen, Whisper decoder | TOWER-WORK | neutral audio codec/encoder and streaming feature contract | Audio foundation lane plus separate integration lanes; not part of a text-only Phi port. |
| 22 | BLT | EXOTIC | byte/patch latent encoder, entropy patching and patch-to-token cross-attention | Primitive lane, then model lane. |
| 23 | DiffLlama | EXOTIC | differential attention with paired query groups and lambda parameterisation | Attention primitive lane, then model lane. |
| 24 | BigBird / Reformer / XLNet | EXOTIC | block-sparse random/global attention; LSH attention; two-stream relative attention respectively | One primitive and one model lane per family; never combine. |
| 25 | xLSTM / HRM / CWM | EXOTIC | matrix-LSTM recurrent cell; hierarchical recurrent cycle; architecture-specific recurrent/world-model state | Research spike before any delivery estimate. |

Ranks 1–12 are the ready queue.  Ranks 13–19 are template or demand gated.
Ranks 20–25 must not be pulled as “just another decoder” work.

## Complete classified inventory

The following groups account for every registry key at the surveyed commit.
Names are the exact Transformers keys; crossed-off and in-flight keys are repeated
so that later registry diffs remain mechanical.

| Class/status | Registry families | Prerequisite or reason |
|---|---|---|
| **SHIPPED** | `bert`; `gemma3`, `gemma3_text`; `gemma4`, `gemma4_assistant`, `gemma4_text`, `gemma4_unified`, `gemma4_unified_assistant`, `gemma4_unified_text`; `llama`; `mamba2`; `ministral`, `ministral3`, `mistral`; `qwen3` | Existing packages; exact sub-architecture support still follows their own contracts. |
| **IN FLIGHT** | `deepseek_v2`, `deepseek_v3`, `deepseek_v32`, `deepseek_v4`; `mixtral` | `lane/moe-family`. |
| **DIRECT** | `apertus`; `arcee`; `biogpt`; `bloom`; `codegen`; `cohere`, `cohere2`; `cpmant`; `ctrl`; `dots1`; `ernie`; `ernie4_5`; `exaone4`; `falcon`; `flex_olmo`; `gemma`, `gemma2`; `glm`, `glm4`; `gpt-sw3`, `gpt2`, `gpt_bigcode`, `gpt_neo`, `gpt_neox`, `gpt_neox_japanese`, `gptj`; `granite`; `helium`; `hunyuan_v1_dense`; `hyperclovax`; `jais2`; `laguna`; `lfm2`; `mellum`; `minicpm3`; `mimo_v2_flash`; `modernbert-decoder`; `mpt`; `nanochat`; `nemotron`; `olmo`, `olmo2`, `olmo3`; `openai-gpt`; `opt`; `persimmon`; `phi`, `phi3`; `qwen2`; `seed_oss`; `smollm3`; `solar_open`; `stablelm`; `starcoder2`; `vaultgemma`; `xglm`; `youtu`; `zaya` | Existing dense primitives, but every row inherits the quirks register below. |
| **DIRECT, legacy/low priority** | `bart`; `bert-generation`; `blenderbot`, `blenderbot-small`; `camembert`; `data2vec-text`; `electra`; `marian`; `mbart`; `megatron-bert`; `mvp`; `pegasus`; `plbart`; `prophetnet`; `rembert`; `roberta`, `roberta-prelayernorm`; `roc_bert`; `roformer`; `xmod`; `xlm`, `xlm-roberta`, `xlm-roberta-xl` | Causal compatibility heads on encoder or encoder-decoder families.  Require a real causal checkpoint before dispatch; otherwise their principal task belongs outside this queue. |
| **TEMPLATE-EXISTS — MoE** | `afmoe`; `cohere2_moe`; `dbrx`; `ernie4_5_moe`; `exaone_moe`; `glm4_moe`, `glm4_moe_lite`, `glm_moe_dsa`; `gpt_oss`; `granitemoe`, `granitemoeshared`; `hunyuan_v1_moe`; `jetmoe`; `lfm2_moe`; `llama4`, `llama4_text`; `longcat_flash`; `minimax`, `minimax_m2`; `olmoe`; `phimoe`; `qwen2_moe`, `qwen3_moe`; `qwen3_5_moe`, `qwen3_5_moe_text`; `qwen3_next` | Land and stabilise Mixtral/DeepSeek router, expert and shared-expert templates; add DSA/MoE attention only where required.  Qwen3.5/Next also require their gated-DeltaNet schedule. |
| **TEMPLATE-EXISTS — SSM/recurrent/hybrid** | `bamba`; `doge`; `falcon_h1`, `falcon_mamba`; `granitemoehybrid`; `jamba`; `mamba`; `nemotron_h`; `olmo_hybrid`; `qwen3_5`, `qwen3_5_text`; `recurrent_gemma`; `rwkv`; `zamba`, `zamba2` | Existing Mamba2/RWKV7/DeltaNet/composition experience, but version-specific state equations and attention schedules need dedicated implementations.  Qwen3.5 alternates gated DeltaNet and full-attention layers. |
| **TOWER-WORK — vision/document** | `aria_text`; `emu3`; `fuyu`; `gemma3n`, `gemma3n_text`; `git`; `got_ocr2`; `minimax_m3_vl_text`; `mllama`; `trocr` | Vision/document encoder, projector, processor and modality-token contract.  A `_text` key only removes the tower at runtime; it does not prove useful standalone checkpoints. |
| **TOWER-WORK — audio/music** | `moshi`; `musicgen`, `musicgen_melody`; `phi4_multimodal`; `whisper` | Audio encoder/codec, feature processor, delay/stream schedule and modality cache. |
| **EXOTIC — sparse/alternative attention** | `big_bird`; `bigbird_pegasus`; `diffllama`; `reformer`; `xlnet` | Block-sparse random/global, differential, LSH, or two-stream relative attention primitive as applicable. |
| **EXOTIC — representation/recurrent** | `bitnet`; `blt`; `cwm`; `hrm_text`; `hy_v3`; `xlstm` | Native ternary linear kernels; byte-patch latent pipeline; or a new family-specific recurrent/state primitive. `hy_v3` requires its hybrid linear-attention primitive to be specified before estimation. |

This grouping is intentionally conservative.  `TEMPLATE-EXISTS` means the neutral
abstraction is plausible, not that weights can be relabelled.  Conversely,
`DIRECT` permits a new configuration and weight map but not backend-specific code.

## Quirks register for future briefs

Every family brief must turn applicable warnings into configuration fixtures and
goldens, not prose-only assumptions.

| Family/group | Port warning inherited by the lane |
|---|---|
| Phi/Phi-3 | Phi-2 and Phi-3 are not one config: partial rotary dimensions, long-context RoPE scaling and sliding-window behaviour differ.  Verify `tie_word_embeddings` per checkpoint. |
| GPT-2/GPT-J/Neo/NeoX/BigCode | Learned absolute positions versus rotary positions; GPT-2 `Conv1D` weights are stored transposed; BigCode can use multi-query attention; GPT-J/NeoX can use parallel residual paths and partial rotary dimensions.  Byte-level BPE special-token behaviour is fixture material. |
| OPT/BLOOM/Falcon | OPT exposes pre/post-norm and projection dimensions; BLOOM uses ALiBi, embedding layer norm and fused QKV layout; Falcon has `new_decoder_architecture`, parallel attention, ALiBi/rotary and MQA/GQA variants. |
| Gemma generations | Embedding scaling, RMSNorm's unit-offset convention, alternating local/global attention, logit soft-capping and tied embeddings vary by generation.  Do not load Gemma/Gemma2 through Gemma3 by resemblance. |
| Llama relatives | RoPE variants include linear, dynamic/NTK, YaRN and Llama-3 scaling; GQA and tied embeddings are config choices.  Llama4 additionally adds sparse experts and, for full models, a vision tower. |
| Mistral/Ministral | Sliding-window schedule, head dimensions and RoPE parameters vary; tokenizer generations differ between SentencePiece and Tekken assets. |
| Qwen2/Qwen3/GLM | Qwen uses GQA, RMSNorm and family-specific RoPE/sliding-window rules; Qwen2 and Qwen3 tokenizer/chat-template details are not interchangeable.  GLM variants may use partial RoPE, multi-token prediction or MoE/DSA paths. |
| Cohere/Granite/OLMo | QK normalisation, sliding-window schedules, residual/logit multipliers, embedding scaling and norm placement are architectural, not harmless config noise. |
| MoE families | Golden router logits, top-k selection, score normalisation, capacity/drop policy, shared experts and expert weight packing.  DeepSeek MLA/latent attention and GLM DSA require attention work beyond an MoE template. |
| SSM/hybrid families | Prefill and one-token decode must agree on convolution/recurrent state.  Record layer schedules, state shapes, reset semantics and cache serialisation; Mamba1 is not Mamba2 and upstream `rwkv` is not RWKV7. |
| Encoder/decoder compatibility heads | Cross-attention flags, causal masks, shifted decoder inputs and tied encoder/decoder/output embeddings vary.  Tokenizers include SentencePiece, byte BPE and WordPiece; a causal head alone does not make a useful continuation model. |
| Multimodal/audio | Modality token insertion, projector shape, tower normalisation, processor resize/resample rules and placeholder expansion affect text positions and cache offsets.  Tokenizer plus processor versions must be pinned together. |
| BitNet/BLT/xLSTM | Do not emulate the defining primitive unnoticed with dense operations: ternary weights, dynamic byte patches and matrix memory respectively need explicit neutral contracts and backend cost measurements. |

## Lane acceptance contract

Each dispatched lane should follow the established Llama/Mistral/Qwen3 method:

1. pin a Transformers commit and a real model revision in the brief;
2. add a configuration fixture copied from that revision, including tokenizer and
   special-token metadata;
3. obtain prompt-token, hidden-state/logit and greedy-token goldens from
   Transformers before writing the Go execution path;
4. map canonical Hugging Face tensor names explicitly, including fused/transposed
   layouts, tied aliases and expert packing;
5. prove prefill and cached decode separately, then malformed configuration and
   missing-weight failures; and
6. keep all execution on neutral primitives.  A missing primitive changes the
   family to **EXOTIC** (or creates a prerequisite lane); it does not justify a
   backend dependency in `go/model`.

Before pulling a queued family, refresh the Transformers registry diff and the Hub
figures.  New aliases should be assigned to an implementation family; they should
not automatically jump the queue.
