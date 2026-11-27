# Qwen3.8-Flash-Next

Runs with the same command as any model: `basalt serve PACK`. What it supports next to the others: [models](index.md). Rounding and
quality: [precision](../precision/qwen38-flash-next.md).

## Pack

Packs on [Hugging Face](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt), all named `Qwen3.8-FN-` plus:

- `NVFP4-DQ6_K` (default) and `NVFP4-DQ8`: NVIDIA's NVFP4 experts. `D` is the dense projections' format: Q6_K, or Q8_0 for DQ8.
  DQ8 is slower than DQ6_K and has a lower KLD ([precision](../precision/qwen38-flash-next.md#6-bit-dense-projections)).
- `UD-IQ1_S`, `UD-IQ1_M`, `UD-IQ2_XS`, `UD-IQ3_XXS`, `UD-IQ4_XS`, `UD-Q4_K_XL`, `UD-Q5_K_XL`, `UD-Q6_K_XL` and `Q8_0`: Unsloth's GGUF types.
- Unsloth's `UD-IQ1_S` GGUF converts as shipped (70.4 GB). One of its layers has IQ2_S gate and IQ2_XXS up rows; every layer runs
  the same fast kernels as the other UD packs.

Sizes are on the card. The layout, and how to build a pack: [model file](../model-file.md).

## Memory

- **Context:** `--max-context`, default 8,192. The KV cache grows with it. `--kv fp16` (default), `int8` (half the VRAM), `q4_0`
  and `k8v4` trade precision for VRAM ([precision](../precision/qwen38-flash-next.md#measured-costs)).
- **Experts:** the main card's cache holds the most routed ones, system RAM the rest, and an adaptive tier moves them as the
  conversation changes ([adaptive tier](../configuration/adaptive.md)).
- **RAM:** before the RAM copy, the checkpoint blocks of every slot are allocated (page-locked, so a checkpoint saves and
  restores at link speed and a save runs beside the first decode window) and the RAM of the page-locked prompt buffers
  is held: the PLE rows at the largest chunk, a split's hand-offs, and the peer's prompt layers' weights, hand-offs and staging.
  Each buffer takes its held RAM when it is made; what none takes goes back before the top-up. The copy then leaves
  `--ram-reserve-mib` free beside the driver's page tracking (`RAM floor X GiB of N GiB free` when experts stay on the SSD).
  The start's pass fills the main card first; the copy takes its RAM as the pass reads into it and is page-locked once read.
- **Cards:** `gpu` names the cards and `gpu-mode` what they do ([GPUs](../configuration/gpus.md)); one card is the default, and the
  start warns when `gpu` is unset and more are visible.
  - `peer` (up to four cards): the main card runs every layer; each peer holds experts the main card does not, computes their rows
    in every decode window and their share of each prompt chunk, and reads a share of the layers of long prompts. A peer faster
    than the main card also takes a share of each GDN mixer and of the head. With the peer, 64K prefill speed is level with the
    path before the multi-GPU rework (9,823 against 9,835 tok/s) and the normal peer run scores BF16 KLD 0.06174 against
    0.06221 on one card, top-1 91.84% against 91.80%.
  - `split` (up to eight cards): the layers are cut by `gpu-share` (free VRAM by default), each card holding its layers' dense
    weights, caches and experts; windows overlap on two cards ([GPUs](../configuration/gpus.md#split-mode)).
  - `tensor` is refused.
- **Elastic KV:** with `--max-context` above 32,768 (and not with `--kv-resident`, `--vram-elastic` or `gpu-mode split`), each session
  backs its first 32,768 cells at start. More cells come from the expert cache when a request needs them, and go back when it does
  not. An idle conversation's KV moves to RAM after 15 s. The startup line `elastic KV: ...` shows the figures, and
  `elastic KV: about N cells fit in the S sessions ...` gives the ceiling: the cells all sessions can hold before the expert
  cache reaches the slots a prompt chunk needs, past which a request is refused. `/v1/status` gives its current value as
  `context.elastic_kv_ceiling` (null without elastic KV), so a client can compact a conversation first.

## Long prompts

- `--prefill auto` reads up to 16,384 tokens per chunk (32,768 for NVFP4). The CPU computes 0.6 of the prompt experts that stream
  (`--prefill-cpu-share`).
- A fresh part of 64 tokens or fewer goes through decode windows (`--short-read`).
- With `gpu-mode peer`, a long prompt is read as a pipeline over the cards ([GPUs](../configuration/gpus.md#peer-mode)).
- Prompts beside decoding slots: [running](../running.md#prompts-beside-decoding-slots).

## Drafting

On when the pack has the MTP drafter. `--spec N` sets the window (default 8: the last accepted token plus up to 7 drafts).
`--spec 0` turns it off. `--suffix-draft` (default 3) also fills a window from an earlier repeat in the context. Output is exactly
plain decoding's ([precision](../precision/qwen38-flash-next.md#drafting-does-not-change-output)). Structured output drafts best
and prose least.

## Teacher export

`bin/basalt --model PACK.basalt --dfly2-export DIR --dfly2-docs FILE --max-context C` generates nothing: it writes what the
model computes for each document of `FILE` into `DIR`, as the shards a DFly2 drafter is trained on
([training](../dfly2-training.md#teacher-shards)). No drafter runs and the expert tiers stay as loaded.

- **Input:** one document a line, as token ids separated by spaces or commas. An empty line or an id outside the vocabulary
  ends the run with the line's number. A document longer than `--max-context` is cut to it and counted.
- **Two kinds of row:** a document's last `--dfly2-window-rows N` rows (default 512; 0: none) are read through verify windows
  of 8, as generated rows are when serving, and the rows before them by the batched prompt path (`--prefill`, default auto).
  With `n` rows the windows start at row `n - min(N, n - 1)`.
- **Per row:** the head's input and the K and V cells of layers 31, 35, 39, 43 and 47, all FP16: 15,360 bytes.
- **Anchors:** one per 64 prompt-path rows and one per 8 window rows, drawn from a fixed seed and the line's number, each with
  the cells those five layers selected for the row before it (up to 2,051 a layer). Every anchor leaves room for a block of 8.
- **Files:** `shard-NNNNN.dfly2`, each closed at a document's end once it holds 262,144 tokens and named `.part` until it is
  whole. `export.json` holds the run's settings: the pack, the documents file, the window rows, the context, the prompt chunk,
  the CPU share, the expert cache's slots and the rotation's base.
- **Restart:** the same command continues after the documents of the closed shards. A setting that differs from `export.json`
  is refused with its name and the stored value; when `--prefill auto` settles on another chunk or the expert cache on
  another size (both follow the free VRAM), pass the stored ones as `--prefill N` and `--expert-cache N`.
- **Check:** at its end the run reads its first document once more and reports whether the rows, cells and selections are
  the same as the first time. It then reads every closed shard back through the shard reader: each sequence's tokens
  against the documents file, its anchors, the form of every selection (every cell up to 2,051; past that whole blocks of
  4, the lowest cells of at most one more block, and the unfinished last block), and the first document's arrays against
  what was computed. A difference is an error line and exit code 1; the shards stay.
- **Cache format:** `--kv fp16`, the one-shot default. A shard holds FP16 cells, which no other format keeps; an INT8
  cache's codes are a function of those values, 64 at a time, so the trainer can form them from a shard.
- **Refused:** `--serve`, `--kv` other than `fp16`, `--kv-resident`, `--prefill 0`, a control vector, a rope scaling and
  `gpu-mode split`.
- **Memory:** VRAM as a one-shot run with `--spec 8`. RAM: two page-locked buffers of one prompt chunk's rows and cells each
  (31 MB at a chunk of 2,048, about 0.5 GB at 32,768).

## Parallel slots

`--parallel N` (1 to 8) serves N requests at once ([running](../running.md#concurrent-requests---parallel-n)).

## Images and video

`--vision dynamic`, a GPU number or `cpu` ([vision](../running.md#vision---vision)). Video works with images on and FFmpeg on the
machine; frames decode on the GPU with NVDEC ([video](../running.md#video)).

## Lossy flags

Off by default. `--relaxed-drafting`, `--route-prior` and `--route-tail-skip` each change output, and each is named in one start
warning. What they do and what they cost: [precision](../precision/qwen38-flash-next.md#relaxed-drafting); the two routing flags
work the same on every MoE model ([quality](../precision/quality.md#lossy-routing-flags)).

## Limits

- **Two-card idle sleep** can drop the second card's experts until restart (seen at context 8,192 without elastic KV, not at 64K or
  262,144).
