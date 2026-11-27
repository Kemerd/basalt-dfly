# Training a DFly2 drafter

`basalt-dfly2-train` trains a DFly2 block drafter for a target whose attention keeps a per-head key/value cache and
selects the cells each row reads (Qwen3.8-Flash-Next, architecture `qwen4exp`). It is a C++ program on LibTorch in
`tools/dfly2-train/`, built apart from the engine.

**Status.** The trainer, its file formats and its CPU tests are in the tree. The engine's teacher export, which writes
the shards the trainer reads, is not written yet, so there is no shard of a real model to train on: the tests train
on a small synthetic target. The trainer has not been run on a GPU.

## What is trained

A drafter proposes `block - 1` tokens in one pass from the last accepted token (the anchor). It is a dense stack of
five layers on one residual stream, each paired with one of the target's last five sparse-attention layers:

- A block's rows are the anchor's embedding, mixed with the target's last hidden state through a gate that starts at
  zero, and `block - 1` copies of the mask token's embedding.
- Each layer's attention reads the cells its paired target layer selected for the row before the anchor, straight
  from that layer's cache, and the block's own rows up to the current one. Its projections start as the paired
  layer's, so the block's own keys and values land in the space of the cached ones.
- A correction step adjusts each row with the embedding of the token before it; it starts as the identity.
- The target's output head turns a row into a draft; a one-logit head estimates the chance the target accepts it.

The target's embedding and output head are frozen and are not stored with the drafter. The forward is written out in
full in [`dfly2_checkpoint.hpp`](../include/basalt/artifact/dfly2_checkpoint.hpp), with every tensor's name and shape.

The loss, per drafted row, against the target's own next-token distribution at that position:

| Term | What it is |
|---|---|
| KL | over the teacher's ten likeliest tokens, both distributions normalised over the whole vocabulary |
| confidence | binary cross-entropy of the confidence logit against the row's accept chance `sum(min(p, q))` |
| row weights | KL and confidence are weighted so a row counts for the rows after it that its draft keeps alive (D-PACE) |
| accepted length | `1 - (1/rows) * sum_k prod_{j<=k} accept_j` per block: the drafts a sampling verifier would reject |

Matrices are trained with Muon and everything else with AdamW, on FP32 master weights behind BF16 weights, with a
joint gradient clip and a linear-warmup cosine schedule.

## Building

LibTorch is not fetched. Unpack a release and name it with `CMAKE_PREFIX_PATH` or `Torch_DIR`. LibTorch for Windows has
no debug build, so configure `Release` or `RelWithDebInfo`. A CUDA build of LibTorch also needs its CUDA toolkit at
configure time, and its CMake package asks the GPU for its architecture once.

```
cmake -S tools/dfly2-train -B build-dfly2-train -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
      -DCMAKE_PREFIX_PATH=<libtorch>
cmake --build build-dfly2-train
ctest --test-dir build-dfly2-train -L cpu --output-on-failure
```

Or with the engine: add `-DBASALT_DFLY2_TRAIN=ON -DCMAKE_PREFIX_PATH=<libtorch>` to the root configuration. The option
is off by default and the engine's build never needs LibTorch.

The program is written to `bin/basalt-dfly2-train`. It loads LibTorch's shared libraries at run time: on Windows put
`<libtorch>/lib` on `PATH`. The tests are CPU-only, hide every GPU from themselves, and take about a minute.

Built and tested with LibTorch 2.15.0 (cu134) and MSVC on Windows. Linux is not built yet.

## Commands

Every path is UTF-8. `--help` lists every option with its default.

### `init`

```
basalt-dfly2-train init --model PACK.basalt --out drafter.safetensors
```

Writes an untrained drafter for the pack's target. The geometry (heads, head width, rotation, selection width) and
the paired layers come from the pack's metadata.

| Option | Meaning |
|---|---|
| `--layers A,B,...` | the target layers to pair with; default: the last `--draft-layers` layers that have the sparse attention |
| `--draft-layers N` | drafter layers (5) |
| `--block N` | rows of a block: the anchor and `N - 1` drafts (8) |
| `--mask-id N` | the mask token (248077). It must be an id the tokenizer never produces; check it against the tokenizer |
| `--rms-eps X` | eps of the drafter's own norms (1e-5) |
| `--seed N` | seed of the random initialisation (42) |

### `train`

```
basalt-dfly2-train train --model PACK.basalt --shards DIR --run DIR --init drafter.safetensors
```

Starts a run from `--init`, or resumes the newest checkpoint in `--run` when there is one (`--init` is then ignored).
A resumed run must repeat the seed, `--blocks-per-step`, the schedule's length, `--dtype` and the shards it was started
with, and then takes the steps the uninterrupted run would have taken.

| Option | Meaning |
|---|---|
| `--steps N`, `--epochs X` | the run's length: optimiser steps, or passes over the shards' blocks (1 epoch) |
| `--blocks-per-step N` | blocks behind one optimiser step (8192) |
| `--blocks N` | blocks of one forward and backward (256). A forward stays inside one sequence |
| `--loss-blocks N` | blocks whose logits are alive at once (32) |
| `--attention-blocks N` | blocks whose attention scores are alive at once (64) |
| `--device D` | `cpu`, `cuda` or `cuda:N` (`cuda`) |
| `--dtype T` | the drafter's weights in the forward: `bf16` or `f32` (`bf16`). The master weights are FP32 either way |
| `--head-dtype T` | the frozen output head on the device: `f32` or `bf16` (`f32`) |
| `--seed N` | seed of the data order (42) |
| `--save-every N`, `--keep N` | steps between checkpoints (250) and checkpoints kept (3) |
| `--log-every N` | steps between progress lines (10) |
| `--queue-gb X` | host memory of the sequences read ahead (4) |
| `--lr`, `--min-lr`, `--warmup-ratio` | the schedule (5e-4, 5e-5, 0.04) |
| `--weight-decay`, `--beta1`, `--beta2`, `--adam-eps` | AdamW (0.01, 0.9, 0.95, 1e-8) |
| `--muon-weight-decay`, `--muon-momentum`, `--muon-ns-steps`, `--muon-rms` | Muon (0.1, 0.95, 5, 0.2) |
| `--muon-plain-momentum` | Muon without Nesterov's look-ahead |
| `--muon-head-rows N` | rows of a query projection orthogonalised together (the head width); 0: the whole matrix |
| `--grad-clip X` | joint gradient norm clip (1) |
| `--kl-topk`, `--dpace-alpha`, `--e2e-weight`, `--confidence-weight` | the loss (10, 0.5, 1, 1) |

Memory on the device is set by three options: `--blocks` bounds the layers' activations, `--loss-blocks` the logits
(`blocks x (block - 1) x vocabulary`), and `--attention-blocks` the attention scores. The step's gradient does not
depend on the last two. The loss's row weights are normalised over one forward, so `--blocks` changes the gradient
slightly when a step spans several forwards.

Sequences are read on a thread of their own, ahead of the loop. The order is a function of the seed: an epoch shuffles
the sequences, and a sequence's blocks are taken in a shuffle of their own.

A progress line, also appended to `train.log` in the run directory:

```
step 40/120  epoch 12.31  tokens 7.7k  loss 2.5841 (kl 1.2060  e2e 0.6841  conf 0.6940)  accepted 0.95 sampled, 1.10 greedy of 3  lr 4.06e-04  grad 4.998  789 blocks/s  0m03s elapsed, 0m06s left
```

`tokens` is the drafted rows trained on so far. `accepted` is the drafts per block the target would accept, over the
blocks since the last line, teacher-forced: under sampling, and under greedy verification (below).

**Checkpoints.** The run directory holds, per saved step:

| File | Contents |
|---|---|
| `step-N.safetensors` | the drafter: what `eval`, a later `--init` and the engine's side read |
| `step-N.state.safetensors` | the optimiser's buffers and counters and the run's place in the data, for a resume |

They are written on a thread of their own, each through a temporary file, the drafter's last: a step whose drafter
file exists is whole.

**Stopping.** Ctrl+C (or SIGTERM) finishes the forward in hand, drops the unfinished step, writes a checkpoint of the
last whole step and exits. The same command resumes. A second Ctrl+C ends the process at once.

### `eval`

```
basalt-dfly2-train eval --model PACK.basalt --shards DIR --checkpoint step-N.safetensors
```

Measures a drafter over shards (held-out ones, for a number that means something) and prints the loss terms, each
drafted row's accuracy, and four accepted lengths in drafts per block. It takes `train`'s device, memory, seed and loss
options, and `--max-blocks N` to measure a sample.

| Line | Meaning |
|---|---|
| teacher-forced, corpus token | `sum_k prod_{j<=k} acc_j`, `acc_j` the share of blocks whose row `j` drafts the corpus token |
| teacher-forced, greedy | the same against the teacher's likeliest token: what greedy verification accepts |
| teacher-forced, sampling | the mean of `sum_k prod_{j<=k} sum(min(p, q))`: what a sampling verifier accepts |
| serving chain, greedy | each row corrected with the token the row before it drafted, as when serving; a block scores the rows before its first draft that is not the teacher's likeliest |

The teacher-forced lines correct each row with the corpus token before it, which serving cannot do; the last line is
the one to compare with served speed. A verify pass yields one more token than it accepts.

### `fixture`

```
basalt-dfly2-train fixture --model PACK.basalt --shards DIR --checkpoint FILE --out FIXTURE.safetensors --count 32
```

Writes blocks of the shards (eval's order, at most four from a sequence, none anchored at a sequence's first row) with
the drafter's forward over them in the checkpoint's own precision. Per block `i`, under `block.<i>.`: the inputs
(`anchor`, `pos`, `h_s`, `emb_anchor`, `emb_mask`, `emb_prev`, and per draft layer `layer.<l>.sel`, `k_ctx`, `v_ctx`)
and the outputs (`z`, `h_corrected`, `conf_logit`, `draft_ids`, each drafted row reading the draft before it). The
engine's `qwen4exp_dfly_real PACK FIXTURE` runs the same blocks through its block pass on the pack's drafter (attach
the same checkpoint first) and reports the error of each output and, per draft position, the share of drafts that
agree ([building](building.md)).

## Teacher shards

A shard is a file named `*.dfly2` holding whole sequences, one document each. The layout, its writer and its reader
are [`dfly2_shard.hpp`](../include/basalt/artifact/dfly2_shard.hpp); the engine's export and the trainer share that
header.

| Part | Contents |
|---|---|
| header (512 bytes) | the geometry (hidden width, vocabulary, key/value heads, head width, rotated width and base, selection width), the paired target layers, and the pack's name, size and `meta` digest |
| per sequence | `tokens` int32 `[L]` |
| | `hidden` FP16 or FP32 `[L][hidden]`: the vector that enters the output head at each row |
| | per paired layer, `K` and `V` FP16 `[L][kv heads][head width]`: the layer's cached cells, keys normed and rotated |
| | `anchors` int32 `[A]`, ascending: the positions the export sampled |
| | `selection` int32 `[A][layers][selection width]`: the cells each paired layer selected for the row before the anchor, ascending, padded with -1 |
| index | each sequence's offset |

A sequence also records the rotary position of its first row. The trainer refuses a shard whose geometry or paired
layers differ from the drafter's, or that names another pack than `--model`. It trains the anchors whose block lies
inside the sequence (`anchor + block <= L`); the teacher's logits are computed in the trainer from `hidden` and the
pack's output head.

Per token a shard holds the hidden row and two cells per paired layer: 15,360 bytes for Flash-Next with FP16 hidden
rows and five layers, plus about 41 KiB per anchor of selections.

## Not verified

- Nothing has run on a GPU: speed, device memory and BF16 on CUDA are untested. BF16 weights are tested on the CPU.
- No shard of a real model exists yet, and the real pack's metadata and tensor types have only been read from the
  converter's code.
- The default mask token 248077 has not been checked against the tokenizer.
- The interrupt path is tested through the library's stop callback, not by sending a signal to the program.
