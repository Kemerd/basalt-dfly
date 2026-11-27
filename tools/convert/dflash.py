"""A DFlash 2 block drafter (z-lab's DFlash2DraftModel: config.json and safetensors, all BF16) as the tensors and the
metadata the engine reads, for any model that carries one:

  dflash.fc.N.weight             the feature projection, one [H][H] block per target layer, Q8_0
  dflash.blk.L.*                 a layer's projections Q8_0, its norms (plain RMS weights, no offset) and its
                                 convolutions' base kernels F32
  dflash.hidden_norm.weight, dflash.norm.weight    F32
  dflash.selector.*              BF16 as stored
  <arch>.dflash.*                the drafter's shape, as GGUF metadata

A pack built from a checkpoint holds the tensors in `gguf`. A pack that exists takes them as a section of its own,
`dflash` (convert.py --drafter DIR --into PACK): each at a multiple of TENSOR_ALIGN, listed in meta.tensors with that
section's name.

Needs numpy.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

import numpy as np

import checkpoint as C
import model_sections as S

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from model_file import TENSOR_ALIGN  # noqa: E402

SECTION = "dflash"

# drafter tensor (after `layers.N.`) -> (GGUF name after `dflash.blk.N.`, type)
LAYER = {
    "input_layernorm.weight": ("attn_norm.weight", "F32"),
    "post_attention_layernorm.weight": ("post_attention_norm.weight", "F32"),
    "self_attn.q_proj.weight": ("attn_q.weight", "Q8_0"), "self_attn.k_proj.weight": ("attn_k.weight", "Q8_0"),
    "self_attn.v_proj.weight": ("attn_v.weight", "Q8_0"), "self_attn.o_proj.weight": ("attn_output.weight", "Q8_0"),
    "self_attn.q_norm.weight": ("attn_q_norm.weight", "F32"), "self_attn.k_norm.weight": ("attn_k_norm.weight", "F32"),
    "mlp.gate_proj.weight": ("ffn_gate.weight", "Q8_0"), "mlp.up_proj.weight": ("ffn_up.weight", "Q8_0"),
    "mlp.down_proj.weight": ("ffn_down.weight", "Q8_0"),
    "attention_conv.base_kernel": ("attn_conv.base", "F32"), "attention_conv.kernel_projection.weight": ("attn_conv.weight", "Q8_0"),
    "mlp_conv.base_kernel": ("ffn_conv.base", "F32"), "mlp_conv.kernel_projection.weight": ("ffn_conv.weight", "Q8_0"),
}
GLOBAL = {
    "hidden_norm.weight": ("hidden_norm.weight", "F32"), "norm.weight": ("norm.weight", "F32"),
    "candidate_selector.hidden_projection.weight": ("selector.hidden.weight", "BF16"),
    "candidate_selector.predecessor_codebook": ("selector.pred", "BF16"),
    "candidate_selector.successor_codebook": ("selector.succ", "BF16"),
}
FORM = "projections Q8_0 from BF16, norms and conv base kernels F32, selector BF16"


def _job(args):
    """BF16 rows [row0, row0 + rows), columns [c0, c0 + ne0) of a [.][width] matrix, as Q8_0 (a string: what stops it)."""
    path, off, width, c0, ne0, row0, rows = args
    mm = np.memmap(path, dtype="<u2", mode="r", offset=off + row0 * width * 2, shape=(rows, width))
    x = S._bf16_to_f32(np.ascontiguousarray(mm[:, c0:c0 + ne0])).reshape(-1)
    return S.q8_0(x) if np.isfinite(x).all() else "a non-finite value"


class Drafter:
    """The drafter in directory `d`, for a model of architecture `arch` with `hidden` channels, a vocabulary of `vocab`
    tokens and `layers` layers."""

    def __init__(self, d, arch: str, hidden: int, vocab: int, layers: int):
        self.dir, self.arch = pathlib.Path(d), arch
        cfg = self.dir / "config.json"
        if not cfg.is_file():
            raise S.ConvertError(f"{cfg} not found")
        try:
            self.config = json.loads(cfg.read_text(encoding="utf-8"))
        except ValueError as e:
            raise S.ConvertError(f"{cfg}: {e}")
        if not isinstance(self.config, dict) or self.config.get("architectures") != ["DFlash2DraftModel"]:
            raise S.ConvertError(f"{self.dir}: not a DFlash2DraftModel")
        try:   # every key the pack records is read here, before anything is written
            targets = self.kv()[arch + ".dflash.target_layers"]
            H, n_layers = int(self.config["hidden_size"]), int(self.config["num_hidden_layers"])
            fits = H == hidden and int(self.config["vocab_size"]) == vocab
        except (KeyError, TypeError, ValueError) as e:
            raise S.ConvertError(f"{cfg}: a key is missing or malformed ({e})")
        if not fits:
            raise S.ConvertError(f"{self.dir}: the drafter's width or vocabulary differs from the model's")
        if not targets or any(not 0 <= t < layers for t in targets):
            raise S.ConvertError(f"{self.dir}: target layers {targets} outside the model's {layers} layers")
        if self.config.get("is_causal") is not False or set(self.config.get("layer_types", [])) != {"sliding_attention"}:
            raise S.ConvertError(f"{self.dir}: only the non-causal sliding-window drafter is converted")
        self.files, self.t = sorted(self.dir.glob("*.safetensors")), {}
        if not self.files:
            raise S.ConvertError(f"{self.dir}: no safetensors files")
        for f in self.files:
            head, base = S.read_safetensors_header(f)
            for k, v in sorted(((k, v) for k, v in head.items() if k != "__metadata__"), key=lambda kv: kv[1]["data_offsets"][0]):
                if k in self.t:
                    raise S.ConvertError(f"{f.name}: duplicate tensor {k}")
                a_, b_ = v["data_offsets"]
                self.t[k] = (f, base + a_, v["dtype"], list(v["shape"]), b_ - a_)
        self.tensors, self.slices = {}, {}
        for k, (f, off, dt, shape, n) in self.t.items():
            if dt != "BF16":
                raise S.ConvertError(f"{k} is {dt}, expected BF16")
            if n != 2 * int(np.prod(shape)):
                raise S.ConvertError(f"{k}: {n} bytes, BF16 of shape {shape} takes {2 * int(np.prod(shape))}")
            if k == "fc.weight":   # one [H][H] block per target layer, so the features project as each layer ends
                if shape != [H, H * len(targets)] or H % 32:
                    raise S.ConvertError(f"{self.dir}: fc.weight is {shape}")
                for j in range(len(targets)):
                    name = f"dflash.fc.{j}.weight"
                    self.tensors[name] = C.Tensor(name, "Q8_0", k, "id", [H, H])
                    self.slices[name] = (j * H, shape[1])
                continue
            m = re.fullmatch(r"layers\.(\d+)\.(.+)", k)
            rule = LAYER.get(m.group(2)) if m else GLOBAL.get(k)
            if rule is None:
                raise S.ConvertError(f"{self.dir}: drafter tensor {k} has no counterpart")
            ne = list(reversed(shape))
            if rule[1] == "Q8_0" and (len(ne) != 2 or ne[0] % 32):
                raise S.ConvertError(f"{k}: a tensor of shape {shape} cannot be Q8_0")
            name = "dflash." + (f"blk.{m.group(1)}.{rule[0]}" if m else rule[0])
            self.tensors[name] = C.Tensor(name, rule[1], k, "id", ne)
        # what the engine reads: a feature block per target layer, the globals and every layer's tensors
        need = {f"dflash.fc.{j}.weight" for j in range(len(targets))} | {"dflash." + r[0] for r in GLOBAL.values()}
        need |= {f"dflash.blk.{l}.{r[0]}" for l in range(n_layers) for r in LAYER.values()}
        if need != set(self.tensors):
            odd = sorted(need ^ set(self.tensors))
            raise S.ConvertError(f"{self.dir}: {len(odd)} tensors missing, or of a layer past the drafter's {n_layers}: "
                                 + ", ".join(odd[:8]))

    def kv(self) -> dict:
        c, dc = self.config, self.config["dflash_config"]
        A = self.arch + ".dflash."
        return {A + "block_count": int(c["num_hidden_layers"]), A + "block_size": int(dc["block_size"]),
                A + "mask_token_id": int(dc["mask_token_id"]), A + "target_layers": [int(x) for x in dc["target_layer_ids"]],
                A + "conv_kernel": int(dc["conv_kernel_size"]), A + "conv_group": int(dc["conv_group_size"]),
                A + "selector_rank": int(dc["selector_rank"]), A + "selector_top_k": int(dc["selector_top_k"]),
                A + "sliding_window": int(c["sliding_window"]), A + "attention.head_count": int(c["num_attention_heads"]),
                A + "attention.head_count_kv": int(c["num_key_value_heads"]), A + "attention.key_length": int(c["head_dim"]),
                A + "feed_forward_length": int(c["intermediate_size"]),
                A + "rope.freq_base": C.f32(c["rope_parameters"]["rope_theta"]), A + "attention.layer_norm_rms_epsilon": C.f32(c["rms_norm_eps"])}

    def record(self, source_entry) -> dict:
        """The pack's meta.dflash: the directory, its files (`source_entry` of each) and the tensors' forms."""
        return {"source": self.dir.name, "files": [source_entry(p) for p in self.files], "form": FORM}

    def values(self, name) -> np.ndarray:
        f, off, dt, shape, n = self.t[self.tensors[name].src]
        u = np.fromfile(f, dtype="<u2", count=n // 2, offset=off)
        return S._bf16_to_f32(u).reshape(shape)

    def float_bytes(self, name) -> bytes:
        t = self.tensors[name]
        x = self.values(name)
        if not np.isfinite(x).all():
            raise S.ConvertError(f"{name} has a non-finite value")
        if t.type_name == "F32":
            return np.ascontiguousarray(x, dtype="<f4").tobytes()
        f, off, dt, shape, n = self.t[t.src]
        return np.fromfile(f, dtype=np.uint8, count=n, offset=off).tobytes()

    def chunks(self, name, pool, depth: int = 64):
        t = self.tensors[name]
        f, off, dt, shape, n = self.t[t.src]
        ne0, ne1 = t.ne
        c0, width = self.slices.get(name, (0, ne0))    # a whole matrix is the slice of all its columns
        step = max(1, C.VALUES_PER_JOB // ne0)
        jobs = ((str(f), off, width, c0, ne0, r, min(step, ne1 - r)) for r in range(0, ne1, step))
        for b in S.bounded_imap(pool, _job, jobs, depth):
            if isinstance(b, str):
                raise S.ConvertError(f"{name}: {b}")
            yield b

    def write(self, path, pool) -> dict:
        """The tensors as the `dflash` section's bytes in the file `path`, each at a multiple of TENSOR_ALIGN; returns
        their meta.tensors entries (offsets from the section's start)."""
        entries, pos = {}, 0
        with open(path, "wb") as f:
            for name, t in self.tensors.items():
                pad = (-pos) % TENSOR_ALIGN
                f.write(b"\0" * pad)
                pos += pad
                n = 0
                for b in (self.chunks(name, pool) if t.type_name == "Q8_0" else (self.float_bytes(name),)):
                    f.write(b)
                    n += len(b)
                if n != t.expected_bytes():
                    raise S.ConvertError(f"{name}: {n} bytes written, expected {t.expected_bytes()}")
                entries[name] = {"type": t.type_name, "type_id": t.type_id, "shape": [int(d) for d in t.ne], "bytes": n,
                                 "offset": pos, "section": SECTION}
                pos += n
        return entries


def section_problems(meta: dict, size) -> list[str]:
    """What is wrong with the `dflash` section of a pack whose meta is `meta`; `size` is the section's bytes, None when
    the pack has none. Its tensors lie one after another, aligned, each of the size its type and shape give."""
    tensors = meta.get("tensors") or {}
    entries = {n: e for n, e in tensors.items() if isinstance(e, dict) and e.get("section") == SECTION}
    rec = meta.get("dflash")
    recorded = isinstance(rec, dict) and rec.get("section") == SECTION    # a drafter held in `gguf` records no section
    if size is None:
        return [f"meta names a {SECTION} section the pack does not have"] if entries or recorded else []
    out = []
    if not entries:
        out.append(f"the {SECTION} section has no tensor in meta.tensors")
    if not recorded or rec.get("bytes") != size:
        out.append(f"meta.dflash does not record the {SECTION} section of {size} bytes")
    spans = []
    for name, e in entries.items():
        try:
            t = C.Tensor(name, e["type"], None, "id", e["shape"])
            off, n, want = int(e["offset"]), int(e["bytes"]), t.expected_bytes()
        except (KeyError, TypeError, ValueError):
            out.append(f"{name}: a malformed meta.tensors entry")
            continue
        if n != want or e.get("type_id") != t.type_id:
            out.append(f"{name}: {e['type']} {e['shape']} is recorded as type {e.get('type_id')} of {n} bytes")
        spans.append((off, n, name))
    end = 0
    for off, n, name in sorted(spans):
        if off % TENSOR_ALIGN or off < end or off + n > size:
            out.append(f"{name}: offset {off} is not {TENSOR_ALIGN} aligned, overlaps the tensor before it or runs past the section")
        end = max(end, off + n)
    return out
