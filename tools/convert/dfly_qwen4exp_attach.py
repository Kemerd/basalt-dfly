"""--dfly FILE --into PACK.basalt on a Flash-Next pack: a trained DFly2 drafter joins it as the `dfly` section.

FILE is a drafter checkpoint as tools/dfly2-train writes it (include/basalt/artifact/dfly2_checkpoint.hpp): one safetensors
file whose `__metadata__` states the drafter's shape and the pack it was trained on, and whose tensors are FP32 masters.
Each is rounded to BF16 (nearest even, the type the trained forward saw) and stored 64-byte aligned; meta.dfly2 records
where each lies (offsets from the start of the section), the settings every DFly2 drafter has and this drafter's own
shape. The target's embedding and output head are not in the file: the drafter uses the pack's.
Like --dspark: no other section moves, and --roll-back restores the pack bit for bit.
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np

import model_sections as S
from dfly_attach import read_header

SECTION = "dfly"
FORMAT = "dfly2-qwen4exp"                  # meta.dfly2.format: the engine's Flash-Next drafter reads this record
CHECKPOINT_FORMAT, CHECKPOINT_VERSION = "basalt-dfly2-drafter", "1"
ALIGN = 64
# what the engine's kernels fix for the target and its drafter (kernels/qsa.hpp, decode/layout.hpp, decode/verify.hpp,
# dfly2/record.hpp): a pack states its width and head counts, these it does not
QSA_INTERVAL, HEAD_DIM, IDX_TOP_K, IDX_BLOCK = 4, 256, 2048, 4
MAX_LAYERS, MAX_BLOCK, Q8_BLOCK = 8, 8, 32
ROPE_BASE = 1e7                            # a pack without <arch>.rope.freq_base turns by the model's own base
INTS = {"hidden": "dfly2.hidden_size", "intermediate": "dfly2.intermediate_size",
        "correction": "dfly2.correction_intermediate_size", "layers": "dfly2.num_layers", "heads": "dfly2.num_heads",
        "kv_heads": "dfly2.num_kv_heads", "head_dim": "dfly2.head_dim", "rope_dims": "dfly2.rope_dims",
        "vocab": "dfly2.vocab_size", "block": "dfly2.block_size", "mask": "dfly2.mask_token_id",
        "sel": "dfly2.selection_width", "step": "dfly2.step", "pack_bytes": "dfly2.pack_bytes"}
REALS = {"rope_base": "dfly2.rope_base", "attn_eps": "dfly2.attn_norm_eps", "rms_eps": "dfly2.rms_norm_eps"}


def die(msg: str):
    raise SystemExit(f"convert: {msg}")


def read_metadata(path: Path) -> dict:
    """The safetensors header's `__metadata__`: the checkpoint's keys, every value a string."""
    with path.open("rb") as f:
        n = int.from_bytes(f.read(8), "little")
        md = json.loads(f.read(n)).get("__metadata__")
    if not isinstance(md, dict) or any(not isinstance(v, str) for v in md.values()):
        die(f"{path}: no drafter checkpoint, its header has no metadata")
    return md


def shape_of(path: Path, md: dict) -> dict:
    """The drafter's shape and settings from the checkpoint's metadata, each key present and well formed."""
    if md.get("format") != CHECKPOINT_FORMAT:
        die(f"{path}: checkpoint format is {md.get('format')!r}, not {CHECKPOINT_FORMAT}")
    if md.get("version") != CHECKPOINT_VERSION:
        die(f"{path}: checkpoint version {md.get('version')}, this converter reads {CHECKPOINT_VERSION}")
    s = {}
    for kind, keys in ((int, INTS), (float, REALS)):
        for field, key in keys.items():
            try:
                s[field] = kind(md[key])
            except KeyError:
                die(f"{path}: checkpoint metadata has no {key}")
            except ValueError:
                die(f"{path}: checkpoint metadata {key} is not a number: {md[key]}")
    for key in ("dfly2.target_layers", "dfly2.weight_dtype", "dfly2.pack_name", "dfly2.pack_meta_sha256"):
        if key not in md:
            die(f"{path}: checkpoint metadata has no {key}")
    try:
        s["target_layers"] = [int(x) for x in md["dfly2.target_layers"].split(",")]
    except ValueError:
        die(f"{path}: checkpoint metadata dfly2.target_layers is malformed: {md['dfly2.target_layers']}")
    if not all(math.isfinite(s[k]) for k in REALS):
        die(f"{path}: checkpoint metadata has a rope base or norm eps that is not finite")
    s["weight_dtype"] = md["dfly2.weight_dtype"]
    s["pack_name"], s["pack_meta_sha256"] = md["dfly2.pack_name"], md["dfly2.pack_meta_sha256"]
    return s


def drafter_tensors(s: dict) -> dict[str, list[int]]:
    """Every tensor of a drafter of shape `s` and its shape: drafter_tensors() of the checkpoint contract."""
    d, i, c, dh = s["hidden"], s["intermediate"], s["correction"], s["head_dim"]
    t = {}
    for layer in range(s["layers"]):
        p = f"layers.{layer}."
        t[p + "input_layernorm.weight"] = [d]
        t[p + "self_attn.q_proj.weight"] = [2 * s["heads"] * dh, d]
        t[p + "self_attn.k_proj.weight"] = [s["kv_heads"] * dh, d]
        t[p + "self_attn.v_proj.weight"] = [s["kv_heads"] * dh, d]
        t[p + "self_attn.o_proj.weight"] = [d, s["heads"] * dh]
        t[p + "self_attn.q_norm.weight"] = [dh]
        t[p + "self_attn.k_norm.weight"] = [dh]
        t[p + "post_attention_layernorm.weight"] = [d]
        t[p + "mlp.gate_proj.weight"] = [i, d]
        t[p + "mlp.up_proj.weight"] = [i, d]
        t[p + "mlp.down_proj.weight"] = [d, i]
    t.update({
        "final_norm.weight": [d], "mtp_fusion.enorm.weight": [d], "mtp_fusion.hnorm.weight": [d],
        "mtp_fusion.eh_proj.weight": [d, 2 * d], "mtp_fusion.gate": [d],
        "hidden_correction.hidden_norm.weight": [d], "hidden_correction.embed_norm.weight": [d],
        "hidden_correction.gate_proj.weight": [c, 2 * d], "hidden_correction.up_proj.weight": [c, 2 * d],
        "hidden_correction.down_proj.weight": [d, c],
        "confidence_head.proj.weight": [1, d], "confidence_head.proj.bias": [1]})
    return t


def pack_geometry(pack: Path, meta: dict) -> dict:
    """What the pack says of the target: its width, layer and head counts, vocabulary and rotary base."""
    try:
        kv = meta["gguf_kv"]
        name = kv["general.architecture"]
        vocab = meta["tensors"]["output.weight"]["shape"][-1] if "output.weight" in meta.get("tensors", {}) \
            else len(kv["tokenizer.ggml.tokens"])
        return {"hidden": int(kv[name + ".embedding_length"]), "layers": int(kv[name + ".block_count"]),
                "heads": int(kv[name + ".attention.head_count"]), "kv_heads": int(kv[name + ".attention.head_count_kv"]),
                "vocab": int(vocab), "rope_base": float(kv.get(name + ".rope.freq_base") or ROPE_BASE)}
    except (KeyError, TypeError, ValueError, IndexError):
        die(f"{pack}: its meta lacks the model's width, layer count, head counts or vocabulary")


def check(path: Path, s: dict, g: dict, pack: Path, pack_bytes: int, pack_digest: str, force: bool) -> None:
    """Stops on a drafter the engine would refuse or that was trained for another target."""
    if not 1 <= s["layers"] <= MAX_LAYERS or len(s["target_layers"]) != s["layers"]:
        die(f"{path}: {s['layers']} draft layers and {len(s['target_layers'])} target layers; the engine reads 1 to {MAX_LAYERS}, one target layer each")
    if min(s["hidden"], s["intermediate"], s["correction"], s["heads"], s["kv_heads"], s["head_dim"]) <= 0:
        die(f"{path}: a drafter dimension is not positive")
    for what, have, want in (("hidden_size", s["hidden"], g["hidden"]), ("num_heads", s["heads"], g["heads"]),
                             ("num_kv_heads", s["kv_heads"], g["kv_heads"]), ("head_dim", s["head_dim"], HEAD_DIM),
                             ("vocab_size", s["vocab"], g["vocab"])):
        if have != want:
            die(f"{path}: {what} is {have}, the pack's is {want}")
    last = -1
    for layer in s["target_layers"]:
        # the drafter reads a full-attention layer's cache and selection: every QSA_INTERVAL-th layer
        if layer <= last or layer >= g["layers"] or layer % QSA_INTERVAL != QSA_INTERVAL - 1:
            die(f"{path}: target layers {s['target_layers']} are not ascending full-attention layers of the pack's {g['layers']}")
        last = layer
    if s["sel"] != IDX_TOP_K + IDX_BLOCK - 1:
        die(f"{path}: selection_width is {s['sel']}, the target selects {IDX_TOP_K + IDX_BLOCK - 1} cells")
    if not 2 <= s["block"] <= MAX_BLOCK:
        die(f"{path}: block_size is {s['block']}; the engine verifies 2 to {MAX_BLOCK} rows")
    if not 0 <= s["mask"] < g["vocab"]:
        die(f"{path}: mask token {s['mask']} is outside the pack's vocabulary of {g['vocab']}")
    if s["rope_dims"] <= 0 or s["rope_dims"] % 2 or s["rope_dims"] > s["head_dim"]:
        die(f"{path}: rope_dims {s['rope_dims']} does not fit a head of {s['head_dim']}")
    if s["attn_eps"] <= 0 or s["rms_eps"] <= 0 or s["step"] < 0:
        die(f"{path}: a norm eps or the step count is out of range")
    # the engine reads every projection as Q8_0: whole blocks of input values
    for what, width in (("hidden_size", s["hidden"]), ("intermediate_size", s["intermediate"]),
                        ("correction_intermediate_size", s["correction"]), ("the heads' width", s["heads"] * s["head_dim"])):
        if width % Q8_BLOCK:
            die(f"{path}: {what} {width} is not a multiple of {Q8_BLOCK}, which the engine's projections need")
    if s["weight_dtype"] not in ("BF16", "F32"):
        die(f"{path}: weight_dtype is {s['weight_dtype']!r}, not BF16 or F32")
    stops = []
    if s["rope_base"] != g["rope_base"]:
        stops.append(f"it turns by a rotary base of {s['rope_base']:g}, the pack by {g['rope_base']:g}")
    if s["pack_bytes"] != pack_bytes or s["pack_meta_sha256"] != pack_digest:
        stops.append(f"it was trained on {s['pack_name']} ({s['pack_bytes']} bytes, meta {s['pack_meta_sha256'][:12]}), "
                     f"not this pack ({pack_bytes} bytes, meta {pack_digest[:12]})")
    for stop in stops:
        if not force:
            die(f"{path}: {stop}; --force attaches it anyway")
        print(f"convert: warning: {path}: {stop} (--force)")


def write_section(path: Path, want: dict[str, list[int]], out: Path) -> dict:
    """The section image at `out`: each tensor rounded to BF16, 64-byte aligned. Returns the record's tensor table."""
    head, base = read_header(path)
    if extra := sorted(head.keys() - want.keys()):
        die(f"{path} has tensors a drafter of its shape does not: {', '.join(extra[:6])}{' ...' if len(extra) > 6 else ''}")
    if missing := sorted(want.keys() - head.keys()):
        die(f"{path} lacks drafter tensors: {', '.join(missing[:6])}{' ...' if len(missing) > 6 else ''}")
    for name, shape in want.items():
        t = head[name]
        lo, hi = t["data_offsets"]
        if t["dtype"] != "F32":
            die(f"{path}: {name} is {t['dtype']}; a checkpoint holds FP32 masters")
        if t["shape"] != shape or hi - lo != math.prod(shape) * 4:
            die(f"{path}: {name} is {t['shape']} in {hi - lo} bytes, a drafter of this shape has {shape}")
    table, at = {}, 0
    with path.open("rb") as src, out.open("wb") as f:
        for name, shape in want.items():
            lo, hi = head[name]["data_offsets"]
            f.write(b"\0" * (-at % ALIGN))
            at += -at % ALIGN
            src.seek(base + lo)
            try:
                raw = S.bf16_bytes(np.frombuffer(src.read(hi - lo), dtype=np.uint8), "F32")
            except ValueError:
                die(f"{path}: {name} holds a value that is not finite")
            f.write(raw)
            table[name] = {"dtype": "BF16", "shape": shape, "offset": at, "bytes": len(raw)}
            at += len(raw)
    return table


def record(s: dict, tensors: dict, size: int) -> dict:
    """meta.dfly2: the settings every DFly2 drafter has (dfly2::parse_record), then this drafter's own."""
    return {
        "section": SECTION, "format": FORMAT, "bytes": size, "layers": s["layers"], "mask_token_id": s["mask"],
        "target_kv_layers": s["target_layers"], "target_index_layers": s["target_layers"],
        "target_index_topk": IDX_TOP_K, "block_size": s["block"],
        "hidden_size": s["hidden"], "intermediate_size": s["intermediate"],
        "correction_intermediate_size": s["correction"], "num_heads": s["heads"], "num_kv_heads": s["kv_heads"],
        "head_dim": s["head_dim"], "rope_dims": s["rope_dims"], "rope_base": s["rope_base"],
        "attn_norm_eps": s["attn_eps"], "rms_norm_eps": s["rms_eps"], "selection_width": s["sel"],
        "causal_block": True, "mtp_fusion": "slot0", "step": s["step"],
        "source": {"pack_name": s["pack_name"], "pack_bytes": s["pack_bytes"], "pack_meta_sha256": s["pack_meta_sha256"]},
        "tensors": tensors,
    }


def section_problems(meta: dict, section_bytes: int | None) -> list[str]:
    """What is wrong with a pack's Flash-Next drafter record against its `dfly` section (None: the pack has none)."""
    rec = meta.get("dfly2")
    if not isinstance(rec, dict) or rec.get("format") != FORMAT:
        return []
    if section_bytes is None:
        return ["meta.dfly2 names a drafter, but the pack has no dfly section"]
    out = []
    if rec.get("bytes") != section_bytes:
        out.append(f"meta.dfly2 says {rec.get('bytes')} bytes, section {SECTION} is {section_bytes}")
    spans = []
    for name, t in rec.get("tensors", {}).items():
        off, n = t.get("offset", -1), t.get("bytes", -1)
        if t.get("dtype") != "BF16" or n != math.prod(t.get("shape", [0])) * 2:
            out.append(f"dfly {name}: {t.get('dtype')} {t.get('shape')} does not fill its {n} bytes")
        if off < 0 or off % ALIGN or off + n > section_bytes:
            out.append(f"dfly {name}: ({off}+{n}) is not {ALIGN}-byte aligned inside the section ({section_bytes} bytes)")
        spans.append((off, off + n, name))
    spans.sort()
    for (_, end, a), (start, _, b) in zip(spans, spans[1:]):
        if start < end:
            out.append(f"dfly {a} and {b} overlap")
    return out


def add_dfly(a, source_record=None) -> None:
    """--dfly with --into on a Flash-Next pack: the drafter joins it, whose other sections stay where and as they are."""
    from model_file import ModelFile, append_section
    pack, src = Path(a.into), Path(a.dfly)
    for p in (pack, src):
        if not p.is_file():
            die(f"missing input: {p}")
    m = ModelFile(str(pack))
    meta = json.loads(m.read("meta"))
    if SECTION in m.names() or meta.get("dfly2"):
        die(f"{pack} has a drafter already")
    g = pack_geometry(pack, meta)
    read_header(src)                       # a file that is no safetensors file stops here
    s = shape_of(src, read_metadata(src))
    check(src, s, g, pack, pack.stat().st_size, m.sections["meta"][2].hex(), bool(getattr(a, "force", False)))
    if s["weight_dtype"] != "BF16":
        print(f"convert: {src}: trained with {s['weight_dtype']} weights; the pack stores them rounded to BF16")
    # the section is written beside the pack first: append_section copies a file, and a refusal leaves the pack alone
    with tempfile.TemporaryDirectory(prefix=".dfly-", dir=str(pack.parent)) as tmp:
        blob = Path(tmp) / SECTION
        meta["dfly2"] = record(s, write_section(src, drafter_tensors(s), blob), 0)
        meta["dfly2"]["bytes"] = blob.stat().st_size
        if source_record is not None:
            meta.setdefault("sources", []).append(source_record(src))
        try:
            append_section(str(pack), SECTION, str(blob), json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), a.undo_dir)
        except (ValueError, FileExistsError) as e:
            die(str(e))
    m = ModelFile(str(pack))
    if not m.verify(SECTION) or not m.verify("meta"):
        die(f"{pack}: the new sections do not read back as written; --roll-back puts the file back")
    print(f"done: {pack} is {pack.stat().st_size} bytes, {len(m.names())} sections")
