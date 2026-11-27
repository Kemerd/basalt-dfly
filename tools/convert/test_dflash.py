"""Tests for dflash.py (the DFlash 2 block drafter's tensors and metadata) on a tiny synthetic DFlash2DraftModel: alone,
through convert.py --checkpoint QWEN35_DIR --drafter DIR, and attached to a small pack with --drafter DIR --into PACK
and rolled back; and, when BASALT_DFLASH2_DIR names the directory of a released drafter
(incoai/Qwen3.8-Flash-Next-DFlash2), on that checkpoint.

    python3 -m unittest discover -s tools/convert -p 'test_dflash.py'"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import convert  # noqa: E402
import dflash  # noqa: E402
import model_sections as S  # noqa: E402
from model_file import ModelFile, ModelWriter  # noqa: E402
from test_checkpoint import write_safetensors  # noqa: E402
from test_checkpoint_qwen35 import H, VOCAB, Fixture, q8_values  # noqa: E402

RELEASED = os.environ.get("BASALT_DFLASH2_DIR")
TARGETS = [0, 1]                       # the synthetic model has two layers
LAYERS, NHEAD, NKV, HD, FF = 2, 2, 1, 32, 96
GROUP, RANK = 16, 8
# the keys the engine reads under `<arch>.dflash.`
KEYS = {"block_count", "block_size", "mask_token_id", "target_layers", "conv_kernel", "conv_group", "selector_rank",
        "selector_top_k", "sliding_window", "attention.head_count", "attention.head_count_kv", "attention.key_length",
        "feed_forward_length", "rope.freq_base", "attention.layer_norm_rms_epsilon"}


def layer_shapes(h, nhead, nkv, hd, ff, group):
    """A drafter layer's checkpoint tensors (after `layers.N.`) -> shape."""
    return {"input_layernorm.weight": (h,), "post_attention_layernorm.weight": (h,),
            "self_attn.q_proj.weight": (nhead * hd, h), "self_attn.k_proj.weight": (nkv * hd, h),
            "self_attn.v_proj.weight": (nkv * hd, h), "self_attn.o_proj.weight": (h, nhead * hd),
            "self_attn.q_norm.weight": (hd,), "self_attn.k_norm.weight": (hd,),
            "mlp.gate_proj.weight": (ff, h), "mlp.up_proj.weight": (ff, h), "mlp.down_proj.weight": (h, ff),
            "attention_conv.base_kernel": (2, 2, h), "attention_conv.kernel_projection.weight": (2 * 2 * (h // group), h),
            "mlp_conv.base_kernel": (2, 2, h), "mlp_conv.kernel_projection.weight": (2 * 2 * (h // group), h)}


def write_drafter(d: Path, drop=(), extra=None, dtype="BF16", poison=None, short=None, **config) -> dict:
    """A DFlash2DraftModel directory for the synthetic model; returns its tensors' values by checkpoint name. `drop`
    leaves tensors out, `extra` adds or reshapes some, `poison` gets an infinite value, `short` loses its last value."""
    rng = np.random.default_rng(3)
    shapes = {"fc.weight": (H, H * len(TARGETS)), "hidden_norm.weight": (H,), "norm.weight": (H,)}
    for l in range(LAYERS):
        shapes.update({f"layers.{l}.{k}": s for k, s in layer_shapes(H, NHEAD, NKV, HD, FF, GROUP).items()})
    shapes.update({"candidate_selector.hidden_projection.weight": (RANK, H),
                   "candidate_selector.predecessor_codebook": (VOCAB, RANK),
                   "candidate_selector.successor_codebook": (VOCAB, RANK)})
    shapes.update(extra or {})
    v, tens = {}, {}
    for k, shape in shapes.items():
        if k in drop:
            continue
        v[k] = (rng.integers(-64, 64, size=shape) / 64).astype(np.float32)
        if k == poison:
            v[k].flat[0] = np.inf
        raw = (v[k].view(np.uint32) >> 16).astype("<u2").tobytes()
        tens[k] = (dtype, shape, raw[:-2] if k == short else raw)
    d.mkdir(exist_ok=True)
    write_safetensors(d / "model.safetensors", tens)
    cfg = {"architectures": ["DFlash2DraftModel"], "hidden_size": H, "vocab_size": VOCAB, "intermediate_size": FF,
           "num_hidden_layers": LAYERS, "num_attention_heads": NHEAD, "num_key_value_heads": NKV, "head_dim": HD,
           "is_causal": False, "layer_types": ["sliding_attention"] * LAYERS, "sliding_window": 2048, "rms_norm_eps": 1e-6,
           "rope_parameters": {"rope_theta": 10000, "rope_type": "default"},
           "dflash_config": {"block_size": 8, "conv_group_size": GROUP, "conv_kernel_size": 2, "mask_token_id": 9,
                             "selector_rank": RANK, "selector_top_k": 16, "target_layer_ids": TARGETS}}
    cfg.update(config)
    (d / "config.json").write_text(json.dumps(cfg))
    return v


def tensor_bytes(dr: dflash.Drafter, name: str, pool) -> bytes:
    """A pack tensor's bytes as the converter writes them."""
    if dr.tensors[name].type_name == "Q8_0":
        return b"".join(dr.chunks(name, pool))
    return dr.float_bytes(name)


class DrafterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name) / "drafter"
        self.pool = ThreadPool(2)
        self.addCleanup(self.pool.terminate)

    def drafter(self, arch="qwen35", hidden=H, vocab=VOCAB, layers=2):
        return dflash.Drafter(self.d, arch, hidden, vocab, layers)

    def test_names_types_and_shapes_in_checkpoint_order(self):
        write_drafter(self.d)
        T = self.drafter().tensors
        want = ["dflash.fc.0.weight", "dflash.fc.1.weight", "dflash.hidden_norm.weight", "dflash.norm.weight"]
        for l in range(LAYERS):
            want += [f"dflash.blk.{l}.{n}" for n in (
                "attn_norm.weight", "post_attention_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
                "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight", "ffn_gate.weight", "ffn_up.weight",
                "ffn_down.weight", "attn_conv.base", "attn_conv.weight", "ffn_conv.base", "ffn_conv.weight")]
        want += ["dflash.selector.hidden.weight", "dflash.selector.pred", "dflash.selector.succ"]
        self.assertEqual(list(T), want)
        conv = 2 * 2 * (H // GROUP)
        for name, ty, ne in (("dflash.fc.1.weight", "Q8_0", [H, H]), ("dflash.norm.weight", "F32", [H]),
                             ("dflash.blk.1.attn_q.weight", "Q8_0", [H, NHEAD * HD]),
                             ("dflash.blk.1.attn_k.weight", "Q8_0", [H, NKV * HD]),
                             ("dflash.blk.0.attn_output.weight", "Q8_0", [NHEAD * HD, H]),
                             ("dflash.blk.0.attn_q_norm.weight", "F32", [HD]),
                             ("dflash.blk.0.ffn_down.weight", "Q8_0", [FF, H]),
                             ("dflash.blk.1.attn_conv.base", "F32", [H, 2, 2]),
                             ("dflash.blk.1.ffn_conv.weight", "Q8_0", [H, conv]),
                             ("dflash.selector.hidden.weight", "BF16", [H, RANK]),
                             ("dflash.selector.succ", "BF16", [RANK, VOCAB])):
            self.assertEqual((T[name].type_name, T[name].ne), (ty, ne), name)
        self.assertEqual({t.type_name: t.type_id for t in T.values()}, {"Q8_0": 8, "F32": 0, "BF16": 30})

    def test_the_feature_projection_is_one_block_per_target_layer(self):
        v = write_drafter(self.d)
        dr = self.drafter()
        for j in range(len(TARGETS)):
            blob = tensor_bytes(dr, f"dflash.fc.{j}.weight", self.pool)
            x = v["fc.weight"][:, j * H:(j + 1) * H]
            self.assertEqual(blob, S.q8_0(np.ascontiguousarray(x)))
            got, step = q8_values(blob, H, H)
            self.assertTrue((np.abs(got - x) <= step + 1e-12).all())    # within one Q8_0 step

    def test_projections_norms_kernels_and_the_selector(self):
        v = write_drafter(self.d)
        dr = self.drafter()
        for src, name in (("layers.1.self_attn.o_proj.weight", "dflash.blk.1.attn_output.weight"),
                          ("layers.0.mlp_conv.kernel_projection.weight", "dflash.blk.0.ffn_conv.weight")):
            blob = tensor_bytes(dr, name, self.pool)
            self.assertEqual(len(blob), dr.tensors[name].expected_bytes())
            self.assertEqual(blob, S.q8_0(v[src]))
        # the norms are plain RMS weights: no offset is folded in
        for src, name in (("norm.weight", "dflash.norm.weight"), ("layers.0.self_attn.k_norm.weight", "dflash.blk.0.attn_k_norm.weight"),
                          ("layers.1.attention_conv.base_kernel", "dflash.blk.1.attn_conv.base")):
            self.assertEqual(tensor_bytes(dr, name, self.pool), v[src].astype("<f4").tobytes())
        for src, name in (("candidate_selector.predecessor_codebook", "dflash.selector.pred"),
                          ("candidate_selector.hidden_projection.weight", "dflash.selector.hidden.weight")):
            self.assertEqual(tensor_bytes(dr, name, self.pool), (v[src].view(np.uint32) >> 16).astype("<u2").tobytes())

    def test_the_metadata_takes_the_architecture_as_its_prefix(self):
        write_drafter(self.d)
        for arch in ("qwen35", "qwen4exp"):
            kv = self.drafter(arch).kv()
            self.assertEqual(set(kv), {f"{arch}.dflash.{k}" for k in KEYS})
            A = arch + ".dflash."
            self.assertEqual([kv[A + k] for k in ("block_count", "block_size", "mask_token_id", "target_layers", "conv_kernel",
                                                  "conv_group", "selector_rank", "selector_top_k", "sliding_window")],
                             [LAYERS, 8, 9, TARGETS, 2, GROUP, RANK, 16, 2048])
            self.assertEqual([kv[A + k] for k in ("attention.head_count", "attention.head_count_kv", "attention.key_length",
                                                  "feed_forward_length")], [NHEAD, NKV, HD, FF])
            self.assertEqual((kv[A + "rope.freq_base"], kv[A + "attention.layer_norm_rms_epsilon"]),
                             (10000.0, float(np.float32(1e-6))))

    def test_a_drafter_that_does_not_fit_is_refused(self):
        write_drafter(self.d)
        for kw in (dict(hidden=H + 32), dict(vocab=VOCAB + 1), dict(layers=1)):
            with self.assertRaises(S.ConvertError):
                self.drafter(**kw)
        for config in (dict(architectures=["DFlashDraftModel"]), dict(is_causal=True), dict(layer_types=["full_attention"] * LAYERS)):
            write_drafter(self.d, **config)
            with self.assertRaises(S.ConvertError):
                self.drafter()
        for kw in (dict(extra={"layers.0.self_attn.bias": (4,)}), dict(dtype="F16"), dict(extra={"fc.weight": (H, H)})):
            write_drafter(self.d, **kw)
            with self.assertRaises(S.ConvertError):
                self.drafter()

    def test_an_incomplete_or_damaged_drafter_is_refused(self):
        block = {"conv_group_size": GROUP, "conv_kernel_size": 2, "mask_token_id": 9, "selector_rank": RANK,
                 "selector_top_k": 16, "target_layer_ids": TARGETS}          # no block_size
        for kw in (dict(drop=("layers.1.mlp.up_proj.weight",)), dict(drop=("norm.weight",)),
                   dict(extra={"layers.2.input_layernorm.weight": (H,)}),     # a layer past num_hidden_layers
                   dict(extra={"layers.0.mlp.down_proj.weight": (H, 40)}),    # rows of 40 values are no Q8_0 blocks
                   dict(short="hidden_norm.weight"), dict(sliding_window=None), dict(dflash_config=block),
                   dict(dflash_config=dict(block, block_size=8, target_layer_ids=[]))):
            write_drafter(self.d, **kw)
            with self.assertRaises(S.ConvertError, msg=str(kw)):
                self.drafter()
        write_drafter(self.d)
        (self.d / "config.json").write_text("{")
        with self.assertRaises(S.ConvertError):
            self.drafter()
        (self.d / "config.json").unlink()
        with self.assertRaises(S.ConvertError):
            self.drafter()
        write_drafter(self.d)
        (self.d / "model.safetensors").unlink()
        with self.assertRaises(S.ConvertError):
            self.drafter()
        # a non-finite value stops the tensor that holds it, whatever its type
        for src, name in (("norm.weight", "dflash.norm.weight"), ("candidate_selector.successor_codebook", "dflash.selector.succ"),
                          ("layers.0.self_attn.q_proj.weight", "dflash.blk.0.attn_q.weight")):
            write_drafter(self.d, poison=src)
            with self.assertRaises(S.ConvertError):
                tensor_bytes(self.drafter(), name, self.pool)

    def test_the_section_holds_every_tensor_aligned_and_is_checked_against_meta(self):
        write_drafter(self.d)
        dr = self.drafter("qwen4exp")
        blob = Path(self.tmp.name) / "section"
        entries = dr.write(blob, self.pool)
        raw = blob.read_bytes()
        self.assertEqual(list(entries), list(dr.tensors))
        end = 0
        for name, e in entries.items():
            t = dr.tensors[name]
            self.assertEqual((e["type"], e["type_id"], e["shape"], e["bytes"], e["section"]),
                             (t.type_name, t.type_id, t.ne, t.expected_bytes(), "dflash"))
            self.assertEqual((e["offset"] % 4096, e["offset"] - end < 4096, raw[end:e["offset"]].strip(b"\0")), (0, True, b""))
            self.assertEqual(raw[e["offset"]:e["offset"] + e["bytes"]], tensor_bytes(dr, name, self.pool), name)
            end = e["offset"] + e["bytes"]
        self.assertEqual(end, len(raw))
        meta = {"tensors": dict(entries, **{"output.weight": {"offset": 0, "bytes": 4}}),
                "dflash": {"section": "dflash", "bytes": len(raw)}}
        self.assertEqual(dflash.section_problems(meta, len(raw)), [])
        # a pack without the section has nothing to check, a drafter held in `gguf` included
        self.assertEqual(dflash.section_problems({"tensors": {"output.weight": {"offset": 0, "bytes": 4}}}, None), [])
        self.assertEqual(dflash.section_problems({"tensors": {}, "dflash": dr.record(lambda p: p.name)}, None), [])
        first, second = list(entries)[:2]
        for change in ({first: dict(entries[first], offset=1)}, {second: dict(entries[second], offset=0)},
                       {first: dict(entries[first], bytes=entries[first]["bytes"] - 2)},
                       {first: dict(entries[first], type_id=1)}, {first: {"section": "dflash"}}):
            bad = dict(meta, tensors=dict(meta["tensors"], **change))
            self.assertTrue(dflash.section_problems(bad, len(raw)), str(change))
        self.assertTrue(dflash.section_problems(meta, len(raw) - 1))                      # the last tensor runs past it
        self.assertTrue(dflash.section_problems(dict(meta, dflash=None), len(raw)))      # no record
        self.assertTrue(dflash.section_problems({"tensors": {}, "dflash": meta["dflash"]}, len(raw)))
        self.assertTrue(dflash.section_problems(meta, None))                              # the section is gone


class Qwen35DrafterTest(unittest.TestCase):
    """convert.py --checkpoint DIR --drafter DIR: the drafter's tensors follow the model's in `gguf`."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        t = Path(cls.tmp.name)
        (t / "ck").mkdir()
        Fixture(t / "ck")
        write_drafter(t / "drafter")
        out = t / "x.basalt"
        r = subprocess.run([sys.executable, str(HERE / "convert.py"), "--checkpoint", str(t / "ck"), "--drafter",
                            str(t / "drafter"), "--out", str(out), "--workers", "2"], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stdout + r.stderr)
        cls.t, cls.mf = t, ModelFile(str(out))
        cls.meta = json.loads(cls.mf.read("meta"))

    def test_the_tensors_the_metadata_and_the_record(self):
        dr = dflash.Drafter(self.t / "drafter", "qwen35", H, VOCAB, 2)
        T, gguf = self.meta["tensors"], self.mf.read("gguf")
        self.assertEqual(list(T)[-len(dr.tensors):], list(dr.tensors))
        with ThreadPool(2) as pool:
            for name, t in dr.tensors.items():
                e = T[name]
                self.assertEqual((e["type"], e["type_id"], e["shape"], e["bytes"]), (t.type_name, t.type_id, t.ne, t.expected_bytes()))
                self.assertEqual(e["offset"] % 4096, 0)
                self.assertNotIn("section", e)
                self.assertEqual(gguf[e["offset"]:e["offset"] + e["bytes"]], tensor_bytes(dr, name, pool), name)
        kv = self.meta["gguf_kv"]
        self.assertEqual({k: v for k, v in kv.items() if ".dflash." in k}, dr.kv())
        rec = self.meta["dflash"]
        self.assertEqual((rec["source"], [f["path"] for f in rec["files"]], rec["form"]), ("drafter", ["model.safetensors"], dflash.FORM))
        self.assertEqual(rec["files"][0]["size"], (self.t / "drafter" / "model.safetensors").stat().st_size)


def write_pack(path: Path, arch="qwen4exp", vocab=VOCAB, head="output.weight") -> None:
    """A small pack with what the attach reads (the architecture, the model's width and layers, the head's shape),
    written by the real writer."""
    w = ModelWriter(str(path))
    w.add_bytes("dense", bytes(range(200)))
    w.add_bytes("gguf", np.arange(H * vocab, dtype="<f4").tobytes())
    meta = {"format": "basalt", "version": 2, "name": "Tiny", "sources": [],
            "gguf_kv": {"general.architecture": arch, arch + ".embedding_length": H, arch + ".block_count": 2},
            "tensors": {head: {"type": "F32", "type_id": 0, "shape": [H, vocab], "bytes": 4 * H * vocab, "offset": 0}}}
    w.add_bytes("meta", json.dumps(meta).encode())
    w.close()


class AttachTest(unittest.TestCase):
    """convert.py --drafter DIR --into PACK --undo-dir DIR, and --roll-back."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.t = Path(self.tmp.name)
        self.pack, self.d, self.undo = self.t / "p.basalt", self.t / "drafter", self.t / "undo"
        write_pack(self.pack)
        write_drafter(self.d)

    def cli(self, *args):
        return subprocess.run([sys.executable, str(HERE / "convert.py"), *map(str, args)], capture_output=True, text=True)

    def refused(self, **kw):
        """An attach in this process stops, and the pack and the undo directory are as they were."""
        a = dict(dict(into=str(self.pack), drafter=str(self.d), undo_dir=str(self.undo), workers=1), **kw)
        pack = Path(a["into"])
        before, had_undo = pack.read_bytes() if pack.is_file() else None, self.undo.exists()
        with self.assertRaises(SystemExit) as stop:
            convert.add_drafter(type("A", (), a)())
        self.assertEqual((before is None or pack.read_bytes() == before, self.undo.exists()), (True, had_undo), stop.exception)
        self.assertEqual([p.name for p in self.t.iterdir() if p.name.startswith(".dflash-")], [])
        return str(stop.exception)

    def test_the_section_joins_the_pack_and_roll_back_restores_it_bit_for_bit(self):
        original, old = self.pack.read_bytes(), ModelFile(str(self.pack))
        was, kept = json.loads(old.read("meta")), {n: old.read(n) for n in ("dense", "gguf")}
        r = self.cli("--drafter", self.d, "--into", self.pack, "--undo-dir", self.undo, "--workers", 2)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        m = ModelFile(str(self.pack))
        self.assertEqual(m.names(), ["dense", "gguf", "dflash", "meta"])
        for n in m.names():
            self.assertTrue(m.verify(n), n)
        for n in ("dense", "gguf"):    # where and as they were
            self.assertEqual((m.sections[n], m.read(n)), (old.sections[n], kept[n]))
        self.assertEqual(m.section("dflash")[0] % (2 << 20), 0)
        self.assertEqual(self.pack.read_bytes()[slice(*np.cumsum(old.section("meta")))].strip(b"\0"), b"")   # the old meta is a gap
        meta = json.loads(m.read("meta"))
        dr = dflash.Drafter(self.d, "qwen4exp", H, VOCAB, 2)
        sec, T = m.read("dflash"), meta["tensors"]
        self.assertEqual(list(T), ["output.weight"] + list(dr.tensors))
        self.assertEqual(T["output.weight"], was["tensors"]["output.weight"])
        with ThreadPool(2) as pool:
            for name, t in dr.tensors.items():
                e = T[name]
                self.assertEqual((e["type_id"], e["shape"], e["section"], e["offset"] % 4096), (t.type_id, t.ne, "dflash", 0))
                self.assertEqual(sec[e["offset"]:e["offset"] + e["bytes"]], tensor_bytes(dr, name, pool), name)
        self.assertEqual(dflash.section_problems(meta, len(sec)), [])
        self.assertEqual(meta["gguf_kv"], dict(was["gguf_kv"], **dr.kv()))
        self.assertEqual(meta["gguf_kv"]["qwen4exp.dflash.target_layers"], TARGETS)
        rec = meta["dflash"]
        self.assertEqual((rec["section"], rec["bytes"], rec["source"], rec["form"]), ("dflash", len(sec), "drafter", dflash.FORM))
        self.assertEqual(rec["files"], [convert.source_record(self.d / "model.safetensors")])
        self.assertEqual({k: meta[k] for k in was if k not in ("gguf_kv", "tensors")}, {k: was[k] for k in was if k not in ("gguf_kv", "tensors")})
        self.assertEqual([p.name for p in self.t.iterdir() if p.name.startswith(".dflash-")], [])
        # a second drafter is refused; the pack keeps the first
        attached = self.pack.read_bytes()
        r = self.cli("--drafter", self.d, "--into", self.pack, "--undo-dir", self.t / "undo2", "--workers", 2)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("has a block drafter already", r.stderr)
        self.assertEqual((self.pack.read_bytes() == attached, (self.t / "undo2").exists()), (True, False))
        r = self.cli("--roll-back", "--into", self.pack, "--undo-dir", self.undo)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.pack.read_bytes(), original)

    def test_what_does_not_fit_leaves_the_pack_as_it_was(self):
        self.assertIn("has a block drafter already", self.refused(into=str(self.packed("tensors", {"dflash.norm.weight": {}}))))
        write_pack(self.pack, arch="qwen35")
        self.assertIn("takes no --drafter", self.refused())
        write_pack(self.pack, arch="other")        # an architecture without rules of its own is not taken for Flash-Next
        self.assertIn("takes no --drafter", self.refused())
        write_pack(self.pack, vocab=VOCAB + 32)
        self.assertIn("vocabulary", self.refused())
        write_pack(self.pack, head="token_embd.weight")
        self.assertIn("output head", self.refused())
        write_pack(self.pack)
        self.assertIn("missing input", self.refused(drafter=str(self.t / "none")))
        self.assertIn("missing input", self.refused(into=str(self.t / "none.basalt")))
        for n in (70000, 10):                      # not a pack, and too short to hold a header
            (self.t / "not.basalt").write_bytes(b"x" * n)
            self.refused(into=str(self.t / "not.basalt"))
        write_drafter(self.d, drop=("layers.0.mlp.gate_proj.weight",))
        self.assertIn("missing", self.refused())
        write_drafter(self.d, poison="layers.1.self_attn.v_proj.weight")      # found while the section is written
        self.assertIn("non-finite", self.refused(workers=2))
        write_drafter(self.d)
        self.undo.mkdir()
        self.assertIn("--undo-dir", self.refused())

    def packed(self, key, value) -> Path:
        """The pack with `value` added under meta[key], as another file."""
        m = ModelFile(str(self.pack))
        meta = json.loads(m.read("meta"))
        meta[key].update(value)
        w = ModelWriter(str(self.t / "other.basalt"))
        for n in ("dense", "gguf"):
            w.add_bytes(n, m.read(n))
        w.add_bytes("meta", json.dumps(meta).encode())
        w.close()
        return self.t / "other.basalt"

    def test_the_drafter_needs_a_pack_or_a_checkpoint(self):
        r = self.cli("--drafter", self.d, "--gguf", self.t / "m.gguf", "--out", self.t / "out.basalt")
        self.assertEqual(r.returncode, 2)
        self.assertIn("--drafter takes --into", r.stderr)
        before = self.pack.read_bytes()
        r = self.cli("--drafter", self.d, "--into", self.pack)
        self.assertEqual((r.returncode, self.pack.read_bytes() == before), (2, True))


@unittest.skipUnless(RELEASED, "BASALT_DFLASH2_DIR names no released drafter")
class ReleasedDrafterTest(unittest.TestCase):
    """incoai/Qwen3.8-Flash-Next-DFlash2: 81 BF16 tensors for a model of 2560 channels, 248320 tokens and 48 layers."""

    @classmethod
    def setUpClass(cls):
        cls.dr = dflash.Drafter(RELEASED, "qwen4exp", 2560, 248320, 48)

    def test_every_tensor_has_its_name_type_and_shape(self):
        T = self.dr.tensors
        self.assertEqual((len(self.dr.t), len(T)), (81, 85))
        want = {f"dflash.fc.{j}.weight": ("Q8_0", [2560, 2560]) for j in range(5)}
        want.update({"dflash.hidden_norm.weight": ("F32", [2560]), "dflash.norm.weight": ("F32", [2560]),
                     "dflash.selector.hidden.weight": ("BF16", [2560, 256]),
                     "dflash.selector.pred": ("BF16", [256, 248320]), "dflash.selector.succ": ("BF16", [256, 248320])})
        for l in range(5):
            P = f"dflash.blk.{l}."
            want.update({P + "attn_norm.weight": ("F32", [2560]), P + "post_attention_norm.weight": ("F32", [2560]),
                         P + "attn_q_norm.weight": ("F32", [128]), P + "attn_k_norm.weight": ("F32", [128]),
                         P + "attn_q.weight": ("Q8_0", [2560, 4096]), P + "attn_k.weight": ("Q8_0", [2560, 1024]),
                         P + "attn_v.weight": ("Q8_0", [2560, 1024]), P + "attn_output.weight": ("Q8_0", [4096, 2560]),
                         P + "ffn_gate.weight": ("Q8_0", [2560, 7680]), P + "ffn_up.weight": ("Q8_0", [2560, 7680]),
                         P + "ffn_down.weight": ("Q8_0", [7680, 2560]),
                         P + "attn_conv.base": ("F32", [2560, 2, 2]), P + "attn_conv.weight": ("Q8_0", [2560, 640]),
                         P + "ffn_conv.base": ("F32", [2560, 2, 2]), P + "ffn_conv.weight": ("Q8_0", [2560, 640])})
        self.assertEqual({n: (t.type_name, t.ne) for n, t in T.items()}, want)
        size = {ty: sum(t.expected_bytes() for t in T.values() if t.type_name == ty) for ty in ("Q8_0", "BF16", "F32")}
        self.assertEqual(size, {"Q8_0": 504832000, "BF16": 255590400, "F32": 537600})

    def test_the_metadata(self):
        kv, A = self.dr.kv(), "qwen4exp.dflash."
        self.assertEqual(set(kv), {A + k for k in KEYS})
        self.assertEqual([kv[A + k] for k in ("block_count", "block_size", "mask_token_id", "target_layers", "conv_kernel",
                                              "conv_group", "selector_rank", "selector_top_k", "sliding_window")],
                         [5, 8, 248077, [5, 15, 25, 35, 45], 2, 16, 256, 16, 2048])
        self.assertEqual([kv[A + k] for k in ("attention.head_count", "attention.head_count_kv", "attention.key_length",
                                              "feed_forward_length", "rope.freq_base")], [32, 8, 128, 7680, 10000.0])

    def test_a_feature_block_and_a_layer_matrix_against_the_checkpoint(self):
        dr = self.dr
        f, off, _, shape, n = dr.t["fc.weight"]
        fc = S._bf16_to_f32(np.fromfile(f, dtype="<u2", count=n // 2, offset=off)).reshape(shape)
        with ThreadPool(4) as pool:
            for j in (0, 3):
                blob = tensor_bytes(dr, f"dflash.fc.{j}.weight", pool)
                x = np.ascontiguousarray(fc[:, j * 2560:(j + 1) * 2560])
                self.assertEqual(blob, S.q8_0(x))
                got, step = q8_values(blob, 2560, 2560)
                self.assertTrue((np.abs(got - x) <= step + 1e-12).all())    # within one Q8_0 step
            blob = tensor_bytes(dr, "dflash.blk.4.attn_conv.weight", pool)
            self.assertEqual(blob, S.q8_0(dr.values("dflash.blk.4.attn_conv.weight")))
        self.assertEqual(dr.float_bytes("dflash.blk.2.ffn_conv.base"),
                         dr.values("dflash.blk.2.ffn_conv.base").astype("<f4").tobytes())

    def test_the_section(self):
        with tempfile.TemporaryDirectory() as tmp, ThreadPool(8) as pool:
            blob = Path(tmp) / "section"
            entries = self.dr.write(blob, pool)
            size = blob.stat().st_size
            self.assertEqual((len(entries), size, sum(e["bytes"] for e in entries.values())), (85, 761018368, 760960000))
            self.assertEqual(dflash.section_problems({"tensors": entries, "dflash": {"section": "dflash", "bytes": size}}, size), [])
            with open(blob, "rb") as f:
                for name in ("dflash.fc.4.weight", "dflash.blk.0.attn_k_norm.weight", "dflash.selector.hidden.weight"):
                    f.seek(entries[name]["offset"])
                    self.assertEqual(f.read(entries[name]["bytes"]), tensor_bytes(self.dr, name, pool), name)


if __name__ == "__main__":
    unittest.main()
