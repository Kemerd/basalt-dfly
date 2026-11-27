"""--dfly --into on a Flash-Next pack: a synthetic drafter checkpoint joins a synthetic pack; the record, the BF16
rounding, the offsets, roll-back, and each refusal."""
import contextlib
import io
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import convert  # noqa: E402
import dfly_qwen4exp_attach as attach  # noqa: E402
import ggufq  # noqa: E402
from model_file import ModelFile, ModelWriter  # noqa: E402

HIDDEN, HEADS, KV_HEADS, VOCAB, PACK_LAYERS = 32, 2, 1, 40, 8
SHAPE = {"hidden": HIDDEN, "intermediate": 64, "correction": 32, "layers": 2, "heads": HEADS, "kv_heads": KV_HEADS,
         "head_dim": attach.HEAD_DIM, "rope_dims": 64, "vocab": VOCAB, "block": 8, "mask": 39,
         "sel": attach.IDX_TOP_K + attach.IDX_BLOCK - 1, "step": 0, "target_layers": [3, 7]}


def metadata(pack: Path, **over) -> dict:
    """The checkpoint's metadata for SHAPE and `pack`, every value a string, with `over` replacing keys (None drops one)."""
    md = {"format": attach.CHECKPOINT_FORMAT, "version": attach.CHECKPOINT_VERSION,
          **{key: str(SHAPE[field]) for field, key in attach.INTS.items() if field in SHAPE},
          "dfly2.rope_base": "10000000", "dfly2.attn_norm_eps": "9.99999997e-07", "dfly2.rms_norm_eps": "9.99999975e-06",
          "dfly2.target_layers": ",".join(str(x) for x in SHAPE["target_layers"]), "dfly2.weight_dtype": "BF16",
          "dfly2.pack_name": pack.name, "dfly2.pack_bytes": str(pack.stat().st_size),
          "dfly2.pack_meta_sha256": ModelFile(str(pack)).sections["meta"][2].hex()}
    md.update(over)
    return {k: v for k, v in md.items() if v is not None}


def write_checkpoint(path: Path, md: dict, tensors: dict, dtype="F32") -> dict:
    """A safetensors file of `tensors` (name -> shape) with seeded FP32 values; returns the values by name."""
    rng = np.random.default_rng(7)
    head, data, values = {"__metadata__": md}, b"", {}
    for name, shape in tensors.items():
        values[name] = rng.standard_normal(shape).astype(np.float32)
        raw = values[name].tobytes()
        head[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [len(data), len(data) + len(raw)]}
        data += raw
    h = json.dumps(head).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + data)
    return values


class DflyQwen4expAttachTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.t = Path(self.tmp.name)
        self.pack = self.t / "p.basalt"
        kv = {"general.architecture": "qwen4exp", "qwen4exp.embedding_length": HIDDEN, "qwen4exp.block_count": PACK_LAYERS,
              "qwen4exp.attention.head_count": HEADS, "qwen4exp.attention.head_count_kv": KV_HEADS,
              "tokenizer.ggml.tokens": ["t"] * VOCAB}
        w = ModelWriter(str(self.pack))
        w.add_bytes("dense", b"d" * 100)
        w.add_bytes("meta", json.dumps({"gguf_kv": kv, "tensors": {"output.weight": {"shape": [HIDDEN, VOCAB]}},
                                        "sources": []}).encode())
        w.close()
        self.before = self.pack.read_bytes()
        self.ck = self.t / "drafter.safetensors"
        self.tensors = attach.drafter_tensors(SHAPE)

    def args(self, **kw):
        d = dict(into=str(self.pack), dfly=str(self.ck), undo_dir=str(self.t / "undo"), roll_back=False, dspark=None, force=False)
        d.update(kw)
        return type("A", (), d)()

    def add(self, **kw):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            convert.add_dfly(self.args(**kw))
        return out.getvalue()

    def refused(self, md=None, tensors=None, dtype="F32", **kw) -> str:
        """Attaches a checkpoint that must be refused: the pack stays as it was and no undo directory is made."""
        write_checkpoint(self.ck, metadata(self.pack) if md is None else md, self.tensors if tensors is None else tensors, dtype)
        with self.assertRaises(SystemExit) as e:
            self.add(**kw)
        self.assertEqual(self.pack.read_bytes(), self.before)
        self.assertFalse((self.t / "undo").exists())
        return str(e.exception)

    def test_the_tensor_set_is_the_contracts(self):
        self.assertEqual(len(self.tensors), 11 * SHAPE["layers"] + 12)
        self.assertEqual(self.tensors["layers.1.self_attn.q_proj.weight"], [2 * HEADS * attach.HEAD_DIM, HIDDEN])
        self.assertEqual(self.tensors["hidden_correction.gate_proj.weight"], [SHAPE["correction"], 2 * HIDDEN])
        self.assertEqual(self.tensors["confidence_head.proj.bias"], [1])

    def test_attach_rounds_to_bf16_reads_back_and_rolls_back(self):
        values = write_checkpoint(self.ck, metadata(self.pack), self.tensors)
        digest = ModelFile(str(self.pack)).sections["meta"][2].hex()
        self.add()
        m = ModelFile(str(self.pack))
        self.assertTrue(m.verify("dfly") and m.verify("meta"))
        self.assertEqual(m.read("dense"), b"d" * 100)
        meta = json.loads(m.read("meta"))
        rec, sec = meta["dfly2"], m.read("dfly")
        self.assertEqual((rec["format"], rec["layers"], rec["mask_token_id"], rec["block_size"]), ("dfly2-qwen4exp", 2, 39, 8))
        self.assertEqual((rec["target_kv_layers"], rec["target_index_layers"], rec["target_index_topk"]), ([3, 7], [3, 7], 2048))
        self.assertEqual((rec["hidden_size"], rec["intermediate_size"], rec["correction_intermediate_size"]), (HIDDEN, 64, 32))
        self.assertEqual((rec["num_heads"], rec["num_kv_heads"], rec["head_dim"], rec["rope_dims"]), (HEADS, KV_HEADS, 256, 64))
        self.assertEqual((rec["rope_base"], rec["selection_width"], rec["causal_block"], rec["mtp_fusion"], rec["step"]),
                         (1e7, 2051, True, "slot0", 0))
        self.assertAlmostEqual(rec["attn_norm_eps"], 1e-6, places=12)
        self.assertNotIn("context_local_window", rec)
        self.assertEqual(rec["source"], {"pack_name": "p.basalt", "pack_bytes": len(self.before), "pack_meta_sha256": digest})
        self.assertEqual(rec["bytes"], len(sec))
        self.assertEqual(set(rec["tensors"]), set(self.tensors))
        self.assertEqual(meta["sources"][-1]["path"], "drafter.safetensors")
        for name, t in rec["tensors"].items():
            # nearest-even BF16 of the FP32 master, at a 64-byte aligned offset from the section's start
            want = ggufq.quantize(values[name], ggufq.GGMLQuantizationType.BF16).tobytes()
            self.assertEqual((t["dtype"], t["shape"], t["bytes"], t["offset"] % 64), ("BF16", self.tensors[name], len(want), 0))
            self.assertEqual(sec[t["offset"]:t["offset"] + t["bytes"]], want)
        self.assertEqual(attach.section_problems(meta, len(sec)), [])
        convert.add_dspark(self.args(roll_back=True))
        self.assertEqual(self.pack.read_bytes(), self.before)

    def test_rounding_is_to_nearest_even(self):
        # 1 + 2^-8 lies halfway between two BF16 values and goes to the even one; 1 + 3 * 2^-8 goes up
        tensors = {"confidence_head.proj.bias": [1]}
        blob = self.t / "blob"
        head = {"__metadata__": {}, "confidence_head.proj.bias": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
        for value, want in ((1.0 + 2.0 ** -8, 0x3F80), (1.0 + 3 * 2.0 ** -8, 0x3F82)):
            h = json.dumps(head).encode()
            self.ck.write_bytes(struct.pack("<Q", len(h)) + h + struct.pack("<f", value))
            attach.write_section(self.ck, tensors, blob)
            self.assertEqual(struct.unpack("<H", blob.read_bytes())[0], want)

    def test_a_drafter_that_is_not_the_packs_is_refused(self):
        for key, value, word in (
                ("dfly2.hidden_size", "64", "hidden_size"), ("dfly2.num_heads", "4", "num_heads"),
                ("dfly2.num_kv_heads", "2", "num_kv_heads"), ("dfly2.head_dim", "128", "head_dim"),
                ("dfly2.vocab_size", "41", "vocab_size"), ("dfly2.target_layers", "2,7", "full-attention"),
                ("dfly2.target_layers", "7,3", "full-attention"), ("dfly2.target_layers", "3,11", "full-attention"),
                ("dfly2.target_layers", "3", "target layers"), ("dfly2.selection_width", "2048", "selection_width"),
                ("dfly2.block_size", "9", "block_size"), ("dfly2.block_size", "1", "block_size"),
                ("dfly2.mask_token_id", "40", "mask token"), ("dfly2.rope_dims", "63", "rope_dims"),
                ("dfly2.correction_intermediate_size", "48", "multiple of 32"), ("dfly2.weight_dtype", "F16", "weight_dtype"),
                ("dfly2.num_layers", "9", "draft layers"), ("format", "other", "format"), ("version", "2", "version"),
                ("dfly2.step", None, "dfly2.step"), ("dfly2.rope_base", "x", "not a number"),
                ("dfly2.target_layers", "3,x", "malformed")):
            with self.subTest(key=key, value=value):
                self.assertIn(word, self.refused(md=metadata(self.pack, **{key: value})))

    def test_a_wrong_tensor_set_is_refused(self):
        self.assertIn("lacks", self.refused(tensors={k: v for k, v in self.tensors.items() if k != "mtp_fusion.gate"}))
        self.assertIn("does not", self.refused(tensors={**self.tensors, "extra.weight": [2]}))
        self.assertIn("FP32", self.refused(dtype="BF16"))
        self.assertIn("a drafter of this shape", self.refused(tensors={**self.tensors, "final_norm.weight": [HIDDEN + 1]}))
        values = write_checkpoint(self.ck, metadata(self.pack), self.tensors)
        raw = bytearray(self.ck.read_bytes())
        raw[-4:] = struct.pack("<f", float("nan"))
        self.ck.write_bytes(bytes(raw))
        with self.assertRaises(SystemExit) as e:
            self.add()
        self.assertIn("not finite", str(e.exception))
        self.assertEqual(self.pack.read_bytes(), self.before)
        self.assertEqual(len(values), len(self.tensors))

    def test_another_pack_or_rotary_base_needs_force(self):
        for over in ({"dfly2.pack_bytes": "1"}, {"dfly2.pack_meta_sha256": "00" * 32}, {"dfly2.rope_base": "10000"}):
            with self.subTest(over=over):
                self.assertIn("--force", self.refused(md=metadata(self.pack, **over)))
        write_checkpoint(self.ck, metadata(self.pack, **{"dfly2.pack_bytes": "1", "dfly2.rope_base": "10000"}), self.tensors)
        out = self.add(force=True)
        self.assertEqual(out.count("warning"), 2)
        rec = json.loads(ModelFile(str(self.pack)).read("meta"))["dfly2"]
        self.assertEqual((rec["rope_base"], rec["source"]["pack_bytes"]), (1e4, 1))

    def test_the_packs_own_rotary_base_is_the_one_compared(self):
        m = ModelFile(str(self.pack))
        meta = json.loads(m.read("meta"))
        meta["gguf_kv"]["qwen4exp.rope.freq_base"] = 10000.0
        w = ModelWriter(str(self.pack))
        w.add_bytes("meta", json.dumps(meta).encode())
        w.close()
        self.before = self.pack.read_bytes()
        self.assertIn("rotary base", self.refused())
        write_checkpoint(self.ck, metadata(self.pack, **{"dfly2.rope_base": "10000"}), self.tensors)
        self.add()

    def test_a_second_drafter_another_architecture_and_a_trained_f32_drafter(self):
        write_checkpoint(self.ck, metadata(self.pack, **{"dfly2.weight_dtype": "F32"}), self.tensors)
        self.assertIn("rounded to BF16", self.add())
        with self.assertRaises(SystemExit) as e:
            self.add(undo_dir=str(self.t / "undo2"))
        self.assertIn("has a drafter already", str(e.exception))
        other = self.t / "o.basalt"
        w = ModelWriter(str(other))
        w.add_bytes("meta", json.dumps({"gguf_kv": {"general.architecture": "other"}}).encode())
        w.close()
        with self.assertRaises(SystemExit) as e:
            self.add(into=str(other), undo_dir=str(self.t / "undo3"))
        self.assertIn("takes no DFly2 drafter", str(e.exception))

    def test_a_record_that_does_not_fit_its_section_is_reported(self):
        write_checkpoint(self.ck, metadata(self.pack), self.tensors)
        self.add()
        m = ModelFile(str(self.pack))
        meta = json.loads(m.read("meta"))
        size = m.section("dfly")[1]
        self.assertEqual(len(attach.section_problems(meta, None)), 1)
        self.assertEqual(len(attach.section_problems(meta, size - 1)), 2)   # the size, and the last tensor's end
        rec = meta["dfly2"]["tensors"]
        rec["final_norm.weight"]["offset"] += 2
        rec["mtp_fusion.gate"]["dtype"] = "F32"
        self.assertEqual(len(attach.section_problems(meta, size)), 3)       # alignment, an overlap, the type
        self.assertEqual(attach.section_problems({"dfly2": {"format": "safetensors"}}, None), [])


if __name__ == "__main__":
    unittest.main()
