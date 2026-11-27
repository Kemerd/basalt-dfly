"""Tests for dflash.py (the DFlash 2 block drafter's tensors and metadata) on a tiny synthetic DFlash2DraftModel, alone
and through convert.py --checkpoint QWEN35_DIR --drafter DIR; and, when BASALT_DFLASH2_DIR names the directory of a
released drafter (incoai/Qwen3.8-Flash-Next-DFlash2), on that checkpoint.

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
import dflash  # noqa: E402
import model_sections as S  # noqa: E402
from model_file import ModelFile  # noqa: E402
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


def write_drafter(d: Path, drop=(), extra=None, dtype="BF16", **config) -> dict:
    """A DFlash2DraftModel directory for the synthetic model; returns its tensors' values by checkpoint name."""
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
        tens[k] = (dtype, shape, (v[k].view(np.uint32) >> 16).astype("<u2").tobytes())
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


if __name__ == "__main__":
    unittest.main()
