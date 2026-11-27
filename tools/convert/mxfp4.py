"""The MXFP4 row repack the checkpoint converters share, for tools/convert/convert.py: a checkpoint's rows (E2M1 codes two
to a byte, one E8M0 scale per 32 values) as ggml MXFP4 rows, codes and scales unchanged. Needs numpy.
"""
from __future__ import annotations

import numpy as np

import model_sections as S


def mxfp4_rows(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """A checkpoint's MXFP4 rows (codes [rows, cols / 2] uint8: value 2i in the low nibble of byte i, 2i + 1 in the high
    one; E8M0 scales [rows, cols / 32] uint8) as ggml MXFP4 rows (type 39), for any cols that is whole blocks of 32: per
    32 values the scale byte, then 16 bytes whose low nibble is value j and high nibble value j + 16."""
    # a row is whole blocks of 16 code bytes, each with its one scale
    if codes.ndim != 2 or codes.shape[1] % 16 != 0 or scales.shape != (codes.shape[0], codes.shape[1] // 16):
        raise S.ConvertError(f"mxfp4_rows: shape mismatch codes={codes.shape} scales={scales.shape}")
    rows, nb = codes.shape[0], codes.shape[1] // 16
    # a block's code bytes as two 8-byte words (values 0..15, then 16..31), so the nibbles of 8 bytes move at once
    c64 = np.ascontiguousarray(codes, dtype=np.uint8).view(np.uint64).reshape(rows, nb, 2)
    lo, hi = c64[..., 0], c64[..., 1]
    # the even values of both halves share a byte, and so do the odd ones
    ev = (lo & np.uint64(0x0F0F0F0F0F0F0F0F)) | ((hi & np.uint64(0x0F0F0F0F0F0F0F0F)) << np.uint64(4))
    od = ((lo >> np.uint64(4)) & np.uint64(0x0F0F0F0F0F0F0F0F)) | (hi & np.uint64(0xF0F0F0F0F0F0F0F0))
    out = np.empty((rows, nb, 17), np.uint8)
    out[..., 0] = scales
    # bytes 2i and 2i + 1 of a block: values 2i | 2i + 16, then 2i + 1 | 2i + 17
    q = out[..., 1:].reshape(rows, nb, 8, 2)
    q[..., 0] = ev.view(np.uint8).reshape(rows, nb, 8)
    q[..., 1] = od.view(np.uint8).reshape(rows, nb, 8)
    return out.reshape(-1)
