# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "torch>=1.13.0",
#     "numpy>=1.19.5",
# ]
#
# [tool.uv.sources]
# torch = [{ index = "pytorch-cpu" }]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""
Regression test: non-power-of-two audio lengths used to crash CQT construction in
the *_complete modes whenever the DC filter length came out odd (the DC band's
mirrored-tail indexing assumed an even length). Every (Ls, numocts) pair below
crashed before the fix.

Uses synthetic noise, so it needs no audio files. Asserts construction succeeds
and round-trip SNR stays near-perfect.

Run with:
    uv run tests/test_odd_lengths.py
"""
import sys

import numpy as np
import torch

import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from cqt_nsgt_pytorch import CQT_nsgt

FS = 44100
# White noise (flat spectrum) exposes the transform's float32 noise floor far more
# than musical signals: even-DC configs that always worked give ~55-73 dB on noise
# (vs ~90-115 dB on music). A broken transform lands near 0-20 dB, so 40 dB cleanly
# separates "working" from "broken" without false alarms.
MIN_SNR = 40.0
# (Ls, numocts) pairs whose DC filter length is odd — all crashed before the fix
CASES = [(525312, 10), (400000, 7), (400000, 9), (400000, 10),
         (300000, 5), (300000, 6), (300000, 10)]


def snr_db(ref, rec):
    rec = rec[..., :ref.shape[-1]]
    noise = ((rec - ref) ** 2).sum().item()
    return 10 * np.log10((ref ** 2).sum().item() / max(noise, 1e-20))


def main():
    torch.manual_seed(0)
    failures = 0
    for Ls, numocts in CASES:
        x = torch.randn(1, 1, Ls) * 0.1
        for mode in ("oct_complete", "matrix_complete"):
            try:
                cqt = CQT_nsgt(numocts, 32, mode=mode, fs=FS, audio_len=Ls,
                               device="cpu", dtype=torch.float32)
                snr = snr_db(x, cqt.bwd(cqt.fwd(x)))
                ok = snr >= MIN_SNR
                failures += not ok
                print(f"Ls={Ls} numocts={numocts:2d} {mode:16s}: {snr:7.2f} dB "
                      f"{'OK' if ok else f'FAIL (<{MIN_SNR})'}")
            except Exception as e:
                failures += 1
                print(f"Ls={Ls} numocts={numocts:2d} {mode:16s}: CRASH {type(e).__name__}: {e}")

    print()
    if failures:
        print(f"{failures} case(s) FAILED.")
        sys.exit(1)
    print("all odd-length cases OK")


if __name__ == "__main__":
    main()
