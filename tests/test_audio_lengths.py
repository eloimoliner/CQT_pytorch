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
Reconstruction must be perfect for ANY audio_len, not just powers of two.

Two length-dependent defects used to break this (both fixed by padding the internal
FFT grid past audio_len and truncating back on the inverse):

  * odd audio_len -- the NSGT places a real Nyquist band at bin nn//2, which only
    exists for an even FFT length. Odd lengths corrupted the reconstruction
    (|H-1| ~ 1.3) or overflowed a half-spectrum index at construction (crash).
  * oct/oct_complete at lengths where the coarsest octave's power-of-2 coefficient
    length did not divide the FFT length -- the frame operator developed a near-gap
    and lost ~1e-5 (e.g. Ls=48000, 88200).

This probes the full per-bin transfer function H = rfft(bwd(fwd(x)))/rfft(x) in
float64 over a spread of awkward lengths: odd, prime, and arbitrary non-power-of-two.

Run with:
    uv run tests/test_audio_lengths.py
"""
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from cqt_nsgt_pytorch import CQT_nsgt

FS = 44100
NUMOCTS = 8
BINSOCT = 32
MAX_DEV = 1e-6            # systematic length defects were 1e-5 .. 1.3; float64 floor is ~1e-11
MODES = ("oct_complete", "matrix_complete")
# odd, prime, x44100 multiples, and assorted non-power-of-two lengths that used to fail
LENGTHS = [44100, 48000, 65537, 88200, 100003, 131073, 131075, 176400,
           262144, 300000, 400001, 525312, 700001]


def main():
    torch.manual_seed(0)
    failures = 0
    for Ls in LENGTHS:
        x = torch.randn(1, 1, Ls, dtype=torch.float64) * 0.1
        F = torch.fft.rfft(x).squeeze()
        for mode in MODES:
            try:
                cqt = CQT_nsgt(NUMOCTS, BINSOCT, mode=mode, fs=FS, audio_len=Ls,
                               device="cpu", dtype=torch.float64)
                xh = cqt.bwd(cqt.fwd(x))[..., :Ls]
            except Exception as e:
                failures += 1
                print(f"Ls={Ls:7d} {mode:16s}: CRASH {type(e).__name__}: {e}")
                continue
            H = (torch.fft.rfft(xh).squeeze() / F).numpy()
            dev = np.abs(H - 1).max()
            ok = dev < MAX_DEV
            failures += not ok
            print(f"Ls={Ls:7d} ({'odd ' if Ls % 2 else 'even'}) {mode:16s}: "
                  f"max|H-1|={dev:.2e} {'OK' if ok else 'FAIL'}")

    print()
    if failures:
        print(f"{failures} case(s) FAILED.")
        sys.exit(1)
    print("all audio lengths reconstruct perfectly")


if __name__ == "__main__":
    main()
