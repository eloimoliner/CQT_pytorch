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
Per-bin transfer-function test: H = rfft(bwd(fwd(x))) / rfft(x) on white noise,
computed in float64 so precision cannot mask systematic errors. For a
perfect-reconstruction mode H must be 1 at EVERY bin — this is the probe that
caught the dropped Nyquist bin (H[N] was exactly 0 while all other bins were 1).

Covers both parities of the Nyquist window length (binsoct=32 -> odd,
binsoct=24 -> even), since the DC/Nyquist band assembly special-cases them.

Run with:
    uv run tests/test_transfer.py
"""
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from cqt_nsgt_pytorch import CQT_nsgt

FS = 44100
LS = 131072
NUMOCTS = 8
MODES = ("matrix_complete", "oct_complete", "critical")
BINSOCTS = (32, 24)   # odd / even Nyquist window length
MAX_DEV = 1e-8        # |H-1| tolerance; systematic errors show up as O(1)


def main():
    torch.manual_seed(0)
    x = torch.randn(1, 1, LS, dtype=torch.float64) * 0.1
    F = torch.fft.rfft(x).squeeze()

    failures = 0
    for binsoct in BINSOCTS:
        for mode in MODES:
            cqt = CQT_nsgt(NUMOCTS, binsoct, mode=mode, fs=FS, audio_len=LS,
                           device="cpu", dtype=torch.float64)
            Lg = len(cqt.g[len(cqt.g) // 2])
            xh = cqt.bwd(cqt.fwd(x))[..., :LS]
            H = (torch.fft.rfft(xh).squeeze() / F).numpy()
            dev = np.abs(H - 1)
            ok = dev.max() < MAX_DEV
            failures += not ok
            print(f"binsoct={binsoct} (nyq Lg {'odd' if Lg % 2 else 'even'}) "
                  f"{mode:16s}: max|H-1|={dev.max():.2e} at bin {dev.argmax()} "
                  f"| H[Nyquist]={np.abs(H[-1]):.6f} {'OK' if ok else 'FAIL'}")

    print()
    if failures:
        print(f"{failures} case(s) FAILED.")
        sys.exit(1)
    print("all transfer functions flat at 1 — perfect reconstruction")


if __name__ == "__main__":
    main()
