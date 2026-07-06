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
Randomised stress test for variable bins-per-octave. Draws many (seeded, so
deterministic) per-octave bin lists -- varying numocts and mixing small/large,
monotonic and non-monotonic bins per octave -- and for each asserts:

  * construction succeeds,
  * the frequency grid matches the request (total bins and per-octave counts),
  * oct_complete and matrix_complete reconstruct perfectly (per-bin |H-1| < 1e-7),
  * oct mode constructs and gives a finite round-trip SNR.

Lengths are powers of two so this isolates the variable-binsoct machinery from the
separate audio-length handling. Non-monotonic bins per octave are the important case:
they make adjacent octaves round to equal window lengths, which the octave-boundary
bookkeeping must handle by position, not by comparing window lengths.

Run with:
    uv run tests/test_variable_binsoct_random.py
"""
import os
import random
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from cqt_nsgt_pytorch import CQT_nsgt

FS = 44100
SEED = 12345
N_CONFIGS = 40
BIN_CHOICES = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
LENGTHS = [131072, 262144]
MAX_DEV = 1e-7


def main():
    random.seed(SEED)
    failures = 0
    worst = 0.0
    for t in range(N_CONFIGS):
        numocts = random.randint(2, 11)
        binsoct = [random.choice(BIN_CHOICES) for _ in range(numocts)]
        Ls = random.choice(LENGTHS)
        torch.manual_seed(t)
        x = torch.randn(1, 1, Ls, dtype=torch.float64) * 0.1
        F = torch.fft.rfft(x).squeeze()
        tag = f"numocts={numocts:2d} binsoct={binsoct}"
        try:
            for mode in ("oct_complete", "matrix_complete"):
                cqt = CQT_nsgt(numocts, binsoct, mode=mode, fs=FS, audio_len=Ls,
                               device="cpu", dtype=torch.float64)
                assert len(cqt.frqs) == sum(binsoct), \
                    f"{mode}: nbins {len(cqt.frqs)} != {sum(binsoct)}"
                for i, b in enumerate(binsoct):
                    seg = np.asarray(cqt.frqs)[cqt.oct_offsets[i]:cqt.oct_offsets[i + 1]]
                    assert len(seg) == b, f"{mode}: octave {i} has {len(seg)} bins, want {b}"
                xh = cqt.bwd(cqt.fwd(x))[..., :Ls]
                dev = np.abs((torch.fft.rfft(xh).squeeze() / F).numpy() - 1).max()
                worst = max(worst, dev)
                if dev >= MAX_DEV:
                    failures += 1
                    print(f"FAIL {tag} {mode}: max|H-1|={dev:.2e}")

            cqt = CQT_nsgt(numocts, binsoct, mode="oct", fs=FS, audio_len=Ls,
                           device="cpu", dtype=torch.float32)
            xf = x.float()
            rec = cqt.bwd(cqt.fwd(xf))[..., :Ls]
            snr = 10 * np.log10(float((xf ** 2).sum()) /
                                max(float(((rec - xf) ** 2).sum()), 1e-30))
            if not np.isfinite(snr):
                failures += 1
                print(f"FAIL {tag} oct: non-finite SNR {snr}")
        except Exception as e:
            failures += 1
            print(f"CRASH {tag}: {type(e).__name__}: {e}")

    print(f"\n{N_CONFIGS} random configs | worst PR transfer dev={worst:.2e} | "
          f"failures={failures}")
    if failures:
        sys.exit(1)
    print("all random bins-per-octave configs reconstruct perfectly")


if __name__ == "__main__":
    main()
