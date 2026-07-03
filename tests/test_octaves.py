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
Check that the CQT frequency scale actually matches the requested geometry.

Convention: `numocts` means exactly `numocts` octaves of coverage. fmin is placed
exactly `numocts` octaves below Nyquist and bins step at exactly 1/binsoct octave;
Nyquist (added as a separate band) lands on the grid as bin numocts*binsoct. So we
assert:

  * fmin == Nyquist / 2**numocts  (exact octave placement),
  * the spacing gives exactly `binsoct` bins per octave,
  * the total number of log bins is `numocts * binsoct`,
  * fmin .. Nyquist spans exactly `numocts` octaves.

Run with:
    uv run tests/test_octaves.py
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
CASES = [(4, 12), (8, 32), (9, 64)]


def measure(numocts, binsoct):
    cqt = CQT_nsgt(numocts, binsoct, mode="oct_complete", fs=FS,
                   audio_len=LS, device="cpu", dtype=torch.float32)
    frqs = np.asarray(cqt.frqs)                     # log bins only (no DC/Nyquist)
    nyq = FS / 2.0
    step_oct = np.log2(frqs[1] / frqs[0])           # octaves between adjacent bins
    return dict(nbins=len(frqs), fmin=frqs[0],
                cover_oct=np.log2(nyq / frqs[0]),    # fmin .. Nyquist coverage
                bins_per_oct=1.0 / step_oct)


def main():
    tol = 1e-3
    nyq = FS / 2.0
    failures = 0
    for numocts, binsoct in CASES:
        m = measure(numocts, binsoct)
        expect_fmin = nyq / 2 ** numocts
        fmin_ok = abs(m["fmin"] - expect_fmin) / expect_fmin < tol
        cover_ok = abs(m["cover_oct"] - numocts) < tol
        count_ok = m["nbins"] == numocts * binsoct
        bpo_ok = abs(m["bins_per_oct"] - binsoct) < tol

        print(f"numocts={numocts} binsoct={binsoct}: "
              f"nbins={m['nbins']} (expect {numocts*binsoct}) {'OK' if count_ok else 'FAIL'} | "
              f"bins/oct={m['bins_per_oct']:.4f} (expect {binsoct}) {'OK' if bpo_ok else 'FAIL'} | "
              f"fmin={m['fmin']:.2f} (expect {expect_fmin:.2f}) {'OK' if fmin_ok else 'FAIL'} | "
              f"cover={m['cover_oct']:.4f} oct (expect {numocts}) {'OK' if cover_ok else 'FAIL'}")

        failures += not (fmin_ok and cover_ok and count_ok and bpo_ok)

    print()
    if failures:
        print(f"{failures}/{len(CASES)} cases FAILED.")
        sys.exit(1)
    print("all cases OK")


if __name__ == "__main__":
    main()
