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
The non-complete modes ("oct", "matrix") discard the DC and Nyquist bands, so a bare
round trip bwd(fwd(x)) only recovers the mid-band (high-pass) part of x. The transform
ships apply_hpf_DC / apply_lpf_DC for exactly this: Hlpf is the combined DC+Nyquist band
transfer and Hhpf = 1 - Hlpf, so the two are complementary and

    apply_hpf_DC(x) + apply_lpf_DC(x) == x                    (exact partition)
    bwd(fwd(x)) + apply_lpf_DC(x)    ~= x   for oct / matrix   (add the discarded DC/Nyq back)

This checks both, over several lengths (incl. odd), asserting near-perfect SNR. The
complete modes are intentionally excluded: they already carry DC and Nyquist, so adding
the LPF term would double-count them (this test would (correctly) fail for them).

Run with:
    uv run tests/test_hpf_lpf.py
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
MODES = ("oct", "matrix")
LENGTHS = (131072, 262144, 100003, 131073)   # incl. odd, to exercise the padded grid
PARTITION_MIN_SNR = 200.0   # Hhpf = 1 - Hlpf is exact up to float round-off
FULL_MIN_SNR = 200.0        # with a correct Hlpf, oct/matrix + lpf hits the float64 floor (~280 dB)


def snr_db(ref, rec):
    rec = rec[..., :ref.shape[-1]]
    noise = ((rec - ref) ** 2).sum().item()
    return 10 * np.log10((ref ** 2).sum().item() / max(noise, 1e-30))


def main():
    torch.manual_seed(0)
    failures = 0
    for Ls in LENGTHS:
        x = torch.randn(1, 1, Ls, dtype=torch.float64) * 0.1
        for mode in MODES:
            cqt = CQT_nsgt(NUMOCTS, BINSOCT, mode=mode, fs=FS, audio_len=Ls,
                           device="cpu", dtype=torch.float64)
            x_hpf = cqt.apply_hpf_DC(x)
            x_lpf = cqt.apply_lpf_DC(x)

            partition = snr_db(x, x_hpf + x_lpf)               # exact complementary filters
            full = cqt.bwd(cqt.fwd(x))[..., :Ls] + x_lpf       # mid bands + discarded DC/Nyq
            full_snr = snr_db(x, full)

            p_ok = partition >= PARTITION_MIN_SNR
            f_ok = full_snr >= FULL_MIN_SNR
            failures += not (p_ok and f_ok)
            print(f"Ls={Ls:7d} {mode:7s}: hpf+lpf==x {partition:6.1f} dB "
                  f"{'OK' if p_ok else 'FAIL'} | bwd(fwd)+lpf==x {full_snr:6.1f} dB "
                  f"{'OK' if f_ok else 'FAIL'}")

    print()
    if failures:
        print(f"{failures} case(s) FAILED.")
        sys.exit(1)
    print("DC/Nyquist high-pass + low-pass split reconstructs perfectly")


if __name__ == "__main__":
    main()
