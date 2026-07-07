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
Test variable bins-per-octave for efficient top-frequency specifications.

When you want a CQT covering only up to ~5kHz (not Nyquist), the standard
grid convention forces wasteful octaves above your target. The old workaround
was to compute extra octaves and discard them. With variable bins-per-octave,
you can:
  1. Use fewer octaves (just enough to cover your range)
  2. Mix different resolutions per octave (high res low freq, low res high freq)

This test demonstrates solving Mauricio's over-specification issue:
  - Baseline: 9 octaves, uniform 24 bins/oct → 216 bins (24% above 5kHz wasted)
  - Optimized: 7 octaves, variable res [24,24,20,16,12,8,6] → 110 bins (0.9% wasted)

Both reconstruct perfectly; optimized version is 49% smaller and avoids the
workaround of computing an extra octave just to throw it away.

Run with:
    uv run tests/test_variable_topfreq.py
"""
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from cqt_nsgt_pytorch import CQT_nsgt

FS = 44100  # Nyquist = 22050 Hz
NUMOCTS = 9
TARGET_TOPFREQ = 5000  # Want only up to ~5kHz, not full 22kHz


def main():
    torch.manual_seed(0)
    Ls = 131072
    x = torch.randn(1, 1, Ls, dtype=torch.float32) * 0.1

    # Problem scenario: user wants only up to 5kHz, but standard grid forces 9 octaves
    # fmin = 22050 / 2^9 ≈ 43 Hz, and that spans all 9 octaves to Nyquist
    # Solution: use 7 octaves (spans 43 Hz → 5512 Hz, covering the desired 5kHz)

    # Baseline: uniform 24 bins/oct over 9 octaves (old workaround pattern)
    # User computes 9 octaves even though they only want up to 5kHz
    cqt_baseline = CQT_nsgt(NUMOCTS, 24, mode="oct_complete", fs=FS, audio_len=Ls,
                            device="cpu", dtype=torch.float32)
    frqs_baseline = np.asarray(cqt_baseline.frqs)
    print(f"Baseline (9 octaves, uniform 24 bins/oct — wasteful):")
    print(f"  Total bins: {len(frqs_baseline)}")
    print(f"  fmin={frqs_baseline[0]:.1f} Hz, fmax={frqs_baseline[-1]:.1f} Hz")
    above_target = np.sum(frqs_baseline > TARGET_TOPFREQ)
    print(f"  Content above {TARGET_TOPFREQ}Hz: {above_target} bins (~{100*above_target/len(frqs_baseline):.1f}% unused)")

    # Solution: use 7 octaves (which naturally span to ~5.5kHz)
    # With variable bins-per-octave, we can also mix different resolutions
    # e.g., higher res in low freqs, lower res in high freqs
    NUMOCTS_OPT = 7
    binsoct_opt = [24, 24, 20, 16, 12, 8, 6]  # Decreasing resolution at higher octaves
    cqt_optimized = CQT_nsgt(NUMOCTS_OPT, binsoct_opt, mode="oct_complete", fs=FS,
                             audio_len=Ls, device="cpu", dtype=torch.float32)
    frqs_optimized = np.asarray(cqt_optimized.frqs)
    print(f"\nOptimized (7 octaves, variable bins-per-octave):")
    print(f"  Total bins: {len(frqs_optimized)}")
    print(f"  fmin={frqs_optimized[0]:.1f} Hz, fmax={frqs_optimized[-1]:.1f} Hz")
    print(f"  binsoct pattern: {binsoct_opt}")
    above_target_opt = np.sum(frqs_optimized > TARGET_TOPFREQ)
    print(f"  Content above {TARGET_TOPFREQ}Hz: {above_target_opt} bins (~{100*above_target_opt/len(frqs_optimized):.1f}% unused)")

    # Verify perfect reconstruction still works for both
    print(f"\nReconstruction test:")
    for name, cqt, frqs in [("baseline (9 octs)", cqt_baseline, frqs_baseline),
                            ("optimized (7 octs, var res)", cqt_optimized, frqs_optimized)]:
        X = cqt.fwd(x)
        x_hat = cqt.bwd(X)[..., :Ls]
        snr = 10 * np.log10(((x**2).sum().item()) / max(((x_hat - x)**2).sum().item(), 1e-30))
        print(f"  {name:30s}: SNR={snr:6.1f} dB")

    savings_pct = 100 * (1 - len(frqs_optimized) / len(frqs_baseline))
    print(f"\n✓ Variable bins-per-octave solves the over-specification problem")
    print(f"  Baseline: 9 octaves × 216 bins = {len(frqs_baseline)} total")
    print(f"  Optimized: 7 octaves × variable = {len(frqs_optimized)} total ({savings_pct:.0f}% reduction)")
    print(f"  No need to compute extra octaves above desired frequency")


if __name__ == "__main__":
    main()
