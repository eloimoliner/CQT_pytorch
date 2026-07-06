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
Regression tests for variable bins-per-octave: binsoct can now be a list of
length numocts (index 0 = lowest octave, closest to DC) instead of one scalar
shared by every octave, letting a single CQT_nsgt cover several frequency
resolutions the way separate CQTs would (e.g. the MR-CQTdiff paper's
{8,16,32} bins/oct split), without needing multiple CQT objects.

Covers:
  1. geometry -- each octave region actually has its requested bins/oct,
     and the exact-octave grid convention (fmin==Nyquist/2**numocts) still holds.
  2. the tie regression this feature exposed: with uniform binsoct, adjacent
     octaves' rounded window lengths (M) never coincide, so the octave-boundary
     bucketing in get_ragged_giis/get_ragged_gdiis_oct (keyed off M changing)
     never had to handle a tie. A bins/oct jump (e.g. 8->16) roughly doubles
     the window length, which can offset the usual per-octave halving and tie
     M across the transition -- this used to merge two octaves into one bucket
     (crash in "oct" mode, misaligned octave list in "oct_complete"). The fix
     detects octave boundaries from the known per-octave bin counts instead.
  3. transfer function (mirrors test_transfer.py) for "oct_complete"/"matrix_complete".
  4. "oct" mode SNR round trip (mirrors test_odd_lengths.py; "oct" drops DC/Nyquist
     so no exact transfer-function check there).
  5. uniform-equivalence guard: scalar binsoct vs [b]*numocts must give the same
     frequency grid and transfer function, proving list support changes nothing
     for existing scalar callers.

Run with:
    uv run tests/test_variable_binsoct.py
"""
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from cqt_nsgt_pytorch import CQT_nsgt

FS = 44100
NUMOCTS = 9
BINSOCT = [8, 8, 8, 16, 16, 16, 16, 32, 32]  # the paper's per-octave split
TIE_LENGTHS = (44100, 131072, 400000)  # Ls values that tie at the 8->16 and 16->32 transitions


def snr_db(ref, rec):
    rec = rec[..., :ref.shape[-1]]
    noise = ((rec - ref) ** 2).sum().item()
    return 10 * np.log10((ref ** 2).sum().item() / max(noise, 1e-20))


def test_geometry():
    Ls = 131072
    cqt = CQT_nsgt(NUMOCTS, BINSOCT, mode="oct_complete", fs=FS, audio_len=Ls,
                   device="cpu", dtype=torch.float32)
    frqs = np.asarray(cqt.frqs)
    offsets = cqt.oct_offsets
    nyq = FS / 2.0
    tol = 1e-3

    failures = 0
    assert len(frqs) == sum(BINSOCT), f"total bins {len(frqs)} != {sum(BINSOCT)}"
    for i, b in enumerate(BINSOCT):
        seg = frqs[offsets[i]:offsets[i + 1]]
        ok_count = len(seg) == b
        step_oct = np.log2(seg[1] / seg[0]) if len(seg) > 1 else np.log2(frqs[offsets[i + 1]] / seg[0])
        bins_per_oct = 1.0 / step_oct
        ok_bpo = abs(bins_per_oct - b) < tol
        print(f"octave {i}: requested {b} bins/oct -> got {len(seg)} bins, "
              f"measured {bins_per_oct:.4f} bins/oct {'OK' if ok_count and ok_bpo else 'FAIL'}")
        failures += not (ok_count and ok_bpo)

    expect_fmin = nyq / 2 ** NUMOCTS
    fmin_ok = abs(frqs[0] - expect_fmin) / expect_fmin < tol
    print(f"fmin={frqs[0]:.3f} (expect {expect_fmin:.3f}) {'OK' if fmin_ok else 'FAIL'}")
    failures += not fmin_ok

    print(f"geometry: {'OK' if not failures else f'{failures} FAILED'}\n")
    return failures


def test_tie_regression():
    failures = 0
    for Ls in TIE_LENGTHS:
        for mode in ("oct", "oct_complete"):
            try:
                cqt = CQT_nsgt(NUMOCTS, BINSOCT, mode=mode, fs=FS, audio_len=Ls,
                                device="cpu", dtype=torch.float32)
                print(f"Ls={Ls} {mode:12s}: size_per_oct={cqt.size_per_oct} constructed OK")
            except Exception as e:
                failures += 1
                print(f"Ls={Ls} {mode:12s}: CRASH {type(e).__name__}: {e}")
    print(f"tie regression: {'OK' if not failures else f'{failures} FAILED'}\n")
    return failures


def test_transfer_function():
    torch.manual_seed(0)
    Ls = 131072
    x = torch.randn(1, 1, Ls, dtype=torch.float64) * 0.1
    F = torch.fft.rfft(x).squeeze()

    failures = 0
    for mode in ("oct_complete", "matrix_complete"):
        cqt = CQT_nsgt(NUMOCTS, BINSOCT, mode=mode, fs=FS, audio_len=Ls,
                       device="cpu", dtype=torch.float64)
        xh = cqt.bwd(cqt.fwd(x))[..., :Ls]
        H = (torch.fft.rfft(xh).squeeze() / F).numpy()
        dev = np.abs(H - 1)
        ok = dev.max() < 1e-8
        failures += not ok
        print(f"{mode:16s}: max|H-1|={dev.max():.2e} at bin {dev.argmax()} "
              f"| H[Nyquist]={np.abs(H[-1]):.6f} {'OK' if ok else 'FAIL'}")
    print(f"transfer function: {'OK' if not failures else f'{failures} FAILED'}\n")
    return failures


def test_oct_mode_snr():
    # plain "oct" mode discards DC and Nyquist by design (see README), so on
    # broadband white noise its SNR is inherently much lower than *_complete
    # modes -- that's expected, not a defect. Guard against a regression by
    # comparing to the uniform-binsoct baseline instead of an absolute dB gate.
    torch.manual_seed(0)
    Ls = 131072
    x = torch.randn(1, 1, Ls) * 0.1

    cqt_uniform = CQT_nsgt(8, 32, mode="oct", fs=FS, audio_len=Ls,
                           device="cpu", dtype=torch.float32)
    snr_uniform = snr_db(x, cqt_uniform.bwd(cqt_uniform.fwd(x)))

    cqt = CQT_nsgt(NUMOCTS, BINSOCT, mode="oct", fs=FS, audio_len=Ls,
                   device="cpu", dtype=torch.float32)
    snr = snr_db(x, cqt.bwd(cqt.fwd(x)))

    ok = snr >= snr_uniform - 5.0  # same ballpark as the uniform baseline
    print(f"oct mode SNR: variable-binsoct={snr:.2f} dB uniform-binsoct={snr_uniform:.2f} dB "
          f"{'OK' if ok else 'FAIL'}\n")
    return not ok


def test_uniform_equivalence():
    torch.manual_seed(0)
    Ls = 131072
    b = 32
    numocts = 8
    x = torch.randn(1, 1, Ls, dtype=torch.float64) * 0.1
    F = torch.fft.rfft(x).squeeze()

    failures = 0
    for mode in ("oct_complete", "matrix_complete"):
        cqt_scalar = CQT_nsgt(numocts, b, mode=mode, fs=FS, audio_len=Ls,
                              device="cpu", dtype=torch.float64)
        cqt_list = CQT_nsgt(numocts, [b] * numocts, mode=mode, fs=FS, audio_len=Ls,
                            device="cpu", dtype=torch.float64)

        frqs_ok = np.allclose(cqt_scalar.frqs, cqt_list.frqs, rtol=1e-12)

        Hs = (torch.fft.rfft(cqt_scalar.bwd(cqt_scalar.fwd(x)))[..., :Ls].squeeze() / F).numpy()
        Hl = (torch.fft.rfft(cqt_list.bwd(cqt_list.fwd(x)))[..., :Ls].squeeze() / F).numpy()
        h_ok = np.allclose(Hs, Hl, atol=1e-10)

        ok = frqs_ok and h_ok
        failures += not ok
        print(f"{mode:16s}: scalar vs [b]*numocts -> frqs match={frqs_ok} "
              f"transfer match={h_ok} {'OK' if ok else 'FAIL'}")
    print(f"uniform equivalence: {'OK' if not failures else f'{failures} FAILED'}\n")
    return failures


def main():
    failures = 0
    failures += test_geometry()
    failures += test_tie_regression()
    failures += test_transfer_function()
    failures += test_oct_mode_snr()
    failures += test_uniform_equivalence()

    if failures:
        print(f"{failures} check(s) FAILED.")
        sys.exit(1)
    print("all variable-binsoct checks OK")


if __name__ == "__main__":
    main()
