# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "torch>=1.13.0",
#     "numpy>=1.19.5",
#     "soundfile>=0.12",
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
Round-trip every audio_examples/*.wav through the CQT and write the residual
x_hat - x to a wav, so it can be inspected/auditioned directly.

    x -> CQT -> X -> iCQT -> x_hat,   save (x_hat - x)

Runs in float32 (real-world audio precision, not float64) so the residual reflects
what you actually get in practice. For the non-complete "oct" mode (which discards the
DC and Nyquist bands) the discarded content is added back with apply_lpf_DC, so the
residual is a fair full-reconstruction error rather than just the missing DC/Nyquist.

For each file/mode it writes two wavs into audio_examples/residuals/:
  * <name>_<mode>_residual.wav       -- the raw residual, same scale as the signal
  * <name>_<mode>_residual_norm.wav  -- residual peak-normalised to -1 dBFS so it is
                                        actually audible (prints the applied gain in dB)
and prints SNR and peak sample error. A clean residual is inaudible / noise-floor only.

Run with:
    uv run tests/save_residual_wavs.py
"""
import os
import sys

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from cqt_nsgt_pytorch import CQT_nsgt

NUMOCTS = 8
BINSOCT = 32
MODES = ("oct_complete", "matrix_complete", "oct")   # oct = non-complete (uses lpf add-back)
IN_DIR = os.path.join(REPO_ROOT, "audio_examples")
OUT_DIR = os.path.join(IN_DIR, "residuals")


def snr_db(ref, rec):
    noise = float(((rec - ref) ** 2).sum())
    return 10 * np.log10(float((ref ** 2).sum()) / max(noise, 1e-30))


def signals():
    """Yield (name, x[1,C,T], fs). Fullband white noise first -- it excites every band
    including DC and the top octave, which music barely touches, so it is the real test
    for a missing/leaking band. Even and odd lengths both, to exercise the grid padding.
    Then every audio_examples/*.wav."""
    torch.manual_seed(0)
    for tag, Ls in (("noise_even", 262144), ("noise_odd", 262143)):
        yield tag, torch.randn(1, 1, Ls, dtype=torch.float32) * 0.1, 44100
    for fn in sorted(f for f in os.listdir(IN_DIR) if f.lower().endswith(".wav")):
        data, fs = sf.read(os.path.join(IN_DIR, fn), dtype="float32", always_2d=True)
        yield os.path.splitext(fn)[0], torch.from_numpy(data.T).unsqueeze(0), fs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for name, x, fs in signals():
        Ls = x.shape[-1]

        for mode in MODES:
            cqt = CQT_nsgt(NUMOCTS, BINSOCT, mode=mode, fs=fs, audio_len=Ls,
                           device="cpu", dtype=torch.float32)
            x_hat = cqt.bwd(cqt.fwd(x))[..., :Ls]
            if mode in ("oct", "matrix"):           # add back the discarded DC/Nyquist bands
                x_hat = x_hat + cqt.apply_lpf_DC(x)[..., :Ls]

            resid = (x_hat - x).squeeze(0).cpu().numpy().T   # [T, C]
            xnp = x.squeeze(0).cpu().numpy().T

            snr = snr_db(xnp, x_hat.squeeze(0).cpu().numpy().T)
            peak = float(np.abs(resid).max())
            sig_peak = float(np.abs(xnp).max())

            # where does the residual energy live in frequency? A missing/leaking band
            # would pile up at one place (e.g. DC or Nyquist); float round-off is flat.
            # Reported as the fraction of Nyquist where |residual spectrum| peaks.
            R = np.abs(np.fft.rfft(resid[:, 0]))
            worst = R.argmax() / (len(R) - 1)

            raw_path = os.path.join(OUT_DIR, f"{name}_{mode}_residual.wav")
            sf.write(raw_path, resid, fs, subtype="FLOAT")

            gain = (10 ** (-1 / 20)) / max(peak, 1e-30)       # normalise residual peak to -1 dBFS
            norm_path = os.path.join(OUT_DIR, f"{name}_{mode}_residual_norm.wav")
            sf.write(norm_path, np.clip(resid * gain, -1.0, 1.0), fs, subtype="FLOAT")

            print(f"{name:10s} {mode:15s}: SNR {snr:6.1f} dB | "
                  f"peak {peak:.2e} ({20*np.log10(max(peak,1e-30)/max(sig_peak,1e-30)):6.1f} dB below sig) | "
                  f"resid spectrum peaks @ {worst:.2f}xNyq")

    print(f"\nwrote residual wavs to {OUT_DIR}")
    print("raw = true scale (near-silent if clean); _norm = amplified to be audible")


if __name__ == "__main__":
    main()
