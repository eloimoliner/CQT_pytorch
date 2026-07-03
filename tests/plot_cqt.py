# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "torch>=1.13.0",
#     "numpy>=1.19.5",
#     "soundfile>=0.12",
#     "matplotlib>=3.5",
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
Plot the CQT magnitude of the audio_examples as log-frequency spectrograms.

Uses `matrix_complete` mode, which returns a dense [freq, time] tensor that maps
straight onto an image. Frequencies run low -> high up the y-axis; the axis is
labelled with the actual center frequencies from the CQT scale.

Writes one PNG per example into audio_examples/plots/.

Run with:
    uv run tests/plot_cqt.py
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from cqt_nsgt_pytorch import CQT_nsgt

NUMOCTS = 8
BINSOCT = 32
OUT_DIR = os.path.join(REPO_ROOT, "audio_examples", "plots")


def load_mono(path):
    data, fs = sf.read(path, dtype="float32", always_2d=True)  # [time, ch]
    mono = data.mean(axis=1)                                    # collapse channels
    return torch.from_numpy(mono).view(1, 1, -1), fs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    wavs = sorted(glob.glob(os.path.join(REPO_ROOT, "audio_examples", "*.wav")))
    assert wavs, "no wav files in audio_examples/"

    for path in wavs:
        x, fs = load_mono(path)
        Ls = x.shape[-1]
        cqt = CQT_nsgt(NUMOCTS, BINSOCT, mode="matrix_complete", fs=fs,
                       audio_len=Ls, device="cpu", dtype=torch.float32)
        X = cqt.fwd(x)                          # [1, 1, fbins+2, T]
        mag = X.abs().squeeze().cpu().numpy()   # [fbins+2, T]
        db = 20.0 * np.log10(mag + 1e-8)
        db = np.clip(db, db.max() - 80.0, db.max())   # 80 dB dynamic range

        # y-axis center freqs: DC, the fbins scale freqs, then Nyquist
        frqs = np.asarray(cqt.frqs)
        yf = np.concatenate(([0.0], frqs, [fs / 2]))

        fig, ax = plt.subplots(figsize=(9, 4))
        im = ax.imshow(db, origin="lower", aspect="auto", cmap="magma",
                       extent=[0, Ls / fs, 0, len(yf)])
        # label a handful of rows with their real frequency
        ticks = np.linspace(0, len(yf) - 1, 9).astype(int)
        ax.set_yticks(ticks + 0.5)
        ax.set_yticklabels([f"{yf[t]:.0f}" for t in ticks])
        ax.set_xlabel("time (s)")
        ax.set_ylabel("frequency (Hz)")
        name = os.path.basename(path)
        ax.set_title(f"CQT |X| (dB)  {name}  "
                     f"[{NUMOCTS} oct x {BINSOCT} bins, shape={tuple(X.shape[-2:])}]")
        fig.colorbar(im, ax=ax, label="dB")
        fig.tight_layout()

        out = os.path.join(OUT_DIR, os.path.splitext(name)[0] + "_cqt.png")
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"{name}: X.shape={tuple(X.shape)} -> {out}")

    print(f"\nwrote {len(wavs)} plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
