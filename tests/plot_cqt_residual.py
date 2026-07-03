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
Visualize the CQT-domain reconstruction residual:

    x -> CQT -> X -> iCQT -> x_hat -> CQT -> X_hat,   plot |X - X_hat|

Both |X| and |X - X_hat| are shown in dB on the SAME color scale, so the
residual panel reads directly as "error relative to signal". Runs on piano.wav
(if present) and on white noise (no files needed); noise exposes error at
frequencies where music has no energy.

Writes PNGs into audio_examples/plots/.

Run with:
    uv run tests/plot_cqt_residual.py
"""
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
FS = 44100
OUT_DIR = os.path.join(REPO_ROOT, "audio_examples", "plots")


def signals():
    piano = os.path.join(REPO_ROOT, "audio_examples", "piano.wav")
    if os.path.exists(piano):
        data, fs = sf.read(piano, dtype="float32", always_2d=True)
        yield "piano", torch.from_numpy(data.mean(axis=1)).view(1, 1, -1), fs
    torch.manual_seed(0)
    yield "white_noise", torch.randn(1, 1, 262144) * 0.1, FS


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, x, fs in signals():
        Ls = x.shape[-1]
        cqt = CQT_nsgt(NUMOCTS, BINSOCT, mode="matrix_complete", fs=fs,
                       audio_len=Ls, device="cpu", dtype=torch.float32)

        X = cqt.fwd(x)
        x_hat = cqt.bwd(X)[..., :Ls]
        X_hat = cqt.fwd(x_hat)

        mag = X.abs().squeeze().cpu().numpy()
        res = (X - X_hat).abs().squeeze().cpu().numpy()

        ref = mag.max()
        top = 20 * np.log10(ref + 1e-12)
        lo = top - 160.0  # shared 160 dB scale so residual reads as error-vs-signal
        mag_db = np.clip(20 * np.log10(mag + 1e-12), lo, top)
        res_db = np.clip(20 * np.log10(res + 1e-12), lo, top)

        frqs = np.asarray(cqt.frqs)
        yf = np.concatenate(([0.0], frqs, [fs / 2]))
        ticks = np.linspace(0, len(yf) - 1, 9).astype(int)

        fig = plt.figure(figsize=(12, 7))
        gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1])
        axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]),
               fig.add_subplot(gs[:, 1])]
        for ax, img, title in ((axs[0], mag_db, r"$|X|$"),
                               (axs[1], res_db, r"residual $|X-\hat{X}|$")):
            im = ax.imshow(img, origin="lower", aspect="auto", cmap="magma",
                           vmin=lo, vmax=top, extent=[0, Ls / fs, 0, len(yf)])
            ax.set_yticks(ticks + 0.5)
            ax.set_yticklabels([f"{yf[t]:.0f}" for t in ticks])
            ax.set_ylabel("frequency (Hz)")
            ax.set_title(f"{title}  (dB, shared scale)")
            fig.colorbar(im, ax=ax, label="dB")
        axs[1].set_xlabel("time (s)")

        # per-bin profile: time-averaged |X| and |X-X_hat| — makes single-row
        # structure (e.g. the Nyquist band) visible
        bins = np.arange(mag.shape[0])
        axs[2].plot(20 * np.log10(mag.mean(1) + 1e-12), bins, label=r"$|X|$")
        axs[2].plot(20 * np.log10(res.mean(1) + 1e-12), bins,
                    label=r"$|X-\hat{X}|$")
        axs[2].set_ylim(0, mag.shape[0] - 1)
        axs[2].set_xlabel("time-mean magnitude (dB)")
        axs[2].set_ylabel("bin (0=DC, last=Nyquist)")
        axs[2].legend(loc="lower left", fontsize=8)
        axs[2].grid(alpha=0.3)
        axs[2].set_title("per-bin profile")
        peak_err = 20 * np.log10(res.max() / ref)
        fig.suptitle(f"CQT round-trip residual  {name}  "
                     f"[matrix_complete, {NUMOCTS} oct x {BINSOCT}]  "
                     f"peak residual {peak_err:.1f} dB rel. |X| peak")
        fig.tight_layout()

        out = os.path.join(OUT_DIR, f"{name}_cqt_residual.png")
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"{name}: peak residual {peak_err:.1f} dB below signal peak -> {out}")


if __name__ == "__main__":
    main()
