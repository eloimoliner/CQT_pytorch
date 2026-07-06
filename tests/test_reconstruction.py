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
Load every wav in audio_examples/, run the CQT forward transform (printing the
shapes), invert it, and report the reconstruction error.

The wavs are not tracked in git (see .gitignore); drop your own audio into
audio_examples/ before running.

Run with:
    uv run tests/test_reconstruction.py
optionally choosing a mode:
    uv run tests/test_reconstruction.py --mode matrix_complete
"""
import argparse
import glob
import os
import sys

import numpy as np
import soundfile as sf
import torch

# make the package importable when run from anywhere
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from cqt_nsgt_pytorch import CQT_nsgt


def load_wav(path):
    """Return audio as [channels, time] float32 and the sample rate."""
    data, fs = sf.read(path, dtype="float32", always_2d=True)  # [time, channels]
    audio = torch.from_numpy(data.T).contiguous()  # [channels, time]
    return audio, fs


def reconstruction_metrics(ref, rec):
    """SNR (dB) and max abs error between two [.., T] tensors of equal length."""
    err = rec - ref
    noise = torch.sum(err ** 2)
    signal = torch.sum(ref ** 2)
    snr = 10.0 * torch.log10(signal / noise.clamp_min(1e-20))
    return snr.item(), err.abs().max().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="oct_complete,matrix_complete",
                        help="comma-separated CQT modes (see README). Defaults to the "
                             "two perfect-reconstruction workhorses. Use a "
                             "*_complete mode for perfect reconstruction.")
    parser.add_argument("--numocts", type=int, default=8)
    parser.add_argument("--binsoct", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32
    modes = [m.strip() for m in args.mode.split(",") if m.strip()]
    min_snr = 60.0  # perfect-reconstruction modes clear this by a wide margin
    print(f"device={device}  modes={modes}  numocts={args.numocts}  binsoct={args.binsoct}")

    wavs = sorted(glob.glob(os.path.join(REPO_ROOT, "audio_examples", "*.wav")))
    assert wavs, "no wav files found in audio_examples/"

    failures = 0
    for mode in modes:
        print(f"\n########## mode = {mode} ##########")
        snrs = []
        for path in wavs:
            audio, fs = load_wav(path)
            audio = audio.to(device=device, dtype=dtype)
            # CQT expects [batch, channels, time]
            x = audio.unsqueeze(0)  # [1, C, T]
            Ls = x.shape[-1]

            cqt = CQT_nsgt(args.numocts, args.binsoct, mode=mode,
                           fs=fs, audio_len=Ls, device=device, dtype=dtype)

            X = cqt.fwd(x)
            if isinstance(X, list):
                shape_str = "[" + ", ".join(str(tuple(t.shape)) for t in X) + "]"
            else:
                shape_str = str(tuple(X.shape))

            rec = cqt.bwd(X)  # [1, C, T]
            rec = rec[..., :Ls]

            snr, max_err = reconstruction_metrics(x, rec)
            snrs.append(snr)

            name = os.path.basename(path)
            print(f"\n{name}  fs={fs}  input={tuple(x.shape)}")
            print(f"    CQT output shape: {shape_str}")
            print(f"    reconstruction SNR: {snr:7.2f} dB   max abs err: {max_err:.3e}")

        mean_snr = float(np.mean(snrs))
        ok = min(snrs) >= min_snr
        failures += not ok
        print(f"\nmode {mode}: mean SNR {mean_snr:.2f} dB, "
              f"min {min(snrs):.2f} dB over {len(snrs)} files "
              f"-> {'OK' if ok else f'FAIL (<{min_snr} dB)'}")

    print()
    if failures:
        print(f"{failures}/{len(modes)} modes FAILED reconstruction.")
        sys.exit(1)
    print(f"all {len(modes)} modes OK")


if __name__ == "__main__":
    main()
