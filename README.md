# CQT_pytorch

PyTorch implementation of an invertible Constant-Q Transform (CQT) built on Non-Stationary
Gabor (NSGT) filters.

- **Perfect reconstruction** — the `*_complete` modes round-trip to the floating-point floor
  (~1e-13 in float64, ~1e-7 in float32); the other modes recover the full signal when paired
  with the DC/Nyquist side-channel (see below).
- **Differentiable** — gradients flow through both the forward and inverse transforms
  (verified against finite differences).
- **Fast & memory-lean** — every mode avoids the dense `[B, C, all_bins, Ls/2+1]` intermediate,
  so peak memory stays roughly flat in batch size.
- **Any audio length** — odd, prime, or non-power-of-two lengths all reconstruct correctly
  (the transform pads an internal grid and truncates back).
- **Variable resolution** — bins-per-octave can be set independently per octave in a single
  transform.
- **Mixed-precision friendly** — safe to call inside `torch.autocast`; NumPy 2.x and Apple
  Silicon (MPS) supported.

## Install

> **Note:** the version on PyPI is **outdated** and does *not* include the features documented
> here (perfect-reconstruction fixes, arbitrary lengths, variable bins-per-octave, the
> accelerated modes, etc.). Install from source for the current version:

```bash
pip install git+https://github.com/eloimoliner/CQT_pytorch.git
# or, for development:
git clone https://github.com/eloimoliner/CQT_pytorch.git && pip install -e CQT_pytorch
```

The stale PyPI package (`pip install cqt-nsgt-pytorch`) still works but lags behind this repo.

## Usage

```py
import torch
from cqt_nsgt_pytorch import CQT_nsgt

numocts   = 8          # number of octaves (fmin = Nyquist / 2**numocts)
binsoct   = 32         # bins per octave (int, or a list of length numocts)
fs        = 44100
audio_len = 262144     # maximum input length

cqt = CQT_nsgt(numocts, binsoct, mode="oct_complete",
               fs=fs, audio_len=audio_len, device="cuda", dtype=torch.float32)

audio = ...                    # [batch, channels, time]
X     = cqt.fwd(audio)         # forward transform
audio_hat = cqt.bwd(X)         # inverse transform  ->  audio_hat ≈ audio
```

On instantiation the transform prints its characteristics (disable with `verbose=False`):

```
CQT_nsgt  |  mode=oct_complete  |  fs=44100 Hz
  octaves=8   bins/oct=32 (uniform)   total log bins=256
  fmin=86.133 Hz   fmax(Nyquist)=22050.000 Hz
  audio_len=262144 == internal grid nn=262144   (no padding)
  oct |   freq range (Hz)   | bins | frames | coeff rate (Hz)
  ----+---------------------+------+--------+----------------
    0 |     86.1 -    172.3 |   32 |     64 |           10.77
    1 |    172.3 -    344.5 |   32 |    128 |           21.53
   ...
    7 |  11025.0 -  22050.0 |   32 |   8192 |         1378.12
  + DC band (0 - 86.1 Hz) and Nyquist band (22050 Hz)
```

`frames` is each octave's coefficient time dimension (a power of two in `oct`/`oct_complete`);
`coeff rate` is that band's effective sample rate. You can also call `cqt.describe()` any time.

## Modes of operation

Select with the `mode` argument. The `*_complete` modes carry the DC and Nyquist bands and give
perfect reconstruction on their own; the others drop DC/Nyquist (recover them with the
side-channel below).

| mode | description | output |
|------|-------------|--------|
| `oct` | *(default)* octave-wise; each octave a tensor with its own (power-of-two) time resolution. Discards DC/Nyquist. | list of per-octave tensors |
| `oct_complete` | as `oct`, plus DC and Nyquist bands. Perfect reconstruction. | list of per-octave tensors + DC + Nyquist |
| `matrix` | one dense 2-D matrix, equal time resolution for every band (max redundancy). Discards DC/Nyquist. | `[B, C, binsoct·numocts, T]` |
| `matrix_pow2` | as `matrix`, time dimension rounded up to a power of two. | `[B, C, binsoct·numocts, T]` |
| `matrix_complete` | as `matrix`, plus DC and Nyquist. Perfect reconstruction. | `[B, C, binsoct·numocts + 2, T]` |

`oct` is the default: it is the most memory-efficient and gives per-octave power-of-two latent
shapes (convenient for U-Nets and the like). Use `oct_complete` if you want DC/Nyquist inside
the coefficient tensor and lossless round-trip with no extra steps.

## DC and Nyquist as a side-channel (`oct` / `matrix`)

The non-complete modes leave out the DC and Nyquist bands. To reconstruct the full signal, add
them back with `apply_lpf_DC` (which returns the DC+Nyquist content as a full-rate time-domain
signal); `apply_hpf_DC` is its complement:

```py
X    = cqt.fwd(x)                       # oct mode: mid bands only
xhat = cqt.bwd(X) + cqt.apply_lpf_DC(x) # full reconstruction (≈ perfect)
```

`apply_hpf_DC(x) + apply_lpf_DC(x) == x` exactly, and this workflow reconstructs to the
numerical floor. It also has a practical bonus: the round-trip **residual is white** (uniform
across frequency), whereas the `*_complete` modes concentrate float32 round-off in the lowest
octave (the DC band's long window). If you need a white residual in float32, prefer
`oct`/`matrix` + `apply_lpf_DC`; if you need DC/Nyquist inside the tensor, use a `*_complete`
mode (optionally in float64 for a white residual).

## Variable bins-per-octave

`binsoct` may be a list of length `numocts` (index 0 = lowest octave, closest to DC), giving
each octave its own frequency resolution in a single transform — no need to stitch several CQTs
together:

```py
numocts = 9
binsoct = [8, 8, 8, 16, 16, 16, 16, 32, 32]   # coarser (better time resolution) near DC

cqt = CQT_nsgt(numocts, binsoct, mode="oct_complete", fs=fs, audio_len=Ls)
```

Paired with `oct`/`oct_complete` (which already give per-octave time resolution) this yields
independent time *and* frequency resolution per octave. Total bins become `sum(binsoct)`.

## Arbitrary audio lengths

`audio_len` need not be a power of two. Internally the transform runs on a grid padded to what
the mode requires (even length for the Nyquist bin; a multiple of the coarsest octave length for
the octave modes) and truncates the inverse back to the true length. For a power-of-two length
the padding is a no-op. `describe()` reports whether padding is applied and how much.

Note: asking for more octaves/bins than the segment can resolve (e.g. `numocts=12`, where the
lowest octaves fall below the `min_win=4` window floor) degrades gracefully but is not perfect —
keep `fmin = fs / 2**(numocts+1)` comfortably above a few Hz for audio.

## Performance

Every mode gathers each band's spectral support directly and overlap-adds the inverse with a
single `index_add`, so none of them build the dense `[B, C, all_bins, Ls/2+1]` intermediate and
peak memory stays flat in batch size. Round-trip (CPU, `Ls=262144`, 8 oct × 32, float32):

| mode | B=1 | B=4 | B=8 |
|------|-----|-----|-----|
| `oct` | 38 ms / 0.68 GB | 57 ms / 0.78 GB | 90 ms / 0.89 GB |
| `oct_complete` | 113 ms / 0.68 GB | 205 ms / 0.75 GB | 280 ms / 0.85 GB |

For comparison, the previous dense `oct_complete` took ~1135 ms / 2.0 GB at B=4 — the current
version is ~5× faster and ~2.7× leaner, with memory that no longer grows with the batch. GPU is
substantially faster still; the FFTs are the floor.

The gather-then-multiply forward, `rfft`, and single-`index_add` inverse that make this possible
were contributed by [@cucuwritescode](https://github.com/cucuwritescode) for `oct` mode
([PR #7](https://github.com/eloimoliner/CQT_pytorch/pull/7)); they are now applied to every mode.

## Mixed precision

`torch.fft` has no fp16/bf16 kernels, so the transform always runs internally in its real float
dtype (`float32`/`float64`) with autocast disabled. It is therefore safe to call inside a
`torch.autocast` region without losing precision:

```py
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    X = cqt.fwd(audio.float())   # CQT stays float32; cast its complex output as you like
```

fp16/bf16 inputs are upcast internally (reconstruction is then limited by the input's
precision). bf16 has full float32 range; fp16 has more mantissa but a limited range (watch for
overflow if you store coefficients in fp16 at very high amplitude).

## Tests

The `tests/` directory holds self-contained `uv` scripts (perfect-reconstruction / transfer
function, arbitrary lengths, variable bins-per-octave incl. a randomised stress test, the
DC/Nyquist split, and reconstruction from real audio). Run e.g.:

```bash
uv run tests/test_transfer.py
uv run tests/test_variable_binsoct.py
```

## Acknowledgements

Thanks to [@cucuwritescode](https://github.com/cucuwritescode) for the efficiency optimization
([PR #7](https://github.com/eloimoliner/CQT_pytorch/pull/7)) — the gather-then-multiply forward,
`rfft`, and single-`index_add` inverse that keep peak memory flat in batch size. Originally for
`oct` mode, the same approach now accelerates every mode of the transform.
