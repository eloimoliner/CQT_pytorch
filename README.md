# CQT_pytorch

Pytorch implementation of the invertible CQT based on Non-stationary Gabor filters.

The transform has near-perfect reconstruction, is differentiable and GPU-efficient.

## Install

```bash
pip install cqt-nsgt-pytorch
```
## Usage
```py
from cqt_nsgt_pytorch import CQT_nsgt

#parameter examples
numocts=9 
binsoct=64
fs=44100 
Ls=131072 

cqt=CQT_nsgt(numocts, binsoct, mode="matrix_complete",fs=fs, audio_len=Ls, device="cuda", dtype=torch.float32)

audio=#load some audio file shape=[Batch, channels, time]

X=cqt.fwd(audio)# forward transform
#X.shape=[batch, channels, frequency, time]
audio_reconstructed=cqt.bwd(X) #backward transform

```
## Modes of operation

Different versions of the transform are implemented. They can be selected by choosing the 'mode' parameter. Except "matrix" and "oct, that discard DC and Nyquist bands, the rest have perfect reconstruction.

mode          | Description  |  Output shape
------------- | ------------- | -------------
"critical"      | (default) critical sampling (no redundancy) (slow implementation) |  list of tensors, each with different time resolution 
"matrix"      |  Equal time resolution per frequency band. maximum redundancy (discards DC and Nyquist) | 2d-Tensor \[binsoct \times numocts, T\]
"matrix_complete  | Same as above, but DC and Nyquist are included.  | 2d-Tensor \[binsoct \times numocts + 2, T\]
"matrix_slow" | Slower version of "matrix_complete". Might show similar efficiency in CPU and consumes way less memory | 2d-Tensor \[binsoct \times numocts + 2, T\]
"oct" | Tradeoff between structure and redundancy. THe frequency bins are grouped by octave bands, each octave with a different time resolution. The time lengths are restricted to be powers of 2. (Discards DC and Nyquist) | list of tensors, one per octave band, each with different time resolution
"oct_complete" | Same as above, but DC and Nyquist are included | list of tensors, one per octave band,DC and Nyquist, each with a different time resolution



## Performance (`oct` mode)

`oct` avoids materialising the dense `[batch, channels, all_bins, Ls/2+1]` intermediate: the
forward gathers each octave's spectral support directly and the inverse overlap-adds with a
single `index_add`, so peak memory stays flat in batch size.

Tesla T4, `numocts=8, binsoct=32, fs=44100, Ls=262144`, batch 4:

| stage      | time    |
| ---------- | ------- |
| forward    | ~1.0 ms |
| inverse    | ~2.0 ms |
| round trip | ~2.9 ms |
| peak memory | ~0.5 GB |

Absolute times vary by GPU; the FFTs are the floor.

## Mixed precision

`torch.fft` does not support fp16/bf16, so the transform always runs internally in its real
float dtype (`float32`/`float64`) with autocast disabled. You can therefore call it safely
inside a `torch.autocast` region. keep the CQT in float32 and let autocast handle the
surrounding network.

```py
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    X = cqt.fwd(audio.float())   # CQT stays fp32; cast its complex output as needed
```

NumPy 2.x and Apple Silicon (MPS) are supported.

## TODO
- [x] On "matrix" mode, give the option to output also the DC and Nyq. Same in "oct" mode. Document how this disacrding thing is implemented.
- [ ] Do some proper documentation
- [x] Test it for mixed precision. The transform runs in fp32 internally and is safe to call inside `torch.autocast` (see Mixed precision).
- [ ] Make the apply_hpf_DC() and apply_lpf_DC() more handy and clear. Document the usage of those.
- [ ] Accelerate the "critical" mode, similar method as in "oct" could also apply. (update: seems a bit tricky memory-wise)
- [ ] Clean the whole __init__() method as now it is a mess. 
- [x] Report the efficiency of the implementation in GPU. (time and frequency). See Performance (`oct` mode).
- [x] Check if there is more redundancy to get rid of. Apparently, there is not
