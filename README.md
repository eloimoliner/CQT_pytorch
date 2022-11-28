# CQT_pytorch

Pytorch implementation of the invertible CQT based on Non-stationary Gabor filters

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

cqt=CQT_nsgt(numocts, binsoct, mode="matrix_complete",fs=fs, audio_len=Ls, dtype=torch.float32)

audio=#load some audio file

X=cqt.fwd(audio)# forward transform

audio_reconstructed=cqt.bwd(X) #backward transform

```
## TODO
- [ ] On "matrix" mode, give the option to output also the DC and Nyq. Same in "oct" mode. Document how this disacrding thing is implemented.
- [ ] Test it for mixed precision. problems with powers of 2, etc. Maybe this will require zero padding...
- [ ] Make the apply_hpf_DC() and apply_lpf_DC() more handy and clear. Document the usage of those.
- [ ] Accelerate the "critical" mode, similar method as in "oct" could also apply
- [ ] Clean the whole __init__() method as now it is a mess
- [ ] Report the efficiency of the implementation in GPU. (time and frequency). Briefly: It is fast as everything is vectorized but maybe consumes too much memory, specially on the backward pass.
- [ ] Check if there is more redundancy to get rid of. 
