# CQT_pytorch


## Install

```bash
pip install cqt-nsgt-pytorch
```
## Usage
```py
from cqt_nsgt_pytorch import CQT_nsgt
```
##TODO
- [ ] On "matrix" mode, give the option to output also the DC and Nyq. Same in "oct" mode. Document how this disacrding thing is implemented.
- [ ] Test it for mixed precision. problems with powers of 2, etc. Maybe this will require zero padding...
- [ ] Make the apply_hpf_DC() and apply_lpf_DC() more handy and clear. Document the usage of those.
- [ ] Accelerate the "critical" mode, similar method as in "oct" could also apply
- [ ] Clean the whole __init__() method as now it is a mess
- [ ] Report the efficiency of the implementation in GPU. (time and frequency). Briefly: It is fast as everything is vectorized but maybe consumes too much memory, specially on the backward pass.
- [ ] Check if there is more redundancy to get rid of. 
