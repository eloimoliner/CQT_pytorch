import os
import sys
import torch
import torchaudio
import numpy as np
from src.CQT_nsgt import CQT_nsgt
import soundfile as sf
import math
import time
import glob
import plotly

from torch.profiler import tensorboard_trace_handler
import wandb

import plotly.express as px

torch.set_printoptions(linewidth=2000, threshold=170000)
np.set_printoptions(threshold=sys.maxsize)

wandb.login()
#STEREO example
example="test_dir/0.wav"
#example="/scratch/work/molinee2/projects/ddpm/diffusion_autumn_2022/CQTdiff/test_dir/22050_long/MIDI-Unprocessed_02_R1_2009_03-06_ORIG_MID--AUDIO_02_R1_2009_02_R1_2009_04_WAV.wav"
a,fs=sf.read(example)
#a=0.5*(a[...,0]+a[...,1])
#resample=torch.resample()
a=a.transpose()
x=torch.Tensor(a)
#x=torchaudio.functional.resample(x, 44100, 22050)

x=x.unsqueeze(0).repeat(2,2,1) #simulate batch size of 2 and stereo
print(x.shape)

numocts=8
binsoct=64

Ls=131072 # most efficient one
cqt=CQT_nsgt(numocts, binsoct, mode="matrix",fs=fs, audio_len=Ls)

x=x[...,0:Ls]

#set profiler
wait, warmup, active, repeat = 10, 10, 2, 1
total_steps = (wait + warmup + active) * (1 + repeat)
schedule =  torch.profiler.schedule(
wait=wait, warmup=warmup, active=active, repeat=repeat)
profiler = torch.profiler.profile(
schedule=schedule, on_trace_ready=tensorboard_trace_handler("wandb/latest-run/tbprofile"), profile_memory=True, with_stack=False)

#run=wandb.init(project="trace")

for i in range(100):
    #x=x[...,44100:(44100+Ls)]
    #X=forward(x)
    x.requires_grad_()
    X=cqt.fwd(x)

    #xrec=backward(X)
    #xrec=nsigtf(X, cqt.gd, cqt.wins, cqt.nn, Ls=Ls, mode=cqt.mode, device=cqt.device)
    xrec=cqt.bwd(X)

    loss=torch.mean(torch.abs(xrec-x))
    out=torch.autograd.grad(outputs=loss, inputs=x)
    print((xrec-x).mean())
    A=torch.stft(x[1,1], 1024)
    Arec=torch.stft(xrec[1,1], 1024)
    error=A[:,:,:]-Arec[:,:,:]
    Error=torch.sqrt(error[...,1]**2+error[...,0]**2).squeeze(0)
    Error=10*torch.log10(Error)

    profiler.step()


profile_art = wandb.Artifact(f"trace-{wandb.run.id}", type="profile")
profile_art.add_file(glob.glob("wandb/latest-run/tbprofile/*.pt.trace.json")[0], "trace.pt.trace.json")
run.log_artifact(profile_art)