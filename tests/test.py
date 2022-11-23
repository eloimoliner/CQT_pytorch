import os
import sys
import torch
import torchaudio
import numpy as np
from cqt_nsgt_pytorch import CQT_nsgt
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

x=x.unsqueeze(0).repeat(2,1,1) #simulate batch size of 2 and stereo
print(x.shape)

numocts=9 
binsoct=64

#oct fails whe numocts=7, binsoct=64

Ls=131072 # most efficient one
cqt=CQT_nsgt(numocts, binsoct, mode="oct",fs=fs, audio_len=Ls, dtype=torch.float32)

x=x[...,0:Ls].to(torch.float32)

#set profiler
wait, warmup, active, repeat = 10, 10, 2, 1
total_steps = (wait + warmup + active) * (1 + repeat)
schedule =  torch.profiler.schedule(
wait=wait, warmup=warmup, active=active, repeat=repeat)
profiler = torch.profiler.profile(
schedule=schedule, on_trace_ready=tensorboard_trace_handler("wandb/latest-run/tbprofile"), profile_memory=True, with_stack=False)

#run=wandb.init(project="trace")
def apply_low_pass_firwin(y,filter):
    """
        Utility for applying a FIR filter, usinf pytorch conv1d
        Args;
            y (Tensor): shape (B,T) signal to filter
            filter (Tensor): shape (1,1,order) FIR filter coefficients
        Returns:
            y_lpf (Tensor): shape (B,T) filtered signal
    """

    #ii=2
    B=filter.to(y.device)
    #B=filter
    #y=y.unsqueeze(1)
    #weight=torch.nn.Parameter(B)
    
    y_lpf=torch.nn.functional.conv1d(y,B,padding="same")
    #y_lpf=y_lpf.squeeze(1) #some redundancy here, but its ok
    #y_lpf=y
    return y_lpf

def process_stft(A):
    A=torch.sqrt(A[...,1]**2+A[...,0]**2).squeeze(0)
    A=10*torch.log10(A)
    fig=px.imshow(A)
    fig.show()

for i in range(100):
    #x=x[...,44100:(44100+Ls)]
    #X=forward(x)
    xhpf=cqt.apply_hpf_DC(x)

    xlpf=cqt.apply_lpf_DC(x)

    X=cqt.fwd(x)
    #X[0]=X[0]*0
    #X[-1]=X[-1]*0

    #xrec=backward(X)
    #xrec=nsigtf(X, cqt.gd, cqt.wins, cqt.nn, Ls=Ls, mode=cqt.mode, device=cqt.device)
    xrec=cqt.bwd(X)

    print("all error",(xrec-x).abs().sum())
    print("error respect xhpf",(xrec-xhpf).abs().sum())
    print("error corrected with xlpf",(xrec+xlpf-x).abs().sum())
    error_hpf=cqt.apply_hpf_DC(xrec-x)
    print("filtered error",error_hpf.abs().sum())
    #print(error_hpf.abs().sum())

    E=torch.stft((xrec+xlpf-x)[1,0],1024)
    AA=torch.stft(x[1,0], 1024)
    Arec=torch.stft(xrec[1,0], 1024)
    Ahpf=torch.stft(xhpf[1,0], 1024)
    Alpf=torch.stft(xlpf[1,0], 1024)
    EE=torch.stft(error_hpf[1,0], 1024)
    error=AA[:,:,:]-Arec[:,:,:]
    process_stft(EE)
    
    profiler.step()
    


profile_art = wandb.Artifact(f"trace-{wandb.run.id}", type="profile")
profile_art.add_file(glob.glob("wandb/latest-run/tbprofile/*.pt.trace.json")[0], "trace.pt.trace.json")
run.log_artifact(profile_art)