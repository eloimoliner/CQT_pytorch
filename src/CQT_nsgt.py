import torch 
#from src.nsgt.cq  import NSGT

from src.nsgt.fscale  import LogScale 

from src.nsgt.nsgfwin import nsgfwin
from src.nsgt.nsdual import nsdual
from src.nsgt.nsgtf import nsgtf
from src.nsgt.nsigtf import nsigtf
from src.nsgt.util import calcwinrange

import math

def next_power_of_2(x):
    return 1 if x == 0 else 2**math.ceil(math.log2(x))

class CQT_nsgt():
    def __init__(self,numocts, binsoct, mode="matrix",fs=44100, audio_len=44100, device="cpu"):
        fmax=fs/2 #the maximum frequency is Nyquist
        self.Ls=audio_len #the length is given

        fmin=fmax/(2**numocts)
        fbins=int(binsoct*numocts) 
       
        self.scale = LogScale(fmin, fmax, fbins)
        self.fs=fs

        self.device=torch.device(device)

        self.frqs,self.q = self.scale()


        self.g,rfbas,self.M = nsgfwin(self.frqs, self.q, self.fs, self.Ls, dtype=torch.float32, device=self.device)

        sl = slice(0,len(self.g)//2+1)

        # coefficients per slice
        self.ncoefs = max(int(math.ceil(float(len(gii))/mii))*mii for mii,gii in zip(self.M[sl],self.g[sl]))        
        if mode=="matrix":
            #just use the maximum resolution everywhere
            self.M[:] = self.M.max()
        elif mode=="oct":
            #round uo all the lengths of an octave to the next power of 2
            idx=2
            for i in range(numocts):
                value=next_power_of_2(self.M[idx:idx+binsoct].max())
                #value=M[idx:idx+binsoct].max()
                self.M[idx:idx+binsoct]=value
                self.M[-idx-binsoct+1:-idx+1]=value
                idx+=binsoct

        # calculate shifts
        self.wins,self.nn = calcwinrange(self.g, rfbas, self.Ls, device=self.device)
        # calculate dual windows
        self.gd = nsdual(self.g, self.wins, self.nn, self.M, device=self.device)

        self.forward = lambda s: nsgtf(s, self.g, self.wins, self.nn, self.M, mode=mode , device=self.device)
        self.backward = lambda c: nsigtf(c, self.gd, self.wins, self.nn, self.Ls, mode=mode,  device=self.device)


    def fwd(self,x):
        s=x.unsqueeze(0)
        c = self.forward(s)
        return c

    def bwd(self,c):
        s = self.backward(c) #messing out with the channels agains...
        return s.squeeze(0)

