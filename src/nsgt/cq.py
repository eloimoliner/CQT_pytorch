
from .nsgfwin import nsgfwin
from .nsdual import nsdual
from .nsgtf import nsgtf
from .nsigtf import nsigtf
from .util import calcwinrange
from math import ceil
import torch
import math


class NSGT:
    def __init__(self, scale, fs, Ls, real=True, mode="matrix",numocts=7, binsoct=64, reducedform=0, multichannel=False,  dtype=torch.float32, device="cpu"):
        assert fs > 0
        assert Ls > 0
        assert 0 <= reducedform <= 2

        self.scale = scale
        self.fs = fs
        self.Ls = Ls
        self.real = real
        self.mode=mode
        self.reducedform = reducedform

        self.device = torch.device(device)
        
        self.frqs,self.q = scale()

        # calculate transform parameters
        self.g,rfbas,self.M = nsgfwin(self.frqs, self.q, self.fs, self.Ls, sliced=False, dtype=dtype, device=self.device)

        sl = slice(0,len(self.g)//2+1)

        # coefficients per slice
        self.ncoefs = max(int(ceil(float(len(gii))/mii))*mii for mii,gii in zip(self.M[sl],self.g[sl]))        

        def next_power_of_2(x):
            return 1 if x == 0 else 2**math.ceil(math.log2(x))

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
    
        if multichannel:
            self.channelize = lambda s: s
            self.unchannelize = lambda s: s
        else:
            self.channelize = lambda s: (s,)
            self.unchannelize = lambda s: s[0]

        # calculate shifts
        self.wins,self.nn = calcwinrange(self.g, rfbas, self.Ls, device=self.device)
        # calculate dual windows
        self.gd = nsdual(self.g, self.wins, self.nn, self.M, device=self.device)
        
        self.fwd = lambda s: nsgtf(s, self.g, self.wins, self.nn, self.M, mode=mode , device=self.device)
        self.bwd = lambda c: nsigtf(c, self.gd, self.wins, self.nn, self.Ls, mode=mode,  device=self.device)
        
    @property
    def coef_factor(self):
        return float(self.ncoefs)/self.Ls
    
    @property
    def slice_coefs(self):
        return self.ncoefs
    
    def forward(self, s):
        'transform'
        s = self.channelize(s)
        #c = list(map(self.fwd, s))

        #I'm messing out with the channels here...
        s=torch.unsqueeze(s[0], dim=0)
        c = self.fwd(s)
        #return self.unchannelize(c)
        return c

    def backward(self, c):
        'inverse transform'
        c = self.channelize(c)
        #s = list(map(self.bwd,c))
        s = self.bwd(c[0]) #messing out with the channels agains...
        return self.unchannelize(s)

    
class CQ_NSGT(NSGT):
    def __init__(self, fmin, fmax, bins, fs, Ls, real=True, matrixform=False, reducedform=0, multichannel=False, measurefft=False, multithreading=False):
        assert fmin > 0
        assert fmax > fmin
        assert bins > 0
        
        self.fmin = fmin
        self.fmax = fmax
        self.bins = bins

        scale = OctScale(fmin, fmax, bins)
        NSGT.__init__(self, scale, fs, Ls, real, matrixform=matrixform, reducedform=reducedform, multichannel=multichannel, measurefft=measurefft, multithreading=multithreading)
