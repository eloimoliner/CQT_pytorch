import torch 
from src.nsgt_new.cq  import NSGT

from src.nsgt_new.fscale  import LogScale 


class CQT_nsgt():
    def __init__(self,numocts, binsoct, mode="matrix",fs=44100, audio_len=44100, device="cpu"):
        fmax=fs/2
        Ls=audio_len

        fmin=fmax/(2**numocts)
        fbins=int(binsoct*numocts) 
       
        scl = LogScale(fmin, fmax, fbins)
        if mode=="matrix":
            self.cq = NSGT(scl, fs, Ls,
                 mode="matrix", 
                 multichannel=False,
                 device=device
                 )
        elif mode=="oct":
            self.cq = NSGT(scl, fs, Ls, 
                     mode="oct", 
                     numocts=numocts,
                     binsoct=binsoct,
                     multichannel=False,
                     device=device
                     )
        else: #potentially mode="critical"
            self.cq = NSGT(scl, fs, Ls, 
                     mode=mode, 
                     numocts=numocts,
                     binsoct=binsoct,
                     multichannel=False,
                     device=device
                     )
        
    def fwd(self,x):
        c= self.cq.forward(x)
        #print(c.shape)
        return c

    def bwd(self,c):
        xrec= self.cq.backward(c)
        #print(xrec.shape)
        return xrec

