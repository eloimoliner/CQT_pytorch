import torch 
#from src.nsgt.cq  import NSGT

from src.nsgt.fscale  import LogScale 

from src.nsgt.nsgfwin import nsgfwin
from src.nsgt.nsdual import nsdual
from src.nsgt.util import calcwinrange

import math
from math import ceil

def next_power_of_2(x):
    return 1 if x == 0 else 2**math.ceil(math.log2(x))

class CQT_nsgt():
    def __init__(self,numocts, binsoct, mode="critical",fs=44100, audio_len=44100, device="cpu", dtype=torch.float32):
        """
            args:
                numocts (int) number of octaves
                binsoct (int) number of bins per octave
                mode (string) defines the mode of operation:
                     "critical": (default) critical sampling (no redundancy) returns a list of tensors, each with different time resolution
                     "matrix": returns a 2d-matrix maximum redundancy
                     "oct": octave-wise rasterization ( modearate redundancy) returns a list of tensors, each from a different octave with different time resolution
                fs (float) sampling frequency
                audio_len (int) sample length
                device
        """

        fmax=fs/2 -10**-6 #the maximum frequency is Nyquist
        self.Ls=audio_len #the length is given

        fmin=fmax/(2**numocts)
        fbins=int(binsoct*numocts) 
        self.numocts=numocts
        self.binsoct=binsoct
       
        self.scale = LogScale(fmin, fmax, fbins)
        self.fs=fs

        self.device=torch.device(device)
        self.mode=mode
        self.dtype=dtype

        self.frqs,self.q = self.scale() 

        self.g,rfbas,self.M = nsgfwin(self.frqs, self.q, self.fs, self.Ls, dtype=self.dtype, device=self.device, min_win=4)

        sl = slice(0,len(self.g)//2+1)

        # coefficients per slice
        self.ncoefs = max(int(math.ceil(float(len(gii))/mii))*mii for mii,gii in zip(self.M[sl],self.g[sl]))        
        if mode=="matrix":
            #just use the maximum resolution everywhere
            self.M[:] = self.M.max()
        elif mode=="oct":
            #round uo all the lengths of an octave to the next power of 2
            self.size_per_oct=[]
            idx=1
            for i in range(numocts):
                value=next_power_of_2(self.M[idx:idx+binsoct].max())
                #value=M[idx:idx+binsoct].max()
                self.size_per_oct.append(value)
                self.M[idx:idx+binsoct]=value
                self.M[-idx-binsoct:-idx]=value
                idx+=binsoct

        # calculate shifts
        self.wins,self.nn = calcwinrange(self.g, rfbas, self.Ls, device=self.device)
        # calculate dual windows
        self.gd = nsdual(self.g, self.wins, self.nn, self.M, device=self.device)


        #FORWARD!! this is from nsgtf
        #self.forward = lambda s: nsgtf(s, self.g, self.wins, self.nn, self.M, mode=self.mode , device=self.device)
        #sl = slice(0,len(self.g)//2+1)
        sl = slice(1,len(self.g)//2) #getting rig of the DC component and the Nyquist

        self.maxLg_enc = max(int(ceil(float(len(gii))/mii))*mii for mii,gii in zip(self.M[sl], self.g[sl]))
    
        self.loopparams_enc = []
        for mii,gii,win_range in zip(self.M[sl],self.g[sl],self.wins[sl]):
            Lg = len(gii)
            col = int(ceil(float(Lg)/mii))
            assert col*mii >= Lg
            assert col == 1
            p = (mii,win_range,Lg,col)
            self.loopparams_enc.append(p)
    
        def get_ragged_giis(g, wins):
            #ragged_giis = [torch.nn.functional.pad(torch.unsqueeze(gii, dim=0), (0, self.maxLg_enc-gii.shape[0])) for gii in gd[sl]]
            c=torch.zeros((len(g),self.Ls//2),dtype=self.dtype,device=self.device)
            ix=torch.zeros((len(g),self.maxLg_enc),dtype=torch.int64,device=self.device)
            for i,(gii, win_range) in enumerate(zip(g,wins)):
                gii=torch.fft.fftshift(gii).unsqueeze(0)
                Lg=gii.shape[1]
                c[i,win_range]=gii
                ix[i,:(Lg+1)//2]=win_range[Lg//2:].unsqueeze(0)
                ix[i,-(Lg//2):]=win_range[:Lg//2].unsqueeze(0)

                #a=torch.unsqueeze(gii, dim=0)
                #b=torch.nn.functional.pad(a, (0, self.maxLg_enc-gii.shape[0]))
                #ragged_giis.append(b)
            return  torch.conj(c), ix

        def get_ragged_giis_oct(g, wins, ms):
            #ragged_giis = [torch.nn.functional.pad(torch.unsqueeze(gii, dim=0), (0, self.maxLg_enc-gii.shape[0])) for gii in gd[sl]]
            ragged_giis=[]
            c=torch.zeros((len(g),self.Ls//2),dtype=self.dtype,device=self.device)
            ix_per_oct=[]
            for i in range(self.numocts):
                ix_per_oct.append(torch.zeros((self.binsoct,self.size_per_oct[i]),dtype=torch.int64,device=self.device))
            j=0
            k=0
            for i,(gii, win_range) in enumerate(zip(g,wins)):
                if i>0:
                    if ms[i]!=ms[i-1]:
                        j+=1
                        k=0

                gii=torch.fft.fftshift(gii).unsqueeze(0)
                Lg=gii.shape[1]
                c[i,win_range]=gii

                ix_per_oct[j][k,:(Lg+1)//2]=win_range[Lg//2:].unsqueeze(0)
                ix_per_oct[j][k,-(Lg//2):]=win_range[:Lg//2].unsqueeze(0)

                k+=1
                #a=torch.unsqueeze(gii, dim=0)
                #b=torch.nn.functional.pad(a, (0, self.maxLg_enc-gii.shape[0]))
                #ragged_giis.append(b)
            #dirty unsqueeze
            for i in range(len(ix_per_oct)):
                ix_per_oct[i]=ix_per_oct[i].unsqueeze(0).unsqueeze(0)
            return  torch.conj(c), ix_per_oct

        def get_ragged_giis_backup(gd):
            #ragged_giis = [torch.nn.functional.pad(torch.unsqueeze(gii, dim=0), (0, self.maxLg_enc-gii.shape[0])) for gii in gd[sl]]
            ragged_giis=[]
            for gii in gd:
                a=torch.unsqueeze(gii, dim=0)
                b=torch.nn.functional.pad(a, (0, self.maxLg_enc-gii.shape[0]))
                ragged_giis.append(b)

            return  torch.conj(torch.cat(ragged_giis))

        if self.mode=="matrix":
            self.giis, self.idx_enc=get_ragged_giis(self.g[sl], self.wins[sl])
            self.idx_enc=self.idx_enc.unsqueeze(0).unsqueeze(0)
        elif self.mode=="oct":
            self.giis, self.idx_enc=get_ragged_giis_oct(self.g[sl], self.wins[sl], self.M[sl])
            #self.idx_enc=self.idx_enc.unsqueeze(0).unsqueeze(0)
        else:
            self.giis, self.idx_enc=get_ragged_giis(self.g[sl], self.wins[sl])
            #ragged_giis = [torch.nn.functional.pad(torch.unsqueeze(gii, dim=0), (0, self.maxLg_enc-gii.shape[0])) for gii in self.g[sl]]

            #self.giis = torch.conj(torch.cat(ragged_giis))

        #FORWARD!! this is from nsigtf
        #self.backward = lambda c: nsigtf(c, self.gd, self.wins, self.nn, self.Ls, mode=self.mode,  device=self.device)

        self.maxLg_dec = max(len(gdii) for gdii in self.gd)
        print(self.maxLg_enc, self.maxLg_dec)
       
        #ragged_gdiis = [torch.nn.functional.pad(torch.unsqueeze(gdii, dim=0), (0, self.maxLg_dec-gdii.shape[0])) for gdii in self.gd]
        #self.gdiis = torch.conj(torch.cat(ragged_gdiis))

        def get_ragged_gdiis(gd, wins):
            ragged_gdiis=[]
            ix=torch.zeros((len(gd),self.Ls//2+1),dtype=torch.int64,device=self.device)+self.maxLg_dec//2#I initialize the index with the center to make sure that it points to a 0
            for i,(g, win_range) in enumerate(zip(gd, wins)):
                Lg=g.shape[0]
                gl=g[:(Lg+1)//2]
                gr=g[(Lg+1)//2:]
                zeros = torch.zeros(self.maxLg_dec-Lg ,dtype=g.dtype, device=g.device)  # pre-allocation
                paddedg=torch.cat((gl, zeros, gr),0).unsqueeze(0)
                ragged_gdiis.append(paddedg)

                wr1 = win_range[:(Lg)//2]
                wr2 = win_range[-((Lg+1)//2):]
                ix[i,wr1]=torch.Tensor([self.maxLg_dec-(Lg//2)+i for i in range(len(wr1))]).to(torch.int64) #the end part
                ix[i,wr2]=torch.Tensor([i for i in range(len(wr2))]).to(torch.int64) #the start part

                
            return torch.conj(torch.cat(ragged_gdiis))*self.maxLg_dec, ix.unsqueeze(0).unsqueeze(0)

        def get_ragged_gdiis_oct(gd, ms):
            seq_gdiis=[]
            ragged_gdiis=[]
            mprev=-1
            for i,(g,m) in enumerate(zip(gd, ms)):
                if i>0 and m!=mprev:
                    gdii=torch.conj(torch.cat(ragged_gdiis))
                    if len(gdii.shape)==1:
                        gdii=gdii.unsqueeze(0)
                    #seq_gdiis.append(gdii[0:gdii.shape[0]//2 +1])
                    seq_gdiis.append(gdii)
                    ragged_gdiis=[]
                    
                Lg=g.shape[0]
                gl=g[:(Lg+1)//2]
                gr=g[(Lg+1)//2:]
                zeros = torch.zeros(m-Lg ,dtype=g.dtype, device=g.device)  # pre-allocation
                paddedg=torch.cat((gl, zeros, gr),0).unsqueeze(0)*m
                ragged_gdiis.append(paddedg)
                mprev=m
            
            gdii=torch.conj(torch.cat(ragged_gdiis))
            seq_gdiis.append(gdii)
            #seq_gdiis.append(gdii[0:gdii.shape[0]//2 +1])
            return seq_gdiis

        if self.mode=="matrix":
            self.gdiis, self.idx_dec= get_ragged_gdiis(self.gd[sl], self.wins[sl])
            #self.gdiis = self.gdiis[sl]
            #self.gdiis = self.gdiis[0:(self.gdiis.shape[0]//2 +1)]
        else:
            #elif self.mode=="oct":
            self.gdiis=get_ragged_gdiis_oct(self.gd[sl], self.M[sl])

        self.loopparams_dec = []
        for gdii,win_range in zip(self.gd[sl], self.wins[sl]):
            Lg = len(gdii)
            wr1 = win_range[:(Lg)//2]
            wr2 = win_range[-((Lg+1)//2):]
            p = (wr1,wr2,Lg)
            self.loopparams_dec.append(p)


    def nsgtf(self,f):
        """
            forward transform
            args:
                t: Tensor shape(B, C, T) time-domain waveform
            returns:
                if mode = "matrix" 
                    ret: Tensor shape (B, C, F, T') 2d spectrogram spectrogram matrix
                else 
                    ret: list([Tensor]) list of tensors of shape (B, C, Fbibs, T') , representing the bands with the same time-resolution.
                    if mode="oct", the elements on the lists correspond to different octaves
                
        """
        

        ft = torch.fft.fft(f)
    
        ft=ft[...,:self.Ls//2]
        Ls = f.shape[-1]

        assert self.nn == Ls
    
        if self.mode=="matrix":
            #c = torch.zeros(*f.shape[:2], len(self.loopparams_enc), self.maxLg_enc, dtype=ft.dtype, device=torch.device(self.device))
    
            t=ft.unsqueeze(-2)*self.giis #this has a lot of rendundant operations and, probably, consumes a lot of memory. Anyways, it is parallelizable, so it is not a big deal, I guess.
            #c=torch.gather(t, 3, self.idx_enc)

            return torch.fft.ifft(torch.gather(t, 3, self.idx_enc))
    
        elif self.mode=="oct": 
            block_ptr = -1
            bucketed_tensors = []
            ret = []
            ret2 = []
        
            t=ft.unsqueeze(-2)*self.giis #this has a lot of rendundant operations and, probably, consumes a lot of memory. Anyways, it is parallelizable, so it is not a big deal, I guess.

            for i in range(self.numocts):
                #c=torch.gather(t[...,i*self.binsoct:(i+1)*self.binsoct,:], 3, self.idx_enc[i]) 
                #ret.append(torch.fft.ifft(torch.cat(bucketed_tensors,2)))
                ret.append(torch.fft.ifft(torch.gather(t[...,i*self.binsoct:(i+1)*self.binsoct,:], 3, self.idx_enc[i]) ))

            return ret
        else: 
            block_ptr = -1
            bucketed_tensors = []
            ret = []
        
            t=ft.unsqueeze(-2)*self.giis #this has a lot of rendundant operations and, probably, consumes a lot of memory. Anyways, it is parallelizable, so it is not a big deal, I guess.

            for j, (mii,win_range,Lg,col) in enumerate(self.loopparams_enc):

                c = torch.zeros(*f.shape[:2], 1, mii, dtype=ft.dtype, device=torch.device(self.device))
        
                #t = ft[:, :, win_range]*torch.fft.fftshift(self.giis[j, :Lg]) #this needs to be parallelized!
                i=win_range[0]
        
                #c[:, :, 0, :(Lg+1)//2] = t[:, :, Lg//2:]  # if mii is odd, this is of length mii-mii//2
                #c[:, :, 0, -(Lg//2):] = t[:, :, :Lg//2]  # if mii is odd, this is of length mii//2
                c[:, :, 0, :(Lg+1)//2] = t[:, :,j, (i+Lg//2):(i+Lg)]  # if mii is odd, this is of length mii-mii//2
                c[:, :, 0, -(Lg//2):] = t[:, :,j, i:(i+Lg//2)]  # if mii is odd, this is of length mii//2
        
                # start a new block
                if block_ptr == -1 or bucketed_tensors[block_ptr][0].shape[-1] != mii:
                    bucketed_tensors.append(c)
                    block_ptr += 1
                else:
                    # concat block to previous contiguous frequency block with same time resolution
                    bucketed_tensors[block_ptr] = torch.cat([bucketed_tensors[block_ptr], c], dim=2)
        
            #bucket-wise ifft
            for bucketed_tensor in bucketed_tensors:
                ret.append(torch.fft.ifft(bucketed_tensor))
        
            return ret

    def nsigtf(self,cseq):
        """
        mode: "matrix"
            args
                cseq: Time-frequency Tensor with shape (B, C, Freq, Time)
            returns
                sig: Time-domain Tensor with shape (B, C, Time)
                
        """


        if self.mode!="matrix":
            #print(cseq)
            assert type(cseq) == list
            nfreqs = 0
            for i, cseq_tsor in enumerate(cseq):
                cseq_dtype = cseq_tsor.dtype
                cseq[i] = torch.fft.fft(cseq_tsor)
                nfreqs += cseq_tsor.shape[2]
            cseq_shape = (*cseq_tsor.shape[:2], nfreqs)
        else:
            assert type(cseq) == torch.Tensor
            cseq_shape = cseq.shape[:3]
            cseq_dtype = cseq.dtype
            fc = torch.fft.fft(cseq)
        
        fbins = cseq_shape[2]
        #fr = torch.zeros(*cseq_shape[:2],fbins, self.nn, dtype=cseq_dtype, device=torch.device(self.device))  # Allocate output
        #temp0 = torch.empty(*cseq_shape[:2], self.maxLg_dec, dtype=fr.dtype, device=torch.device(self.device))  # pre-allocation
        
        
        # The overlap-add procedure including multiplication with the synthesis windows
        #tart=time.time()
        if self.mode=="matrix":
            temp0=fc*self.gdiis.unsqueeze(0).unsqueeze(0)
            fr=torch.gather(temp0, 3, self.idx_dec).sum(2)
            #fr=fr.sum(2)

            #for i,(wr1,wr2,Lg) in enumerate(self.loopparams_dec[:fbins]):
        
            #    #r = (Lg+1)//2
            #    #l = (Lg//2)
            #    fr2[:, :, wr1] += temp0[:,:,i,self.maxLg_dec-(Lg//2):self.maxLg_dec]
            #    fr2[:, :, wr2] += temp0[:,:,i, :(Lg+1)//2]
        
        else:
            # frequencies are bucketed by same time resolution
            fbin_ptr = 0
            for j, (fc, gdii_j) in enumerate(zip(cseq, self.gdiis)):
                Lg_outer = fc.shape[-1]
        
                nb_fbins = fc.shape[2]
                temp0 = torch.zeros(*cseq_shape[:2],nb_fbins, Lg_outer, dtype=cseq_dtype, device=torch.device(self.device))  # Allocate output
        
                temp0=fc*gdii_j.unsqueeze(0).unsqueeze(0)

                for i,(wr1,wr2,Lg) in enumerate(self.loopparams_dec[fbin_ptr:fbin_ptr+nb_fbins][:fbins]):
                    r = (Lg+1)//2
                    l = (Lg//2)
        
                    fr[:, :, wr1] += temp0[:,:,i,Lg_outer-l:Lg_outer]
                    fr[:, :, wr2] += temp0[:,:,i, :r]

                fbin_ptr += nb_fbins
        
        #end=time.time()
        #rint("in for loop",end-start)
        ftr = fr[:, :, :self.nn//2+1]
        sig = torch.fft.irfft(ftr, n=self.nn)
        sig = sig[:, :, :self.Ls] # Truncate the signal to original length (if given)
        return sig

    def fwd(self,x):
        """
            x: [B,C,T]
        """
        c = self.nsgtf(x)
        return c

    def bwd(self,c):
        s = self.nsigtf(c) #messing out with the channels agains...
        return s

