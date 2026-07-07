import torch 
#from src.nsgt.cq  import NSGT

from .fscale  import VariableQLogScale

from .nsgfwin import nsgfwin
from .nsdual import nsdual
from .util import calcwinrange

import math
from math import ceil

def next_power_of_2(x):
    return 1 if x == 0 else 2**math.ceil(math.log2(x))



class CQT_nsgt():
    def __init__(self,numocts, binsoct,  mode="oct", window="hann", fs=44100, audio_len=44100, device="cpu", dtype=torch.float32, verbose=True):
        """
            args:
                numocts (int) number of octaves
                binsoct (int or list[int]) number of bins per octave. If a list, its length
                    must equal numocts and index 0 is the lowest octave (closest to DC),
                    matching the per-octave time-resolution convention (self.size_per_oct).
                    This lets each octave have its own frequency resolution within a single
                    invertible transform (e.g. binsoct=[8,8,8,16,16,16,16,32,32]).
                mode (string) defines the mode of operation:
                     "matrix": returns a 2d-matrix maximum redundancy (discards DC and Nyquist)
                     "matrix_pow2": returns a 2d-matrix maximum redundancy (discards DC and Nyquist) (time-resolution is rounded up to a power of 2)
                     "matrix_complete": returns a 2d-matrix maximum redundancy (with DC and Nyquist)
                     "oct": (default) octave-wise rasterization ( modearate redundancy) returns a list of tensors, each from a different octave with different time resolution (discards DC and Nyquist)
                     "oct_complete": octave-wise rasterization ( modearate redundancy) returns a list of tensors, each from a different octave with different time resolution (with DC and Nyquist)
                fs (float) sampling frequency
                audio_len (int) sample length
                device
        """

        valid_modes = ("matrix","matrix_pow2","matrix_complete","oct","oct_complete")
        if mode not in valid_modes:
            raise ValueError(f"unknown mode {mode!r}; expected one of {valid_modes}")

        fmax=fs/2 -10**-6 #the maximum frequency is Nyquist
        # The transform runs on an internal FFT grid whose length may need to be padded past
        # audio_len for correct reconstruction; the true (possibly odd) output length is kept
        # separately and the inverse truncates back to it. Two requirements on the grid:
        #  * even, because the NSGT places a real Nyquist band at bin nn//2 which only exists
        #    for even length (odd length corrupts/overflows that band);
        #  * for the octave modes, a multiple of the coarsest octave's coefficient length
        #    (max size_per_oct), so the octave downsampling divides the FFT length exactly --
        #    otherwise the frame operator develops a near-gap and loses ~1e-5.
        # The octave multiple is computed by a cheap pre-pass below (it depends on the window
        # lengths, which depend on the grid length). For an even, aligned audio_len (e.g. a
        # power of two) this is all a no-op and self.Ls==audio_len as before.
        self.Ls_out=audio_len                 #true output length (what bwd returns)
        self.Ls=audio_len + (audio_len & 1)   #even grid length; may grow in the pre-pass below

        # fmin sits exactly `numocts` octaves below Nyquist. Each octave i gets its own
        # binsoct[i] bins spaced at exactly 1/binsoct[i] octave (VariableQLogScale), so the
        # grid hits Nyquist exactly at the end of the last octave regardless of whether
        # binsoct is uniform or varies per octave. self.oct_offsets[i]:self.oct_offsets[i+1]
        # gives the flat-bin-index range for octave i.
        if isinstance(binsoct, (list, tuple)):
            assert len(binsoct) == numocts, "binsoct list must have length numocts"
            binsoct = [int(b) for b in binsoct]
        else:
            binsoct = [int(binsoct)] * numocts

        # Validate: bins per octave must be >= 2 (1 bin creates unrealistic time resolution)
        for i, b in enumerate(binsoct):
            if b < 2:
                raise ValueError(f"binsoct[{i}]={b} is invalid; bins per octave must be >= 2. "
                                f"1 bin/octave creates exponentially high time resolution and breaks the inverse transform.")

        self.binsoct = binsoct  # always list[int] of length numocts from here on
        self.oct_offsets = [0]
        for b in self.binsoct:
            self.oct_offsets.append(self.oct_offsets[-1] + b)

        fmin=fmax/(2**numocts)
        self.numocts=numocts
        self.scale = VariableQLogScale(fmin, self.numocts, self.binsoct)

        self.fs=fs

        self.device=torch.device(device)
        self.mode=mode
        self.dtype=dtype
        self.verbose=verbose

        self.frqs,self.q = self.scale()

        # pre-pass for the octave modes: find the coarsest octave coefficient length and pad
        # the grid up to a multiple of it (see the length comment above). Padding by less than
        # one such length changes the window lengths only negligibly, so the rounded octave
        # sizes are stable and a single pass suffices.
        if mode=="oct" or mode=="oct_complete":
            _g,_rf,_M = nsgfwin(self.frqs, self.q, self.fs, self.Ls, dtype=self.dtype, device=self.device, min_win=4, window=window, verbose=False)
            msz = 1
            for i in range(numocts):
                lo, hi = self.oct_offsets[i]+1, self.oct_offsets[i+1]+1
                msz = max(msz, next_power_of_2(int(_M[lo:hi].max())))
            self.Ls = ((self.Ls + msz - 1)//msz)*msz

        self.g,rfbas,self.M = nsgfwin(self.frqs, self.q, self.fs, self.Ls, dtype=self.dtype, device=self.device, min_win=4, window=window, verbose=verbose)

        sl = slice(0,len(self.g)//2+1)

        # coefficients per slice
        self.ncoefs = max(int(math.ceil(float(len(gii))/mii))*mii for mii,gii in zip(self.M[sl],self.g[sl]))        
        if mode=="matrix" or mode=="matrix_complete":
            #just use the maximum resolution everywhere
            self.M[:] = self.M.max()
        elif mode=="matrix_pow2":
            self.size_per_oct=[]
            self.M[:]=next_power_of_2(self.M.max())

        elif mode=="oct" or mode=="oct_complete":
            #round up all the window lengths of an octave to the next power of 2
            self.size_per_oct=[]
            nb=len(self.M)
            for i in range(numocts):
                lo, hi = self.oct_offsets[i]+1, self.oct_offsets[i+1]+1
                value=next_power_of_2(self.M[lo:hi].max())

                self.size_per_oct.append(value)
                self.M[lo:hi]=value
                #mirror to the negative-frequency half: positive bins [lo:hi] sit opposite
                #bins [nb-hi+1 : nb-lo+1]. The old [-hi:-lo] was one index too low -- it
                #clobbered the Nyquist slot and left the widest boundary window one octave
                #short on M, so Lg>M (time aliasing, the painless condition) at non-power-of-2
                #lengths. Powers of two happened to dodge it; other lengths lost ~1e-5.
                self.M[nb-hi+1 : nb-lo+1]=value


        # calculate shifts
        self.wins,self.nn = calcwinrange(self.g, rfbas, self.Ls, device=self.device)
        # calculate dual windows
        self.gd = nsdual(self.g, self.wins, self.nn, self.M, dtype=self.dtype, device=self.device)

        # Hlpf is the combined transfer of the DC and Nyquist bands -- the bands that the
        # non-complete "oct"/"matrix" modes drop -- so that Hhpf = 1 - Hlpf is exactly the
        # transfer of everything they keep, and x = bwd(fwd(x)) + apply_lpf_DC(x) holds to
        # the reconstruction floor. Build each band's transfer the same way nsdual builds the
        # frame operator: fftshift(g)*fftshift(gd)*M scattered at that band's win_range. This
        # is exactly consistent with the transform (summing it over ALL bands gives 1 to
        # float precision) and needs no per-parity index juggling. The previous hand-indexed
        # Nyquist split was ~1e-3 off in the band's skirt just below Nyquist, which left a
        # ~-140 dB residual there in the oct/matrix + apply_lpf_DC reconstruction.
        self.Hlpf=torch.zeros(self.Ls, dtype=self.dtype, device=self.device)
        for k in (0, len(self.g)//2):   # DC band, Nyquist band
            self.Hlpf[self.wins[k]] += (torch.fft.fftshift(self.g[k])
                                        * torch.fft.fftshift(self.gd[k]) * self.M[k])

        self.Hhpf=1-self.Hlpf

        #FORWARD!! this is from nsgtf
        #self.forward = lambda s: nsgtf(s, self.g, self.wins, self.nn, self.M, mode=self.mode , device=self.device)
        #sl = slice(0,len(self.g)//2+1)
        if mode=="matrix" or mode=="oct" or mode=="matrix_pow2":
            sl = slice(1,len(self.g)//2) #getting rid of the DC component and the Nyquist
        else:
            sl = slice(0,len(self.g)//2+1)

        self.maxLg_enc = max(int(ceil(float(len(gii))/mii))*mii for mii,gii in zip(self.M[sl], self.g[sl]))
    
        self.loopparams_enc = []
        for mii,gii,win_range in zip(self.M[sl],self.g[sl],self.wins[sl]):
            Lg = len(gii)
            col = int(ceil(float(Lg)/mii))
            assert col*mii >= Lg
            assert col == 1
            p = (mii,win_range,Lg,col)
            self.loopparams_enc.append(p)
    

        def get_ragged_giis(g, wins, ms, mode):
            #ragged_giis = [torch.nn.functional.pad(torch.unsqueeze(gii, dim=0), (0, self.maxLg_enc-gii.shape[0])) for gii in gd[sl]]
            #ragged_giis=[]
            c=torch.zeros((len(g),self.Ls//2+1),dtype=self.dtype,device=self.device)
            ix=[]

            # octave-boundary detection: for "oct"/"oct_complete", derive it from the known
            # per-octave bin counts (self.oct_offsets) rather than comparing rounded window
            # lengths (ms). With a non-uniform binsoct list, adjacent octaves can round to
            # the SAME window length (e.g. a bins/octave increase offsetting the usual
            # per-octave halving), which would make an ms-based boundary test silently merge
            # two octaves into one bucket. matrix_complete keeps the ms-based test: M is
            # already forced globally uniform there before this runs, so it can't tie.
            bucket_id = None
            if mode in ("oct", "oct_complete"):
                base = 1 if mode == "oct_complete" else 0
                bucket_id = [None]*len(g)
                if mode == "oct_complete":
                    bucket_id[0] = 0
                    bucket_id[-1] = self.numocts + 1
                for oi in range(self.numocts):
                    for p in range(self.oct_offsets[oi], self.oct_offsets[oi+1]):
                        bucket_id[p + base] = oi + base

            if mode=="oct":
                for i in range(self.numocts):
                    ix.append(torch.zeros((self.binsoct[i],self.size_per_oct[i]),dtype=torch.int64,device=self.device))
            elif mode=="matrix" or mode=="matrix_pow2":
                ix.append(torch.zeros((len(g),self.maxLg_enc),dtype=torch.int64,device=self.device))

            elif mode=="oct_complete" or mode=="matrix_complete":
                ix.append(torch.zeros((1,ms[0]),dtype=torch.int64,device=self.device))
                count=0
                for i in range(1,len(g)-1):
                    same = (bucket_id[i]==bucket_id[i-1]) if mode=="oct_complete" else (ms[i]==ms[i-1])
                    if count==0 or same:
                        count+=1
                    else:
                        ix.append(torch.zeros((count,ms[i-1]),dtype=torch.int64,device=self.device))
                        count=1

                ix.append(torch.zeros((count,ms[i-1]),dtype=torch.int64,device=self.device))

                ix.append(torch.zeros((1,ms[-1]),dtype=torch.int64,device=self.device))

            j=0
            k=0
            for i,(gii, win_range) in enumerate(zip(g,wins)):
                if i>0:
                    if mode in ("oct","oct_complete"):
                        boundary = bucket_id[i]!=bucket_id[i-1]
                    else:
                        boundary = ms[i]!=ms[i-1] or (mode=="matrix_complete" and (j==0 or i==len(g)-1))
                    if boundary:
                        j+=1
                        k=0

                gii=torch.fft.fftshift(gii).unsqueeze(0)
                Lg=gii.shape[1]

                if (i==0 or i==len(g)-1) and (mode=="oct_complete" or mode=="matrix_complete"):
                    #special case for the DC and Nyquist, as we don't want to use the mirrored frequencies, take this into account during forward! we would just need to conjugate or sth!
                    if i==0:
                        c[i,win_range[Lg//2:]]=gii[...,Lg//2:]

                        ix[j][0,:(Lg+1)//2]=win_range[Lg//2:].unsqueeze(0)
                        #mirrored tail = the band's Lg//2 negative freqs, gathered from positive bins
                        #and conjugated in fwd. (Lg+1)//2 keeps the source one entry shorter than the
                        #head when Lg is odd (skipping the center bin) so sizes match; for even Lg it
                        #equals Lg//2 and behavior is unchanged.
                        ix[j][0,-(Lg//2):]=torch.flip(win_range[(Lg+1)//2:].unsqueeze(0),(-1,))
                    if i==len(g)-1:
                        #place taps -Lg//2..0 on bins N-Lg//2..N so the peak sits on the Nyquist
                        #bin itself (win_range[Lg//2] is N for both parities)
                        c[i,win_range[:Lg//2+1]]=gii[...,:Lg//2+1]

                        #head = band freqs 0,+1,..: position p gathers bin N-p (true mirror,
                        #conjugated in fwd). For odd Lg the slice starts at win_range[0]; for
                        #even Lg it starts one later so the flip still ends on bin N.
                        ix[j][0,:(Lg+1)//2]=torch.flip(win_range[(Lg+1)%2:Lg//2+1].unsqueeze(0),(-1,))
                        ix[j][0,-(Lg//2):]=win_range[:(Lg)//2].unsqueeze(0)
                else:
                    c[i,win_range]=gii 

                    ix[j][k,:(Lg+1)//2]=win_range[Lg//2:].unsqueeze(0)
                    ix[j][k,-(Lg//2):]=win_range[:Lg//2].unsqueeze(0)

                k+=1
                #a=torch.unsqueeze(gii, dim=0)
                #b=torch.nn.functional.pad(a, (0, self.maxLg_enc-gii.shape[0]))
                #ragged_giis.append(b)
            #dirty unsqueeze
            return  torch.conj(c), ix


        if self.mode=="matrix" or self.mode=="matrix_complete" or self.mode=="matrix_pow2":
            self.giis, self.idx_enc=get_ragged_giis(self.g[sl], self.wins[sl], self.M[sl],self.mode)
            #self.idx_enc=self.idx_enc[0]
            #self.idx_enc=self.idx_enc.unsqueeze(0).unsqueeze(0)
            if self.mode=="matrix" or self.mode=="matrix_pow2":
                #pregather the window at the encode indices so the forward can gather then
                #multiply over each band support ([all_bands, maxLg]) instead of building the
                #dense [B,C,all_bands,Ls/2+1] product. giis_mat gives the same result as
                #gather(ft.unsqueeze(-2)*giis, idx) because gather commutes with the elementwise
                #multiply: ft[...,idx]*gather(giis,idx) == gather(ft*giis, idx). (same trick as oct)
                self.giis_mat = torch.gather(self.giis, 1, self.idx_enc[0])
            elif self.mode=="matrix_complete":
                #same gather-then-multiply trick, but with the DC (band 0), middle (1:-1) and
                #Nyquist (last) groups pregathered separately -- the forward conjugates part of the
                #DC and Nyquist supports (they carry no mirrored frequencies).
                self.giis_mc_dc  = torch.gather(self.giis[0:1],  1, self.idx_enc[0])
                self.giis_mc_mid = torch.gather(self.giis[1:-1], 1, self.idx_enc[1])
                self.giis_mc_nyq = torch.gather(self.giis[-1:],  1, self.idx_enc[-1])
        elif self.mode=="oct" or self.mode=="oct_complete":
            self.giis, self.idx_enc=get_ragged_giis(self.g[sl], self.wins[sl], self.M[sl], self.mode)
            #self.idx_enc=self.idx_enc.unsqueeze(0).unsqueeze(0)
            if self.mode=="oct":
                #pregather each octave window at the encode indices so the forward can gather then multiply
                #over each band support instead of multiplying the full half spectrum for every band
                #giis_oct[i] is [binsoct[i], M_i] and gives the same result as gather(ft*giis)
                self.giis_oct = [
                    torch.gather(self.giis[self.oct_offsets[i]:self.oct_offsets[i+1], :], 1, self.idx_enc[i])
                    for i in range(self.numocts)
                ]
            elif self.mode=="oct_complete":
                #same gather-then-multiply trick, with DC (band 0), each octave, and Nyquist (last band)
                #pregathered separately -- the forward does ft[...,idx]*giis_pregathered per group instead
                #of building the dense [B,C,all_bands,Ls/2+1] product (gather commutes with the multiply).
                #idx_enc for oct_complete is [DC, oct_0..oct_{N-1}, Nyquist]; DC/Nyquist supports get part
                #conjugated in the forward (they carry no mirrored frequencies).
                self.giis_oc_dc = torch.gather(self.giis[0:1, :], 1, self.idx_enc[0])
                self.giis_oc = [
                    torch.gather(self.giis[self.oct_offsets[i]+1:self.oct_offsets[i+1]+1, :], 1, self.idx_enc[i+1])
                    for i in range(self.numocts)
                ]
                self.giis_oc_nyq = torch.gather(self.giis[-1:, :], 1, self.idx_enc[-1])

        #FORWARD!! this is from nsigtf
        #self.backward = lambda c: nsigtf(c, self.gd, self.wins, self.nn, self.Ls, mode=self.mode,  device=self.device)

        self.maxLg_dec = max(len(gdii) for gdii in self.gd)
        if self.mode=="matrix_pow2":
            self.maxLg_dec=self.maxLg_enc
        #self.maxLg_dec=self.maxLg_enc
        #print(self.maxLg_enc, self.maxLg_dec)
       
        #ragged_gdiis = [torch.nn.functional.pad(torch.unsqueeze(gdii, dim=0), (0, self.maxLg_dec-gdii.shape[0])) for gdii in self.gd]
        #self.gdiis = torch.conj(torch.cat(ragged_gdiis))

        def get_ragged_gdiis(gd, wins, mode, ms=None):
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
                if mode=="matrix_complete" and i==0:
                    #ix[i,wr1]=torch.Tensor([self.maxLg_dec-(Lg//2)+i for i in range(len(wr1))]).to(torch.int64) #the end part
                    ix[i,wr2]=torch.Tensor([i for i in range(len(wr2))]).to(torch.int64).to(self.device) #the start part
                elif mode=="matrix_complete" and i==len(gd)-1:
                    ix[i,wr1]=torch.Tensor([self.maxLg_dec-(Lg//2)+i for i in range(len(wr1))]).to(torch.int64).to(self.device) #the end part
                    #the Nyquist bin N (=win_range[Lg//2]) reads band position 0 (the dual
                    #window's peak); without this the bin is dropped and H(Nyquist)=0
                    ix[i,win_range[Lg//2]]=0
                else:
                    ix[i,wr1]=torch.Tensor([self.maxLg_dec-(Lg//2)+i for i in range(len(wr1))]).to(torch.int64).to(self.device) #the end part
                    ix[i,wr2]=torch.Tensor([i for i in range(len(wr2))]).to(torch.int64).to(self.device) #the start part

                
            return torch.conj(torch.cat(ragged_gdiis)).to(self.dtype)*self.maxLg_dec, ix

        def get_ragged_gdiis_oct(gd, ms, wins, mode):
            seq_gdiis=[]
            ragged_gdiis=[]
            mprev=-1
            ix=[]

            # same octave-boundary detection as get_ragged_giis: derive it from the known
            # per-octave bin counts instead of comparing rounded window lengths, since those
            # can tie across octaves when binsoct varies (see get_ragged_giis for details).
            # As a side effect this also robustly forces the DC->octave0 and lastoctave->Nyquist
            # boundaries (bucket_id differs there by construction), so no extra i==0/i==len-1
            # special-casing is needed here.
            base = 1 if mode=="oct_complete" else 0
            bucket_id = [None]*len(gd)
            if mode=="oct_complete":
                bucket_id[0] = 0
                bucket_id[-1] = self.numocts + 1
            for oi in range(self.numocts):
                for p in range(self.oct_offsets[oi], self.oct_offsets[oi+1]):
                    bucket_id[p + base] = oi + base

            if mode=="oct_complete":
                ix+=[torch.zeros((1,self.Ls//2+1),dtype=torch.int64,device=self.device)+ms[0]//2]

            ix+=[torch.zeros((self.binsoct[j],self.Ls//2+1),dtype=torch.int64,device=self.device)+self.size_per_oct[j]//2 for j in range(len(self.size_per_oct))]
            if mode=="oct_complete":
                ix+=[torch.zeros((1,self.Ls//2+1),dtype=torch.int64,device=self.device)+ms[-1]//2]

            #I nitialize the index with the center to make sure that it points to a 0
            j=0
            k=0
            for i,(g,m, win_range) in enumerate(zip(gd, ms, wins)):
                if i>0 and bucket_id[i]!=bucket_id[i-1]:
                    #take care when size of DC is the same as the next octave, or last octave has the same size as nyquist!
                    gdii=torch.conj(torch.cat(ragged_gdiis))
                    if len(gdii.shape)==1:
                        gdii=gdii.unsqueeze(0)
                    #seq_gdiis.append(gdii[0:gdii.shape[0]//2 +1])
                    seq_gdiis.append(gdii.to(self.dtype))
                    ragged_gdiis=[]
                    j+=1
                    k=0
                    
                Lg=g.shape[0]
                gl=g[:(Lg+1)//2]
                gr=g[(Lg+1)//2:]
                zeros = torch.zeros(m-Lg ,dtype=g.dtype, device=g.device)  # pre-allocation
                paddedg=torch.cat((gl, zeros, gr),0).unsqueeze(0)*m
                ragged_gdiis.append(paddedg)
                mprev=m

                wr1 = win_range[:(Lg)//2]
                wr2 = win_range[-((Lg+1)//2):]
                if mode=="oct_complete" and i==0:
                    #ix[i,wr1]=torch.Tensor([self.maxLg_dec-(Lg//2)+i for i in range(len(wr1))]).to(torch.int64) #the end part
                    #ix[i,wr2]=torch.Tensor([i for i in range(len(wr2))]).to(torch.int64) #the start part
                    ix[0][k,wr2]=torch.Tensor([i for i in range(len(wr2))]).to(self.device).to(torch.int64) #the start part
                elif mode=="oct_complete" and i==len(gd)-1:
                    ix[-1][k,wr1]=torch.Tensor([m-(Lg//2)+i for i in range(len(wr1))]).to(self.device).to(torch.int64) #the end part
                    #the Nyquist bin N (=win_range[Lg//2]) reads band position 0 (the dual
                    #window's peak); without this the bin is dropped and H(Nyquist)=0
                    ix[-1][k,win_range[Lg//2]]=0
                else:
                    #ix[i,wr1]=torch.Tensor([self.maxLg_dec-(Lg//2)+i for i in range(len(wr1))]).to(torch.int64) #the end part
                    #ix[i,wr2]=torch.Tensor([i for i in range(len(wr2))]).to(torch.int64) #the start part

                    ix[j][k,wr1]=torch.Tensor([m-(Lg//2)+i for i in range(len(wr1))]).to(self.device).to(torch.int64) #the end part
                    ix[j][k,wr2]=torch.Tensor([i for i in range(len(wr2))]).to(self.device).to(torch.int64) #the start part
                k+=1
            
            gdii=torch.conj(torch.cat(ragged_gdiis))
            seq_gdiis.append(gdii.to(self.dtype))
            #seq_gdiis.append(gdii[0:gdii.shape[0]//2 +1])

            return seq_gdiis, ix

        if self.mode=="matrix" or self.mode=="matrix_complete":
            self.gdiis, self.idx_dec= get_ragged_gdiis(self.gd[sl], self.wins[sl], self.mode)
            #self.gdiis = self.gdiis[sl]
            #self.gdiis = self.gdiis[0:(self.gdiis.shape[0]//2 +1)]
        elif self.mode=="matrix_pow2":
            self.gdiis, self.idx_dec= get_ragged_gdiis(self.gd[sl], self.wins[sl], self.mode, ms=self.M[sl])
        elif self.mode=="oct" or self.mode=="oct_complete":
            self.gdiis, self.idx_dec=get_ragged_gdiis_oct(self.gd[sl], self.M[sl], self.wins[sl], self.mode)
            for gdiis in self.gdiis:
                gdiis.to(self.dtype)


        self.loopparams_dec = []
        for gdii,win_range in zip(self.gd[sl], self.wins[sl]):
            Lg = len(gdii)
            wr1 = win_range[:(Lg)//2]
            wr2 = win_range[-((Lg+1)//2):]
            p = (wr1,wr2,Lg)
            self.loopparams_dec.append(p)

        if self.mode=="oct":
            #precompute per octave the output position that each padded dual window column scatters to
            #this replaces the dense gather(temp0, idx_dec).sum (which also breaks on mps for complex)
            #with one scatter_add over each band support. the zero middle columns map to position 0 and
            #add exactly 0 since gdii is zero there, so they do nothing. same mapping as the overlap add
            #reference, cols [0,r) go to wr2 and cols [Lo-l,Lo) go to wr1
            self.scatter_pos = []
            for j in range(self.numocts):
                Lo = int(self.size_per_oct[j])
                bands = self.loopparams_dec[self.oct_offsets[j]:self.oct_offsets[j+1]]
                pos = torch.zeros((len(bands), Lo), dtype=torch.int64, device=self.device)
                for k,(wr1,wr2,Lg) in enumerate(bands):
                    r = (Lg+1)//2  #gl length goes to wr2
                    l = Lg//2      #gr length goes to wr1
                    pos[k, :r] = wr2
                    if l>0:
                        pos[k, Lo-l:Lo] = wr1
                self.scatter_pos.append(pos)
            #flat index over all octaves so the inverse can overlap add with one index_add
            #(single index_add beats per octave scatter_add and a sparse matmul on gpu, bit identical)
            self.scatter_pos_flat = torch.cat([p.reshape(-1) for p in self.scatter_pos])

        if self.mode=="matrix" or self.mode=="matrix_pow2":
            #same idea as oct's scatter_pos, but every band shares Lo=maxLg_dec: precompute the
            #output bin each padded dual-window column scatters to, so the inverse overlap-adds
            #with ONE index_add instead of gather(temp0,idx_dec).sum which builds a dense
            #[B,C,all_bands,nn//2+1] tensor. Middle (zero) columns map to bin 0 and add 0.
            Lo = self.maxLg_dec
            pos = torch.zeros((len(self.loopparams_dec), Lo), dtype=torch.int64, device=self.device)
            for k,(wr1,wr2,Lg) in enumerate(self.loopparams_dec):
                r = (Lg+1)//2   #gl length goes to wr2
                l = Lg//2       #gr length goes to wr1
                pos[k, :r] = wr2
                if l>0:
                    pos[k, Lo-l:Lo] = wr1
            self.scatter_pos_mat = pos.reshape(-1)

        if self.mode=="oct_complete":
            #per bucket (DC / each octave / Nyquist) invert idx_dec (out_bin -> column) into
            #(column -> out_bin), so the inverse overlap-adds with ONE index_add instead of the
            #per-bucket dense gather(temp0,idx_dec).sum. Same inversion+mask trick as matrix_complete:
            #columns the gather never reads (DC mirrored tail, Nyquist mirrored head) are masked to 0
            #so they don't leak into bin 0. Bucket shapes differ, so masks are kept per-bucket and the
            #inverted positions are concatenated into one flat index.
            half = self.nn//2+1
            self.scatter_mask_oc = []
            pos_flat = []
            for j, idxj in enumerate(self.idx_dec):
                nb_j = idxj.shape[0]; Lg_j = self.gdiis[j].shape[-1]
                pos = torch.zeros((nb_j, Lg_j), dtype=torch.int64, device=self.device)
                mask = torch.zeros((nb_j, Lg_j), dtype=self.dtype, device=self.device)
                outbins = torch.arange(half, device=self.device).unsqueeze(0).expand(nb_j, -1)
                pos.scatter_(1, idxj, outbins)
                mask.scatter_(1, idxj, torch.ones_like(outbins, dtype=self.dtype))
                pos_flat.append(pos.reshape(-1))
                self.scatter_mask_oc.append(mask.unsqueeze(0).unsqueeze(0))   # [1,1,nb_j,Lg_j]
            self.scatter_pos_oc_flat = torch.cat(pos_flat)

        if self.mode=="matrix_complete":
            #build the scatter by INVERTING idx_dec (out_bin -> band column) into
            #(band column -> out_bin), so the inverse overlap-adds with one index_add instead of
            #the dense gather(temp0,idx_dec).sum. The DC/Nyquist special mappings are baked into
            #idx_dec, so inverting handles them for free. BUT the gather never reads some nonzero
            #columns (the DC band's mirrored tail, the Nyquist band's mirrored head); those must be
            #masked to 0 before scattering, else their value would leak into bin 0. A column is
            #"read" iff idx_dec maps some output bin to it, so mask exactly those.
            nb = self.idx_dec.shape[0]
            pos = torch.zeros((nb, self.maxLg_dec), dtype=torch.int64, device=self.device)
            mask = torch.zeros((nb, self.maxLg_dec), dtype=self.dtype, device=self.device)
            outbins = torch.arange(self.nn//2+1, device=self.device).unsqueeze(0).expand(nb, -1)
            pos.scatter_(1, self.idx_dec, outbins)
            mask.scatter_(1, self.idx_dec, torch.ones_like(outbins, dtype=self.dtype))
            self.scatter_pos_matc = pos.reshape(-1)
            self.scatter_mask_matc = mask.unsqueeze(0).unsqueeze(0)   # [1,1,nb,maxLg_dec]

        if self.verbose:
            self.describe()

    def describe(self):
        """Print a table of the transform's per-octave characteristics: frequency range,
        bins, coefficient time-frames and coefficient sample rate per octave band, plus the
        global fmin/fmax and whether the internal FFT grid was zero-padded past audio_len."""
        nyq = self.fs/2.0
        fmin = float(self.frqs[0])
        bl = self.binsoct
        binstr = f"{bl[0]} (uniform)" if len(set(bl))==1 else str(bl)
        lines = []
        lines.append("="*64)
        lines.append(f"CQT_nsgt  |  mode={self.mode}  |  fs={self.fs:g} Hz")
        lines.append(f"  octaves={self.numocts}   bins/oct={binstr}   total log bins={sum(bl)}")
        lines.append(f"  fmin={fmin:.3f} Hz   fmax(Nyquist)={nyq:.3f} Hz")
        pad = self.nn - self.Ls_out
        if pad > 0:
            if self.mode in ("oct","oct_complete"):
                why = f"even & multiple of coarsest band length {max(self.size_per_oct)}"
            else:
                why = "even length (Nyquist band)"
            lines.append(f"  audio_len={self.Ls_out}  ->  internal grid nn={self.nn}   "
                         f"(zero-padded +{pad} samples: {why})")
        else:
            lines.append(f"  audio_len={self.Ls_out} == internal grid nn={self.nn}   (no padding)")
        lines.append(f"  {'oct':>3} | {'freq range (Hz)':^19} | {'bins':>4} | {'frames':>6} | {'coeff rate (Hz)':>15}")
        lines.append(f"  {'-'*3}-+-{'-'*19}-+-{'-'*4}-+-{'-'*6}-+-{'-'*15}")
        for i in range(self.numocts):
            lo = float(self.frqs[self.oct_offsets[i]])
            hi = fmin * 2.0**(i+1)
            frames = int(self.M[self.oct_offsets[i]+1])   # M of octave i's first band (==size_per_oct[i] for oct modes)
            rate = self.fs * frames / self.nn
            lines.append(f"  {i:>3} | {lo:>8.1f} - {hi:>8.1f} | {bl[i]:>4} | {frames:>6} | {rate:>15.2f}")
        if self.mode in ("oct_complete","matrix_complete"):
            lines.append(f"  + DC band (0 - {fmin:.1f} Hz) and Nyquist band ({nyq:.0f} Hz)")
        lines.append("="*64)
        print("\n".join(lines))

    def apply_hpf_DC(self, x):
        Lin=x.shape[-1]
        if Lin<self.Ls:
            #pad zeros
            x=torch.nn.functional.pad(x, (0, self.Ls-Lin))
        elif Lin> self.Ls:
            raise ValueError("Input signal is longer than the maximum length. I could have patched it, but I didn't. sorry :(")

        X=torch.fft.fft(x)
        X=X*torch.conj(self.Hhpf)
        out= torch.fft.ifft(X).real
        if Lin<self.Ls:
            out=out[..., :Lin]
        return out


    def apply_lpf_DC(self, x):
        Lin=x.shape[-1]
        if Lin<self.Ls:
            #pad zeros
            x=torch.nn.functional.pad(x, (0, self.Ls-Lin))
        elif Lin> self.Ls:
            raise ValueError("Input signal is longer than the maximum length. I could have patched it, but I didn't. sorry :(")
        X=torch.fft.fft(x)
        X=X*torch.conj(self.Hlpf)
        out= torch.fft.ifft(X).real
        if Lin<self.Ls:
            out=out[..., :Lin]
        return out


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
        

        Ls = f.shape[-1]

        #accept the true (possibly odd) length and pad up to the even internal grid (self.nn)
        if Ls < self.nn:
            f = torch.nn.functional.pad(f, (0, self.nn - Ls))
        assert f.shape[-1] == self.nn

        #every mode uses only ft[..., :Ls//2+1]; for real input rfft gives exactly that,
        #about 2x cheaper and half the memory with the same values as fft()[:half]
        if not f.is_complex():
            ft = torch.fft.rfft(f)
        else:
            ft = torch.fft.fft(f)

        if self.mode=="matrix" or self.mode=="matrix_pow2":
            ft=ft[...,:self.Ls//2+1]
            #gather each band's spectral support directly to [B,C,all_bands,maxLg] and multiply
            #by the pregathered window -- never builds the dense [B,C,all_bands,Ls/2+1] tensor
            a = ft[..., self.idx_enc[0]] * self.giis_mat
            return torch.fft.ifft(a)
    
        elif self.mode=="oct":
            ft=ft[...,:self.Ls//2 +1]
            ret = []
            #gather each octave spectral support directly to [B,C,binsoct,M_i] with advanced indexing
            #then multiply by the pregathered window. never builds the dense [B,C,all_bands,Ls/2+1] tensor
            for i in range(self.numocts):
                a = ft[..., self.idx_enc[i]] * self.giis_oct[i]
                ret.append(torch.fft.ifft(a))

            return ret
        elif self.mode=="oct_complete":
            ft=ft[...,:self.Ls//2 +1]
            #gather-then-multiply per group (DC / each octave / Nyquist), never building the dense
            #[B,C,all_bands,Ls/2+1] product. bit-identical to gather(ft.unsqueeze(-2)*giis, idx).
            ret = []
            L=self.idx_enc[0].shape[-1]
            a = ft[..., self.idx_enc[0]] * self.giis_oc_dc            #DC
            a[...,(L+1)//2:]=torch.conj(a[...,(L+1)//2:])             #DC carries no mirrored freqs
            ret.append(torch.fft.ifft(a))

            for i in range(self.numocts):                            #octaves
                a = ft[..., self.idx_enc[i+1]] * self.giis_oc[i]
                ret.append(torch.fft.ifft(a))

            L=self.idx_enc[-1].shape[-1]
            a = ft[..., self.idx_enc[-1]] * self.giis_oc_nyq         #Nyquist
            a[...,:(L)//2]=torch.conj(a[...,:(L)//2])                #Nyquist carries no mirrored freqs
            ret.append(torch.fft.ifft(a))

            return ret

        elif self.mode=="matrix_complete":
            ft=ft[...,:self.Ls//2+1]
            #gather-then-multiply per band group (DC / middle / Nyquist), never building the
            #dense [B,C,all_bands,Ls/2+1] product. bit-identical to gather(ft*giis, idx).
            ret=[]
            L=self.idx_enc[0].shape[-1]
            a = ft[..., self.idx_enc[0]] * self.giis_mc_dc            #DC
            a[...,(L+1)//2:]=torch.conj(a[...,(L+1)//2:])
            ret.append(torch.fft.ifft(a))

            a = ft[..., self.idx_enc[1]] * self.giis_mc_mid           #middle bands
            ret.append(torch.fft.ifft(a))

            a = ft[..., self.idx_enc[-1]] * self.giis_mc_nyq          #Nyquist
            a[...,:(L)//2]=torch.conj(a[...,:(L)//2])
            ret.append(torch.fft.ifft(a))
            return torch.cat(ret,dim=2)

    def nsigtf(self,cseq):
        """
        mode: "matrix"
            args
                cseq: Time-frequency Tensor with shape (B, C, Freq, Time)
            returns
                sig: Time-domain Tensor with shape (B, C, Time)
                
        """


        if self.mode!="matrix" and self.mode!="matrix_complete" and self.mode!="matrix_pow2":
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
        #temp0 = torch.empty(*cseq_shape[:2], self.maxLg_dec, dtype=fr.dtype, device=torch.device(self.device))  # pre-allocation
        
        

        # The overlap-add procedure including multiplication with the synthesis windows
        #tart=time.time()
        if self.mode=="matrix" or self.mode=="matrix_pow2":
            #overlap-add every band support into the output spectrum with ONE index_add instead
            #of gather(temp0,idx_dec).sum, which materialises a dense [B,C,all_bands,nn//2+1]
            #tensor. real/imag added apart so it also runs on mps (no complex scatter). bit identical.
            B_, C_ = cseq_shape[:2]
            src = (fc*self.gdiis.unsqueeze(0).unsqueeze(0)).reshape(B_, C_, -1)
            fr = torch.zeros(B_, C_, self.nn//2+1, dtype=cseq_dtype, device=torch.device(self.device))
            torch.view_as_real(fr).index_add_(2, self.scatter_pos_mat, torch.view_as_real(src))

        elif self.mode=="matrix_complete":
            #single index_add overlap-add (see scatter_pos_matc), no dense [B,C,all_bands,nn//2+1].
            #mask drops the DC/Nyquist mirrored columns the gather never read (else they leak to bin 0).
            B_, C_ = cseq_shape[:2]
            src = (fc*self.gdiis.unsqueeze(0).unsqueeze(0)*self.scatter_mask_matc).reshape(B_, C_, -1)
            fr = torch.zeros(B_, C_, self.nn//2+1, dtype=cseq_dtype, device=torch.device(self.device))
            torch.view_as_real(fr).index_add_(2, self.scatter_pos_matc, torch.view_as_real(src))

        elif self.mode=="oct":
            #overlap add every octave support into the output spectrum with ONE index_add instead of
            #a per octave atomic scatter or a dense gather then sum. real and imag added apart so it also
            #runs on mps, which has no complex scatter. measured fastest on cuda and bit identical
            B_, C_ = cseq_shape[:2]
            src = torch.cat([(fc * g.unsqueeze(0).unsqueeze(0)).reshape(B_, C_, -1)
                             for fc, g in zip(cseq, self.gdiis)], dim=-1)  #[B,C,total]
            fr = torch.zeros(B_, C_, self.nn//2+1, dtype=cseq_dtype, device=torch.device(self.device))
            torch.view_as_real(fr).index_add_(2, self.scatter_pos_flat, torch.view_as_real(src))

        elif self.mode=="oct_complete":
            #single index_add overlap-add (see scatter_pos_oc_flat / scatter_mask_oc), no per-bucket
            #dense gather(temp0,idx_dec).sum. per-bucket masks drop the DC/Nyquist mirrored columns the
            #gather never read (else they leak to bin 0). real/imag added apart for mps. bit identical.
            B_, C_ = cseq_shape[:2]
            src = torch.cat([(fc * gdii_j.unsqueeze(0).unsqueeze(0) * mask_j).reshape(B_, C_, -1)
                             for fc, gdii_j, mask_j in zip(cseq, self.gdiis, self.scatter_mask_oc)], dim=-1)
            fr = torch.zeros(B_, C_, self.nn//2+1, dtype=cseq_dtype, device=torch.device(self.device))
            torch.view_as_real(fr).index_add_(2, self.scatter_pos_oc_flat, torch.view_as_real(src))

        #end=time.time()
        #rint("in for loop",end-start)
        ftr = fr[:, :, :self.nn//2+1]
        sig = torch.fft.irfft(ftr, n=self.nn)
        sig = sig[:, :, :self.Ls_out] # Truncate the signal to the true output length
        return sig

    def fwd(self,x):
        """
            x: [B,C,T]
        """
        #torch.fft has no fp16 or bf16, so run the transform in self.dtype with autocast off.
        #the complex output keeps its complex dtype, cast it to your training dtype afterwards
        with torch.autocast(device_type=x.device.type, enabled=False):
            if x.is_floating_point() and x.dtype not in (torch.float32, torch.float64):
                x = x.to(self.dtype)
            c = self.nsgtf(x)
        return c

    def bwd(self,c):
        #inverse, same autocast off handling as fwd
        def _up(t):
            if torch.is_tensor(t) and t.is_complex() and t.dtype == torch.complex32:
                return t.to(torch.complex64)
            return t
        c = [_up(t) for t in c] if isinstance(c, list) else _up(c)
        dev = (c[0] if isinstance(c, list) else c).device
        with torch.autocast(device_type=dev.type, enabled=False):
            s = self.nsigtf(c) #messing out with the channels agains...
        return s

