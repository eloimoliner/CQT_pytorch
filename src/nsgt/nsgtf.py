# -*- coding: utf-8

"""
Python implementation of Non-Stationary Gabor Transform (NSGT)
derived from MATLAB code by NUHAG, University of Vienna, Austria

Thomas Grill, 2011-2015
http://grrrr.org/nsgt

Austrian Research Institute for Artificial Intelligence (OFAI)
AudioMiner project, supported by Vienna Science and Technology Fund (WWTF)
"""

import numpy as np
import torch
from math import ceil



def nsgtf(f, g, wins, nn, M=None, mode="matrix", device="cpu"):
    #M = chkM(M,g)
    dtype = g[0].dtype
    
    fft=torch.fft.fft
    ifft=torch.fft.ifft
    
    sl = slice(0,len(g)//2+1)
    
    maxLg = max(int(ceil(float(len(gii))/mii))*mii for mii,gii in zip(M[sl],g[sl]))
    temp0 = None
    
    mmap = map

    loopparams = []
    for mii,gii,win_range in zip(M[sl],g[sl],wins[sl]):
        Lg = len(gii)
        col = int(ceil(float(Lg)/mii))
        assert col*mii >= Lg
        assert col == 1

        p = (mii,win_range,Lg,col)
        loopparams.append(p)

    jagged_indices = [None]*len(loopparams)

    ragged_giis = [torch.nn.functional.pad(torch.unsqueeze(gii, dim=0), (0, maxLg-gii.shape[0])) for gii in g[sl]]
    giis = torch.conj(torch.cat(ragged_giis))

    ft = fft(f)

    Ls = f.shape[-1]
    #print("yo",nn, Ls)
    assert nn == Ls

    if mode=="matrix":
        c = torch.zeros(*f.shape[:2], len(loopparams), maxLg, dtype=ft.dtype, device=torch.device(device))

        for j, (mii,win_range,Lg,col) in enumerate(loopparams):
            t = ft[:, :, win_range]*torch.fft.fftshift(giis[j, :Lg])

            sl1 = slice(None,(Lg+1)//2)
            sl2 = slice(-(Lg//2),None)

            c[:, :, j, sl1] = t[:, :, Lg//2:]  # if mii is odd, this is of length mii-mii//2
            c[:, :, j, sl2] = t[:, :, :Lg//2]  # if mii is odd, this is of length mii//2


        return ifft(c)
    elif mode=="oct":
        block_ptr = -1
        bucketed_tensors = []
        ret = []
        
        for j, (mii,win_range,Lg,col) in enumerate(loopparams):

            c = torch.zeros(*f.shape[:2], 1, mii, dtype=ft.dtype, device=torch.device(device)) #that is inefficient, should be allocated outside maybe
        
            t = ft[:, :, win_range]*torch.fft.fftshift(giis[j, :Lg])
        
            sl1 = slice(None,(Lg+1)//2)
            sl2 = slice(-(Lg//2),None)
        
            c[:, :, 0, sl1] = t[:, :, Lg//2:]  # if mii is odd, this is of length mii-mii//2
            c[:, :, 0, sl2] = t[:, :, :Lg//2]  # if mii is odd, this is of length mii//2
        
            #alternative implementation, more elegant but idk if more efficient
            #pad=((mii-Lg+1)//2, (mii-Lg)//2)
            #c=torch.nn.functional.pad(t.unsqueeze(2), pad, "constant", 0)
            #c=torch.roll(c,(mii+1)//2 , dims=-1)
        
            # start a new block
            if block_ptr == -1 or bucketed_tensors[block_ptr][0].shape[-1] != mii:
                bucketed_tensors.append(c)
                block_ptr += 1
            else:
                # concat block to previous contiguous frequency block with same time resolution
                bucketed_tensors[block_ptr] = torch.cat([bucketed_tensors[block_ptr], c], dim=2)

        # bucket-wise ifft
        for bucketed_tensor in bucketed_tensors:
            ret.append(ifft(bucketed_tensor))

        return ret
    else:
        block_ptr = -1
        bucketed_tensors = []
        ret = []

        for j, (mii,win_range,Lg,col) in enumerate(loopparams):
            
            c = torch.zeros(*f.shape[:2], 1, Lg, dtype=ft.dtype, device=torch.device(device))

            t = ft[:, :, win_range]*torch.fft.fftshift(giis[j, :Lg])

            #alternative implementation, more elegant but idk if more efficient. Need to test in gpu
            #pad=((mii-Lg+1)//2, (mii-Lg)//2)
            #c=torch.nn.functional.pad(t.unsqueeze(2), pad, "constant", 0)
            #c=torch.roll(c,(mii+1)//2 , dims=-1)

            sl1 = slice(None,(Lg+1)//2)
            sl2 = slice(-(Lg//2),None)

            c[:, :, 0, sl1] = t[:, :, Lg//2:]  # if mii is odd, this is of length mii-mii//2
            c[:, :, 0, sl2] = t[:, :, :Lg//2]  # if mii is odd, this is of length mii//2

            # start a new block
            if block_ptr == -1 or bucketed_tensors[block_ptr][0].shape[-1] != Lg:
                bucketed_tensors.append(c)
                block_ptr += 1
            else:
                # concat block to previous contiguous frequency block with same time resolution
                bucketed_tensors[block_ptr] = torch.cat([bucketed_tensors[block_ptr], c], dim=2)

        # bucket-wise ifft
        for bucketed_tensor in bucketed_tensors:
            ret.append(ifft(bucketed_tensor))

        return ret
        

