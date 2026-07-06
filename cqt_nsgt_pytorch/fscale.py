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

class Scale:
    dbnd = 1.e-8
    
    def __init__(self, bnds):
        self.bnds = bnds
        
    def __len__(self):
        return self.bnds
    
    def Q(self, bnd=None):
        # numerical differentiation (if self.Q not defined by sub-class)
        if bnd is None:
            bnd = np.arange(self.bnds)
        return self.F(bnd)*self.dbnd/(self.F(bnd+self.dbnd)-self.F(bnd-self.dbnd))
    
    def __call__(self):
        f = np.array([self.F(b) for b in range(self.bnds)],dtype=float)
        q = np.array([self.Q(b) for b in range(self.bnds)],dtype=float)
        return f,q

    def suggested_sllen_trlen(self, sr):
        f,q = self()

        Ls = int(np.ceil(max((q*8.*sr)/f)))

        # make sure its divisible by 4
        Ls = Ls + -Ls % 4

        sllen = Ls

        trlen = sllen//4
        trlen = trlen + -trlen % 2 # make trlen divisible by 2

        return sllen, trlen


class LogScale(Scale):
    def __init__(self, fmin, fmax, bnds, beyond=0):
        """
        @param fmin: minimum frequency (Hz)
        @param fmax: maximum frequency (Hz)
        @param bnds: number of frequency bands (int)
        @param beyond: number of frequency bands below fmin and above fmax (int)
        """
        Scale.__init__(self, bnds+beyond*2)
        lfmin = np.log2(fmin)
        lfmax = np.log2(fmax)
        odiv = (lfmax-lfmin)/(bnds-1)
        lfmin_ = lfmin-odiv*beyond
        lfmax_ = lfmax+odiv*beyond
        self.fmin = 2**lfmin_
        self.fmax = 2**lfmax_
        self.pow2n = 2**odiv
        self.q = np.sqrt(self.pow2n)/(self.pow2n-1.)/2.
        
    def F(self, bnd=None):
        return self.fmin*self.pow2n**(bnd if bnd is not None else np.arange(self.bnds))
    
    def Q(self, bnd=None):
        return self.q

class VariableQLogScale(Scale):
    """
    Per-octave log-spaced scale: octave i (0 = lowest, closest to DC) gets
    binsoct[i] bins spaced at exactly 1/binsoct[i] octave. Reduces to the
    classic uniform LogScale(fmin, fmin*2**(numocts-1/b), numocts*b) when
    binsoct == [b]*numocts (max relative deviation ~1e-16, pure floating
    point evaluation-order noise -- verified numerically).
    """
    def __init__(self, fmin, numocts, binsoct):
        assert len(binsoct) == numocts
        assert all(b > 0 for b in binsoct)
        self.numocts = numocts
        self.binsoct = list(binsoct)
        Scale.__init__(self, sum(self.binsoct))

        frqs, qs = [], []
        for i, b in enumerate(self.binsoct):
            pow2n = 2.0 ** (1.0 / b)
            q_i = np.sqrt(pow2n) / (pow2n - 1.0) / 2.0
            oct_fmin = fmin * (2.0 ** i)
            frqs.append(oct_fmin * pow2n ** np.arange(b))
            qs.append(np.full(b, q_i))
        self._frqs = np.concatenate(frqs)
        self._qs = np.concatenate(qs)
        self.fmin = fmin
        self.fmax = self._frqs[-1]

    def F(self, bnd=None):
        return self._frqs if bnd is None else self._frqs[bnd]

    def Q(self, bnd=None):
        return self._qs if bnd is None else self._qs[bnd]




