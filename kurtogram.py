'''
Copyright (c) 2015, Jerome Antoni
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the distribution

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
'''

#------------------------------------------------------------------------------
# kurtogram.py
#
# Implement the Fast Kurtogram Algorithm in Python
# 
#
# Created: 09/21/2019 - Daniel Newman -- danielnewman09@gmail.com
#
# Modified:
#   * 09/21/2019 - DMN -- danielnewman09@gmail.com
#           - Replicated Jerome Antoni's Fast Kurtogram algorithm in Python
#             https://www.mathworks.com/matlabcentral/fileexchange/48912-fast-kurtogram
#
#
#------------------------------------------------------------------------------


from scipy.signal import firwin
from scipy.signal import lfilter
import numpy as np

def fast_kurtogram(x,fs,nlevel=7):
    N = x.flatten().size
    
    N2 = np.log2(N) - 7
    
    if nlevel > N2:
        raise ValueError('Please enter a smaller number of decomposition levels')
        
    x -= np.mean(x)
    
    N = 16
    fc = 0.4
    
    h = firwin(N+1,fc) * np.exp(2j * np.pi * np.arange(N+1) * 0.125)
    
    n = np.arange(2,N+2)
    
    g = h[(1-n) % N] * (-1.)**(1-n)
    
    N = int(np.fix(3/2*N))
    
    h1 = firwin(N+1,2/3 * fc) * np.exp(2j * np.pi * np.arange(0,N+1) * 0.25/3)
    h2 = h1 * np.exp(2j * np.pi * np.arange(0,N+1) / 6)
    h3 = h1 * np.exp(2j * np.pi * np.arange(0,N+1) / 3)
    
    Kwav = _K_wpQ(x,h,g,h1,h2,h3,nlevel,'kurt2')
    Kwav = np.clip(Kwav,0,np.inf)
    Level_w = np.arange(1,nlevel+1)
    Level_w = np.vstack((Level_w,
                         Level_w + np.log2(3)-1)).flatten()
    Level_w = np.sort(np.insert(Level_w,0,0)[:2*nlevel])

    freq_w = fs*(np.arange(3*2**nlevel)/(3*2**(nlevel+1)) + 1/(3*2**(2+nlevel)))
    
    max_level_index = np.argmax(Kwav[np.arange(Kwav.shape[0]),np.argmax(Kwav,axis=1)])
    max_kurt = np.amax(Kwav[np.arange(Kwav.shape[0]),np.argmax(Kwav,axis=1)])
    level_max = Level_w[max_level_index]

    bandwidth = fs*2**(-(Level_w[max_level_index] + 1))

    J = np.argmax(Kwav[max_level_index,:])

    fi = (J) / 3/ 2 ** (nlevel + 1)
    fi = fi + fs * 2**(-2-Level_w[max_level_index])

    fc = fs * fi
    
    print('Max Level: {}'.format(level_max))
    print('Freq: {}'.format(fi))
    print('Fs: {}'.format(fs))
    print('Max Kurtosis: {}'.format(max_kurt))
    print('Bandwidth: {}'.format(bandwidth))
    
    c, _, _, _ = find_wav_kurt(x,h,g,h1,h2,h3,level_max,fi,'kurt2',fs)
    
    return Kwav, Level_w, freq_w, c
    
def _kurt(this_x,opt):
    
    eps = 2.2204e-16

    if opt.lower() == 'kurt2':
        if np.all(this_x == 0):
            K = 0
            return K
        this_x -= np.mean(this_x)
        
        E = np.mean(np.abs(this_x)**2)
        
        if E < eps:
            K = 0
            return K
        K = np.mean(np.abs(this_x)**4) / E**2
        
        if np.all(np.isreal(this_x)):
            K -= 3
        else:
            K -= 2
    elif opt.lower() == 'kurt1':
        if np.all(this_x == 0):
            K = 0
            return K
        x -= np.mean(this_x)
        E = np.mean(np.abs(this_x))
        
        if E < eps:
            K = 0
            return K
        
        K = np.mean(np.abs(this_x)**2) / E**2
        
        if np.all(np.isreal(this_x)):
            K -= 1.57
        else:
            K -= 1.27        
            
    
    return K


def _K_wpQ(x,h,g,h1,h2,h3,nlevel,opt,level=None):
    '''
    Computes the kurtosis K of the complete "binary-ternary" wavelet packet transform w of signal x, 
    up to nlevel, using the lowpass and highpass filters h and g, respectively. 
    The values in K are sorted according to the frequency decomposition.
    '''
    
    if level == None:
        level = nlevel
        
    x = x.flatten()
    L = np.floor(np.log2(x.size))
    x = np.atleast_2d(x).T
    
    KD,KQ = _K_wpQ_local(x,h,g,h1,h2,h3,nlevel,opt,level)
    
    K = np.zeros((2 * nlevel,3 * 2**nlevel))
    
    K[0,:] = KD[0,:]
    
    for i in np.arange(1,nlevel):
        K[2*i-1,:] = KD[i,:]
        K[2*i,:] = KQ[i-1,:]
    
    K[2*nlevel-1,:] = KD[nlevel,:]
    

    return K
    
def _K_wpQ_local(x,h,g,h1,h2,h3,nlevel,opt,level):
    
    
    
    a,d = _DBFB(x,h,g)
    
    N = np.amax(a.shape)
    
    d = d * (-1)**(np.atleast_2d(np.arange(1,N+1)).T)

    Lh = np.amax(h.shape)
    Lg = np.amax(g.shape)
        
    K1 = _kurt(a[Lh-1:],opt)
    K2 = _kurt(d[Lg-1:],opt)
    
    if level > 2:
        a1,a2,a3 = _TBFB(a,h1,h2,h3)
        d1,d2,d3 = _TBFB(d,h1,h2,h3)
        
        Ka1 = _kurt(a1[Lh-1:],opt)
        Ka2 = _kurt(a2[Lh-1:],opt)
        Ka3 = _kurt(a3[Lh-1:],opt)
        Kd1 = _kurt(d1[Lh-1:],opt)
        Kd2 = _kurt(d2[Lh-1:],opt)
        Kd3 = _kurt(d3[Lh-1:],opt)
        
    else:
        Ka1 = 0
        Ka2 = 0
        Ka3 = 0
        Kd1 = 0
        Kd2 = 0
        Kd3 = 0
    
    if level == 1:
        K = np.concatenate((K1 * np.ones(3),K2 * np.ones(3)))
#         print(K.shape)
        KQ = np.array([Ka1,Ka2,Ka3,Kd1,Kd2,Kd3])

    if level > 1:
        
        Ka,KaQ = _K_wpQ_local(a,h,g,h1,h2,h3,nlevel,opt,level-1)
        Kd,KdQ = _K_wpQ_local(d,h,g,h1,h2,h3,nlevel,opt,level-1)
        

        K1 *= np.ones(np.amax(Ka.shape))
        K2 *= np.ones(np.amax(Kd.shape))
        
        
        K = np.vstack((np.concatenate([K1,K2]),
                       np.hstack((Ka,Kd))))
        
        
        Long = int(2/6 * np.amax(KaQ.shape))
        Ka1 *= np.ones(Long)
        Ka2 *= np.ones(Long)
        Ka3 *= np.ones(Long)
        Kd1 *= np.ones(Long)
        Kd2 *= np.ones(Long)
        Kd3 *= np.ones(Long)
        
        KQ = np.vstack((np.concatenate([Ka1,Ka2,Ka3,Kd1,Kd2,Kd3]),
                        np.hstack((KaQ,KdQ))))
        
        

    if level == nlevel:
        
        K1 = _kurt(x,opt)
        
        K = np.vstack((K1 * np.ones(np.amax(K.shape)),K))
        
        a1,a2,a3 = _TBFB(x,h1,h2,h3)
        
        Ka1 = _kurt(a1[Lh-1:],opt)
        Ka2 = _kurt(a2[Lh-1:],opt)
        Ka3 = _kurt(a3[Lh-1:],opt)
        
        Long = int(1/3 * np.amax(KQ.shape))
        
        Ka1 *= np.ones(Long)
        Ka2 *= np.ones(Long)
        Ka3 *= np.ones(Long)

        
        KQ = np.vstack((np.concatenate([Ka1,Ka2,Ka3]),
                        KQ[:-2,:]))

    return K,KQ
    
        
    
def _TBFB(x,h1,h2,h3):
    
    N = x.flatten().size
    
    a1 = lfilter(h1,1,x.flatten())
    a1 = a1[2:N:3]
    a1 = np.atleast_2d(a1).T
    
    a2 = lfilter(h2,1,x.flatten())
    a2 = a2[2:N:3]
    a2 = np.atleast_2d(a2).T
    
    a3 = lfilter(h3,1,x.flatten())
    a3 = a3[2:N:3]
    a3 = np.atleast_2d(a3).T
    
    return a1,a2,a3
    
def _DBFB(x,h,g):
    

    N = x.flatten().size
    
    
    a = lfilter(h,1,x.flatten())
    a = a[1:N:2]
    a = np.atleast_2d(a).T
    
    d = lfilter(g,1,x.flatten())
    
    d = d[1:N:2]
    
    d = np.atleast_2d(d).T
    
    return a,d

def binary(i,k):
    
    k = int(k)
    
    if i > 2**k:
        raise ValueError('i must be such that i < 2^k')
    
    a = np.zeros(k)
    
    temp = i
    
    for l in np.arange(k)[::-1]:
        a[-(l+1)] = np.fix(temp / 2**l)
        temp -= a[-(l+1)] * 2 ** l
    
    return a


def find_wav_kurt(x,h,g,h1,h2,h3,Sc,Fr,opt,Fs):
    level = np.fix(Sc) + (np.remainder(Sc,1)>=0.5) * (np.log2(3)-1)
    
    Bw = 2**(-level - 1)
    freq_w = np.arange(2**level) / (2**(level+1)) + Bw/2
    J = np.argmin(np.abs(freq_w - Fr))
    fc = freq_w[J]
    i = np.round((fc/Bw - 1/2))
    
    
    
    if np.remainder(level,1) == 0:
        
        acoeff = binary(i,level)
        bcoeff = np.array([])
        temp_level = level
    
    else:
        
        i2 = np.fix(i/3)
        temp_level = np.fix(level) - 1
        acoeff = binary(i2,temp_level)
        bcoeff = i - i2 * 3
    
    acoeff = acoeff[::-1]

    c = K_wpQ_filt(x,h,g,h1,h2,h3,acoeff,bcoeff,temp_level)
    
    kx = _kurt(c,opt)
    
    sig = np.median(np.abs(c)) / np.sqrt(np.pi / 2)
    
    threshold = sig * np.sqrt((-2*1**2) * np.log(1 - 0.999))
    
    return c, Bw, fc, i
    
def K_wpQ_filt(x,h,g,h1,h2,h3,acoeff,bcoeff,level=None):
    
    nlevel = acoeff.size
    
    L = np.floor(np.log2(np.amax(x.shape)))
    
    if level == None:
        if nlevel >= L:
            raise ValueError('nlevel must be smaller')
        
        level = nlevel
    
    x = np.atleast_2d(x.flatten()).T

    if nlevel == 0:
        if bcoeff.size == 0:
            c = x
        else:
            c1, c2, c3 = _TBFB(x,h1,h2,h3)
            
            if bcoeff == 0:
                c = c1[h1.size - 1:]
            elif bcoeff == 1:
                c = c2[h2.size - 1:]
            elif bcoeff == 2:
                c = c3[h3.size - 1:]
        
    else:
        
        c = K_wpQ_filt_local(x,h,g,h1,h2,h3,acoeff,bcoeff,level)

    
    return c

def K_wpQ_filt_local(x,h,g,h1,h2,h3,acoeff,bcoeff,level):
    
    a,d = _DBFB(x,h,g)
    
    N = a.size
    
    level = int(level)
    
    d = d*np.array([(-1)**(np.arange(1,N+1))]).T
    
    if level == 1:
        if bcoeff.size == 0:
            if acoeff[level-1] == 0:
                c = a[h.size-1:]
            else:
                c = d[g.size-1:]
        else:
            if acoeff[level-1] == 0:
                c1,c2,c3 = _TBFB(a,h1,h2,h3)
            else:
                c1,c2,c3 = _TBFB(d,h1,h2,h3)
            
            if bcoeff == 0:
                c = c1[h1.size - 1:]
            elif bcoeff == 1:
                c = c2[h2.size - 1:]
            elif bcoeff == 2:
                c = c3[h3.size - 1:]
    
    if level > 1:
        if acoeff[level-1] == 0:
            c = K_wpQ_filt_local(a,h,g,h1,h2,h3,acoeff,bcoeff,level-1)
        else:
            c = K_wpQ_filt_local(d,h,g,h1,h2,h3,acoeff,bcoeff,level-1)
            
    return c

