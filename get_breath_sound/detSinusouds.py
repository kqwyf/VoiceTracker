# -*- coding: utf-8 -*-
import numpy as np
from scipy.signal import get_window
from get_breath_sound import dftStft as stft, peakDetectCorrect as pdc
from scipy.fftpack import fft
def getSinusoids(x, fs):
    M = (20 * fs) / 1000.0  # 20ms window
    N = 1024
    H = M / 4
    window = 'hamming'
    w = get_window(window, int(M))
    mx, px = stft.stftAnal(x, fs, w, N, H)
    mxLinear = 10 ** (mx / 20.0)
    hN = N / 2
    hM = M / 2
    fftbuffer = np.zeros(N)
    mX1 = np.zeros(N)
    hamWin = np.hamming(M)
    hamWin = hamWin / sum(hamWin)
    fftbuffer[int(hN - hM): int(hN + hM)] = hamWin

    X = fft(fftbuffer)
    eps = 1e-8
    X[np.where(abs(X) < eps)] = eps
    mXDb = 20 * np.log10(abs(X))
    mX1[:int(hN)] = mXDb[int(hN):]
    mX1[N - int(hN):] = mXDb[:int(hN)]
    mXLinear = 10 ** (mX1 / 20.0)
    hamMainLobe = mXLinear[507:518]
    maxBinFreq = np.round((5000.0 * N) / fs)
    widthHalfHamMainLobe = hamMainLobe.size / 2

    sinThresh = 0.6
    t = 0.001

    detSinusoids = np.zeros([np.shape(mxLinear)[0], int(maxBinFreq)])

    for i in range(np.shape(mxLinear)[0]):
        tempMag = mxLinear[i, :int(maxBinFreq)]
        ploc = pdc.peakDetection(tempMag, t)
        collS = np.array([])
        index = np.where(ploc <= widthHalfHamMainLobe)[0]
        ploc = np.delete(ploc, index)
        index = np.where(ploc > (maxBinFreq - (widthHalfHamMainLobe + 1)))[0]
        ploc = np.delete(ploc, index)

        for j in range(np.size(ploc)):
            begSilce = ploc[j] - widthHalfHamMainLobe
            endSilce = ploc[j] + widthHalfHamMainLobe + 1
            endSilce = len(hamMainLobe) + int(begSilce)
            measuredSin = tempMag[int(begSilce):endSilce]
            Am = np.sum(hamMainLobe * measuredSin)
            mainLobeEng = np.sum(np.power(hamMainLobe, 2))
            Am = Am / mainLobeEng
            Em = np.sum(np.power((measuredSin - (Am * hamMainLobe)), 2))
            S = 1 - (Em / np.sum(np.power(measuredSin, 2)))
            collS = np.append(collS, S)
        sinPeaksIndx = np.where(collS > sinThresh)[0]
        sinPeaksBin = ploc[sinPeaksIndx]
        sinPeaksMag = tempMag[sinPeaksBin]
        for k in range(sinPeaksBin.size):
            detSinusoids[i, sinPeaksBin[k] - 5:sinPeaksBin[k] + 5] = tempMag[sinPeaksBin[k] - 5:sinPeaksBin[k] + 5]

    return detSinusoids, mxLinear, sinPeaksMag, sinPeaksBin
