import numpy as np
import math
from scipy.fftpack import fft


def isPower2(num):
    return ((num & (num - 1)) == 0) and num > 0


def dftAnal(x, w, N):
    if not (isPower2(N)):  # raise error if N not a power of two
        raise ValueError("FFT size (N) is not a power of 2")

    if w.size > N:  # raise error if window size bigger than fft size
        raise ValueError("Window size (M) is bigger than FFT size")

    hN = (N / 2) + 1  # size of positive spectrum, it includes sample 0
    hM1 = int(math.floor((w.size + 1) / 2))  # half analysis window size by rounding
    hM2 = int(math.floor(w.size / 2))  # half analysis window size by floor
    fftbuffer = np.zeros(N)  # initialize buffer for FFT
    w = w / sum(w)  # normalize analysis window
    xw = x * w  # window the input sound
    fftbuffer[:hM1] = xw[hM2:]  # zero-phase window in fftbuffer
    fftbuffer[-hM2:] = xw[:hM2]
    X = fft(fftbuffer)  # compute FFT
    absX = abs(X[: int(hN)])  # compute ansolute value of positive side
    absX[absX < np.finfo(float).eps] = np.finfo(float).eps  # if zeros add epsilon to handle log
    mX = 20 * np.log10(absX)  # magnitude spectrum of positive frequencies in dB
    pX = np.unwrap(np.angle(X[: int(hN)]))  # unwrapped phase spectrum of positive frequencies
    return mX, pX


def stftAnal(x, fs, w, N, H):
    if H <= 0:  # raise error if hop size 0 or negative
        raise ValueError("Hop size (H) smaller or equal to 0")
    M = w.size  # size of analysis window
    hM1 = int(math.floor((M + 1) / 2))  # half analysis window size by rounding
    hM2 = int(math.floor(M / 2))  # half analysis window size by floor
    x = np.append(np.zeros(hM2), x)  # add zeros at beginning to center first window at sample 0
    x = np.append(x, np.zeros(hM2))  # add zeros at the end to analyze last sample
    pin = hM1  # initialize sound pointer in middle of analysis window
    pend = x.size - hM1  # last sample to start a frame
    w = w / sum(w)  # normalize analysis window
    while pin <= pend:  # while sound pointer is smaller than last sample
        x1 = x[int(pin) - hM1:int(pin) + hM2]  # select one frame of input sound
        mX, pX = dftAnal(x1, w, N)  # compute dft
        if pin == hM1:  # if first frame create output arrays
            xmX = np.array([mX])
            xpX = np.array([pX])
        else:  # append output to existing array
            xmX = np.vstack((xmX, np.array([mX])))
            xpX = np.vstack((xpX, np.array([pX])))
        pin += H  # advance sound pointer
    return xmX, xpX
