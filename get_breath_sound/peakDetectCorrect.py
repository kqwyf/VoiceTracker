import numpy as np
def peakDetection(mX, t):
    thresh = np.where(mX[1:-1] > t, mX[1:-1], 0);  # locations above threshold
    next_minor = np.where(mX[1:-1] > mX[2:], mX[1:-1], 0)  # locations higher than the next one
    prev_minor = np.where(mX[1:-1] > mX[:-2], mX[1:-1], 0)  # locations higher than the previous one
    ploc = thresh * next_minor * prev_minor  # locations fulfilling the three criteria
    ploc = ploc.nonzero()[0] + 1  # add 1 to compensate for previous steps
    return ploc


def peakInterp(mX, pX, ploc):
    val = mX[ploc]  # magnitude of peak bin
    lval = mX[ploc - 1]  # magnitude of bin at left
    rval = mX[ploc + 1]  # magnitude of bin at right
    iploc = ploc + 0.5 * (lval - rval) / (lval - 2 * val + rval)  # center of parabola
    ipmag = val - 0.25 * (lval - rval) * (iploc - ploc)  # magnitude of peaks
    ipphase = np.interp(iploc, np.arange(0, pX.size), pX)  # phase of peaks by linear interpolation
    return iploc, ipmag, ipphase
