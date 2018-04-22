import numpy as np
import os
import librosa as rosa
import detSinusouds as detSin
import ssh as ssh
import getSSHVUV as sshVUV
import matplotlib.pyplot as plt


def getUnvoicedIntervals(x, fs):
    x = x / (1.01 * np.max(np.abs(x)))
    signal_length = x.size
    detSinusoids, mxLinear, sinPeaksMag, sinPeaksBin = detSin.getSinusoids(x, fs)

    H = 0.005 * fs  # 5ms = 80 samples hop size for 16kHz
    N = 1024  # FFT size
    resntFreq, sshVal = ssh.sumSpectHarm(detSinusoids, fs, H, N)
    sshVal = np.power(sshVal, 2)
    sshVal = sshVal / np.max(sshVal)
    begVoic, endVoic = sshVUV.sshVUV(sshVal, H, fs, N)

    timeBegVoicSamp = np.floor(begVoic * H)
    timeEndVoicSamp = np.floor(endVoic * H)
    timeBegUnvoicSamp = np.concatenate([[0], timeEndVoicSamp])
    timeEndUnvoicSamp = np.concatenate([timeBegVoicSamp, [signal_length - 1]])
    return timeBegUnvoicSamp, timeEndUnvoicSamp


path = r'.\train'
files = [file for file in os.listdir(path) if file[-4:] == '.wav']
for f in files:
    x, fs = rosa.load(os.path.join(path, f))
    timeBegUnvoicSamp, timeEndUnvoicSamp = getUnvoicedIntervals(x, fs)
    plt.figure()
    plt.plot(range(len(x)), x)
    plt.stem(timeBegUnvoicSamp, np.ones(len(timeBegUnvoicSamp)), 'r')
    plt.stem(timeEndUnvoicSamp, np.ones(len(timeEndUnvoicSamp)), 'g')
plt.show()
