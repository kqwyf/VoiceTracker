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
    timeBegVoicSamp = np.array(begVoic * H, dtype='int32')
    timeEndVoicSamp = np.array(endVoic * H, dtype='int32')
    timeBegUnvoicSamp = np.concatenate([[0], timeEndVoicSamp])
    timeEndUnvoicSamp = np.concatenate([timeBegVoicSamp, [signal_length - 1]])
    length = 0
    for i in range(len(timeBegUnvoicSamp)):
        length += timeEndUnvoicSamp[i] - timeBegUnvoicSamp[i] + 1
    y = np.zeros(length)
    end = 0
    for i in range(len(timeBegUnvoicSamp)):
        interval_length = timeEndUnvoicSamp[i] + 1 - timeBegUnvoicSamp[i]

        y[end: end + interval_length] = x[timeBegUnvoicSamp[i]: timeEndUnvoicSamp[i] + 1]
        end = end + interval_length
    return y


path = r'.\train'
files = [f for f in os.listdir(path) if f[-4:] == '.wav']
for f in files:
    x, fs = rosa.load(os.path.join(path, f))
    y = getUnvoicedIntervals(x, fs)
    plt.figure()
    plt.plot(range(len(y)), y)
plt.show()
