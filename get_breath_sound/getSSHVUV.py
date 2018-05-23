import numpy as np


def sshVUV(sshVal, H, fs, N):
    thresh = 0.01
    sshVUV = sshVal > thresh  # Keep zeros at uv and ones at v regions
    sshVUV = np.array(sshVUV, dtype=int)  # converting bool to logical to take diff
    diffShhVal = np.diff(sshVUV)  # Take the difference of signal
    diffShhVal = np.append(diffShhVal, diffShhVal[-1])  # Adjust the signal length
    begVoic = np.where(diffShhVal == 1)[0]
    endVoic = np.where(diffShhVal == -1)[0]  # Values = -1 corresponds to UV regions
    return begVoic, endVoic
