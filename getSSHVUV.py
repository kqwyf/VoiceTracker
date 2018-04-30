"""
Created on Wed Mar 11 19:11:06 2015
Find the voiced regions in the wave file and then modify pitch of each voiced region
@author: gurunath
"""

import numpy as np


def sshVUV(sshVal, H, fs, N):
    # TODO: Threshold may not work all the time. Come up with some different solution
    thresh = 0.01

    sshVUV = sshVal > thresh  # Keep zeros at uv and ones at v regions

    sshVUV = np.array(sshVUV, dtype=int)  # converting bool to logical to take diff

    diffShhVal = np.diff(sshVUV)  # Take the difference of signal
    diffShhVal = np.append(diffShhVal, diffShhVal[-1])  # Adjust the signal length

    begVoic = np.where(diffShhVal == 1)[
        0]  # Values = 1 in diffSgfiltEpssp1 corresponds to the beg. instant of voiced segment

    endVoic = np.where(diffShhVal == -1)[0]  # Values = -1 corresponds to UV regions

    return begVoic, endVoic
