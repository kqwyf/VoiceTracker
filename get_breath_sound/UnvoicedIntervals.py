import numpy as np
import os
import librosa
from get_breath_sound import detSinusouds as detSin, ssh as ssh, getSSHVUV as sshVUV
from config import CONFIG
config = CONFIG()


def get_unvoiced_intervals(x, fs):
    x = x / (1.01 * np.max(np.abs(x)))
    signal_length = x.size
    detSinusoids, mxLinear, sinPeaksMag, sinPeaksBin = detSin.getSinusoids(x, fs)
    H = 0.005 * fs  # 5ms
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


def get_feature(wav_dir, extract_breath=False):
    config = CONFIG()
    data = {}
    for d in os.listdir(wav_dir):
        speaker_dir = os.path.join(wav_dir, d)
        if not os.path.isdir(speaker_dir):
            continue
        data[d] = []
        for file in os.listdir(speaker_dir):
            if file.endswith('.WAV'):
                y, sr = librosa.load(os.path.join(speaker_dir, file), config.sampling_frequency)
                if extract_breath:
                    y = get_unvoiced_intervals(y, sr)
                mfcc = librosa.feature.mfcc(y, sr, n_mfcc=config.feature_size)
                data[d].append(mfcc)
    return data
