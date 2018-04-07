import HNGD
import librosa as rosa
import matplotlib.pyplot as plt
import numpy as np

def FD(hngd):
    return np.argmax(hngd)


def SD(hngd):
    return max([i*i for i in hngd])


def ER(hngd):
    # 计算中间位置频率和最小位置频率的能量比值
    end=len(hngd)
    if end % 2 == 0:
        middle_strength = hngd[end//2]
    else:
        middle_strength = (hngd[(end+1)//2]+hngd[(end-1)//2])/2
    return middle_strength*middle_strength / (hngd[0] * hngd[0])

def SV(hngd):
    return np.cov(hngd)
#test
# f="./train/s1.wav"
# a,sr=rosa.load(f)
# hngd=HNGD.HNGD(a,sr,5000,6000)[1]
# hngdd=HNGD.HNGD(a,sr,5000,12000)
# fig=plt.figure()
# for i in range(len(hngdd)):
#     ax=fig.add_subplot(1,len(hngdd),i+1)
#     ax.plot(hngdd[i],list(range(len(hngdd[i]))))
# plt.show()
# print(FD(hngd))
# print(SD(hngd))
# print(ER(hngd))
# print(SV(hngd))