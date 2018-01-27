#! python3
# encoding=utf-8

import os
import numpy as np
import librosa

N_MFCC=20   #MFCC系数个数

class DTW:
    def __init__(self):
        self.names = None
        self.mfccs = None
    def readTemplate(self, names, mfccs):
        self.names = names
        self.mfccs = mfccs
    def calDTW(self, mfcc1, mfcc2):
        n = len(mfcc1); m = len(mfcc2);
        d = np.array([np.linalg.norm(i - j) for i in mfcc1 for j in mfcc2]).reshape((n, m))
        D = np.zeros((n + 1, m + 1))
        D[0, :] = D[:, 0] = (1 << 30)
        D[0, 0] = 0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                D[i, j] = min((D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])) + d[i - 1, j - 1]
        return D[n, m]
    def test(self, mfcc):
        ret = ''
        mmin = (1 << 30)
        for name in self.names:
            tmp = self.calDTW(self.mfccs[name], mfcc)
            if tmp < mmin:
                mmin = tmp; ret = name;
        return ret

def list_wavs(path): #获得目录下所有wav文件有序列表
    files=[file for file in os.listdir(path) if os.path.isfile(os.path.join(path,file)) and file[-4:]=='.wav']
    files.sort()
    return files
def readTemplate(dtw, path):
    files = list_wavs(path)
    names = [file[: -4] for file in files]
    mfccs = dict()
    for name in names:
        mfccs[name] = []
    for file in files:
        y, sr = librosa.load(os.path.join(path, file))
        mfccs[file[: -4]] += list(librosa.feature.mfcc(y, sr, n_mfcc=N_MFCC).T)
    dtw.readTemplate(names, mfccs)
def test(dtw, path):
    files = list_wavs(path)
    for file in files:
        y, sr = librosa.load(os.path.join(path, file))
        mfcc = list(librosa.feature.mfcc(y, sr, n_mfcc=N_MFCC).T)
        print("文件 %s 经辨识为 %s" %(file, dtw.test(mfcc)))

def main():
    print('''
命令列表:
    train <train path>  -   使用train path目录中的所有wav文件训练模型。
                            训练用文件命名格式为"人名.wav"。例如"Tom.wav"
    test <test path>    -   使用test path目录中的所有wav文件测试模型，输出测试结
                            果。（测试前需要首先训练模型）
    quit                -   退出
    ''')
    dtw = None
    while True:
        op = input('> ').split()
        if op[0] == 'train':
            if dtw == None:
                dtw = DTW()
            readTemplate(dtw, op[1])
        elif op[0] == 'test':
            if dtw == None:
                print('先建模型')
                continue
            test(dtw, op[1])
        elif op[0] == 'quit':
            break
        else:
            print('命令不正确')

main()