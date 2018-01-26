#encoding=utf-8

import os
import numpy as np
import pickle
import librosa
import math
import matplotlib.pyplot as plt

EPS=1e-3 #失真差阈值
N_MFCC=2 #MFCC系数个数
V_RATE=0.2 #人声阈值，即波形幅值高于最大值的V_RATE倍的区域被认为是人声

def dist2(vec1,vec2): #求两向量距离平方
    return np.linalg.norm(vec1-vec2)**2

def dot(k,l): #列表数乘
    return [k*item for item in l]

def lsum(l): #列表求和
    if l==None or len(l)==0:
        return 0
    ans=l[0]
    for i in l:
        ans+=i
    return ans

class VQ:
    def __init__(self):
        self.cv=None #码矢列表
        self.name="NoName" #模型名称
        self.names=None #身份序列
        self.p=None #模型概率表（训练结果）
    def lbg(self,data,cv_num,eps=EPS): #LBG算法
        '''
        data:训练矢量表(n*k二维np.ndarray，n为矢量数，k为矢量维数)
        cv_num:期望码书大小
        eps:失真阈值差
        '''
        M=len(data) #训练数据量
        k=len(data[0]) #训练矢量维数
        self.cv=[sum(data)/M] #初始码本为所有训练数据平均值
        D=sum([dist2(data[i],self.cv[0]) for i in range(M)])/(M*k) #计算失真度
        print("初始失真度:%f"%(D))
        while len(self.cv)<cv_num: #在码本大小达到期望前不断分裂
            self.cv=(dot(1+eps,self.cv)+dot(1-eps,self.cv)) #分裂码矢
            new_D=D#new_D=sum([(sum([dist2(data[son[i][j]],self.cv[i]) for j in range(len(son[i]))]) if len(son[i]>0) else 0) for i in range(len(son))]) #计算分裂后的新失真度
            while True: #当失真改进量大于阈值时不断更新码矢以减小失真度
                D=new_D
                son=[[] for i in range(len(self.cv))] #准备统计每个码矢代表哪些数据
                for i in range(len(data)): #为每个数据分配码矢
                    index=np.argmin([np.linalg.norm(v-data[i]) for v in self.cv]) #找出与其距离最近的码矢
                    son[index]+=[i] #将数据i分配给码矢index
                for i in range(len(self.cv)): #更新每个码矢
                    self.cv[i]=sum([data[son[i][j]] for j in range(len(son[i]))])/len(son[i]) if len(son[i])>0 else np.array(shape=k,dtype=np.float) #将属于该码矢的所有向量的均值作为新码矢
                new_D=sum([(sum([dist2(data[son[i][j]],self.cv[i]) for j in range(len(son[i]))]) if len(son[i])>0 else np.array(shape=k,dtype=np.float)) for i in range(len(son))])/(M*k) #计算新失真度
                print("失真度:%f"%(new_D))
                if abs(D-new_D)<=eps:
                    break #当失真改进量小于阈值时退出
            print("码本大小:%d"%(len(self.cv)))
    def train(self,mfcc,names): #训练模型
        '''
        mfcc:P*T*N梅尔频率倒谱系数表(list(list(np.ndarray))，P为人数，T为帧数，K为系数个数)
        例如mfcc[i][j][k]表示第i个人的语音第j帧的梅尔频率倒谱系数中第k个系数。
        同一人的语音可拼接后传入。
        names:身份（人名）序列，长度为P
        '''
        cv_num=1
        self.names=names
        P=len(mfcc)
        while cv_num<P: #求出合适的码本大小
            cv_num*=2
        self.p=np.zeros(shape=(cv_num,P),dtype=np.float) #初始化模型概率表
        data=np.array(lsum(mfcc)) #获得训练矢量表
        self.lbg(data,cv_num,EPS) #使用LBG算法训练
        for i in range(P): #对于每个码矢，统计其代表不同人的频数
            for j in range(len(mfcc[i])):
                index=np.argmin([np.linalg.norm(self.cv[k]-mfcc[i][j]) for k in range(cv_num)])
                self.p[index][i]+=1
        for i in range(cv_num): #求概率
            self.p[i]/=np.linalg.norm(self.p[i])
    def test_vec(self,mfcc): #进行测试，每次输入1个向量
        '''
        mfcc:待测梅尔频率倒谱系数向量，维数为K
        '''
        p=np.zeros(shape=len(self.names),dtype=np.float) #识别结果概率向量
        for i in range(len(self.cv)): #对每个码矢进行计算
            p+=self.p[i]*math.exp(-np.linalg.norm(mfcc-self.cv[i])) #根据距离加权计算概率和
        p/=np.linalg.norm(p) #标准化概率向量
        return p
    def test(self,mfcc): #进行测试，每次输入同一声源的一组MFCC向量
        '''
        mfcc:待测N*K梅尔频率到谱向量列表，N为帧数，K为维数
        '''
        return sum([self.test_vec(mfcc_vec) for mfcc_vec in mfcc])/len(mfcc)

def list_wavs(path): #获得目录下所有wav文件有序列表
    files=[file for file in os.listdir(path) if os.path.isfile(os.path.join(path,file)) and file[-4:]=='.wav']
    files.sort()
    return files

def train(vq,path): #训练模型
    files=list_wavs(path) #读取wav文件列表
    files=[(filename.split("_")[0],filename) for filename in files] # 切割文件名，获取身份
    names=list(set([filesplit[0] for filesplit in files])) #获取身份列表
    names.sort() #对身份列表进行排序
    mfcc_dic=dict() #每个人的MFCC表
    for name in names:
        mfcc_dic[name]=[]
    for filesplit in files: #统计每个人的MFCC表
        y,sr=librosa.load(os.path.join(path,filesplit[1])) #读取音频波形
        maxy=max(y)
        mfcc=list(librosa.feature.mfcc(y,sr,n_mfcc=N_MFCC).T)
        mfcc=[mfcc_vec for mfcc_vec in mfcc if np.max(mfcc_vec)>V_RATE*maxy]
        mfcc_dic[filesplit[0]]+=mfcc #计算MFCC并加入MFCC表
    mfcc=[mfcc_dic[name] for name in names] #生成训练用MFCC表
    colors=['b','g','r','c','m','y','k','w']
    for i in range(len(mfcc)):
        plt.scatter([x[0] for x in mfcc[i]],[x[1] for x in mfcc[i]],c=colors[i])
    plt.show()
    vq.train(mfcc,names) #开始训练
    for i in vq.p:
        print(i)
    for i in vq.cv:
        print(i)

def test(vq,path): #测试模型
    files=list_wavs(path) #读取wav文件列表
    for filename in files:
        y,sr=librosa.load(os.path.join(path,filename)) #读取待测波形
        maxy=max(y)
        mfcc=librosa.feature.mfcc(y,sr,n_mfcc=N_MFCC).T; #计算各帧MFCC
        mfcc=[mfcc_vec for mfcc_vec in mfcc if np.max(mfcc_vec)>V_RATE*maxy] #过滤非人声
        print(vq.test(mfcc))
        print("文件 %s 经辨识为 %s"%(filename,vq.names[np.argmax(vq.test(mfcc))]))

def main():
    print('''MFCC & VQ Voice Tracker
命令列表:
    train <train path>  -   使用train path目录中的所有wav文件训练模型。
                            训练用文件命名格式为"人名_编号.wav"。例如"Tom_1.wav"
    test <test path>    -   使用test path目录中的所有wav文件测试模型，输出测试结
                            果。（测试前需要首先训练模型）
    save <model file>       将模型在程序目录下保存为文件名为model file.dat的文件
                            ，以供日后使用。
    load <model file path>  从文件model file path（路径+文件名）加载模型。
    quit                -   退出程序，未保存的模型将被丢弃。
''')
    vq=None
    while True:
        op=input("> ").split()
        if op[0]=='train':
            if vq==None:
                print("建立新模型中...")
                vq=VQ()
                print("建立完毕。")
            train(vq,op[1])
        elif op[0]=='test':
            if vq==None:
                print("尚未建立模型。")
                continue
            test(vq,op[1])
        elif op[0]=='save':
            if vq==None:
                print("尚未建立模型。")
                continue
            print("保存中...")
            vq.name=op[1]
            f=open(op[1]+".dat","wb")
            pickle.dump(vq,f)
            f.close()
            print("保存完毕。")
        elif op[0]=='load':
            if not os.path.exists(op[1]):
                print("未找到文件。")
                continue
            print("正在加载模型...")
            f=open(op[1],"rb")
            vq=pickle.load(f)
            f.close()
            print("模型 %s 加载完毕。"%(vq.name))
        elif op[0]=='quit':
            print("正在退出...")
            exit()
        else:
            print("未知命令。")

main()
