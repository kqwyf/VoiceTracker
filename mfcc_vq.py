#encoding=utf-8

import os
import numpy as np
import pickle
import librosa

EPS=1e-1

def dist2(vec1,vec2): #求两向量距离平方
    return np.linalg.norm(vec1-vec2)**2

def dot(k,l): #列表数乘
    return [k*l[i] for i in range(len(l))]

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
        self.cv=[sum(data)/m] #初始码本为所有训练数据平均值
        D=sum([dist2(data[i],self.cv[0]) for i in range(M)])/(M*k) #计算失真度
        print("初始失真度:%f"%(D))
        while len(self.cv)<cv_num: #在码本大小达到期望前不断分裂
            self.cv=(dot(1+eps,self.cv)+dot(1-eps,self.cv)) #分裂码矢
            new_D=sum([(sum([dist2(data[son[i][j]],self.cv[i]) for j in range(len(son[i]))]) if len(son[i]>0) else 0) for i in range(len(son))]) #计算分裂后的新失真度
            while True: #当失真改进量大于阈值时不断更新码矢以减小失真度
                D=new_D
                son=[[]*len(self.cv)] #准备统计每个码矢代表哪些数据
                for i in range(len(data)): #为每个数据分配码矢
                    index=np.argmin([np.linalg.norm(self.cv[j]-data[i]) for j in range(len(self.cv))]) #找出与其距离最近的码矢
                    son[index]+=i #将数据i分配给码矢index
                for i in range(len(self.cv)): #更新每个码矢
                    self.cv[i]=sum([data[son[i][j]] for j in range(len(son[i]))])/len(son[i]) if len(son[i])>0 else 0 #将属于该码矢的所有向量的均值作为新码矢
                new_D=sum([(sum([dist2(data[son[i][j]],self.cv[i]) for j in range(len(son[i]))]) if len(son[i]>0) else 0) for i in range(len(son))]) #计算新失真度
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
        cv_num=0
        self.names=names
        P=len(mfcc)
        while 2**cv_num<P: #求出合适的码本大小
            cv_num+=1
        self.p=np.zeros(shape=(cv_num,P),dtype=np.float) #初始化模型概率表
        data=np.array(sum(mfcc))#获得训练矢量表
        self.lbg(data,cv_num,EPS) #使用LBG算法训练
        for i in range(P): #对于每个码矢，统计其代表不同人的频数
            for j in range(len(mfcc[i])):
                index=np.argmin([np.linalg.norm(self.cv[k]-mfcc[i][j]) for k in range(cv_num)])
                self.p[index][i]+=1
        for i in range(cv_num): #求概率
            self.p[i]/=np.linalg.norm(self.p[i])
    def test(self,mfcc): #进行测试，每次输入1个向量
        '''
        mfcc:待测梅尔频率倒谱系数向量，维数为K
        '''
        p=np.zeros(shape=len(names),dtype=np.float) #识别结果概率向量
        for i in range(len(self.cv)): #对每个码矢进行计算
            p+=self.cv[i]*exp(-np.linalg.norm(mfcc-self.cv[i])) #根据距离加权计算概率和
        p/=np.linalg.norm(p) #标准化概率向量
        return p

def list_wavs(path): #获得目录下所有wav文件有序列表
    files=[file for file in os.listdir(path) if os.path.isfile(os.path.join(path,file)) and file[-4:]=='.wav']
    files.sort()
    return files

def train(vq,path): #训练模型
    files=list_wavs(path) #读取wav文件列表

def test(vq,path): #测试模型
    files=list_wavs(path) #读取wav文件列表

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
