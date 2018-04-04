import math
import numpy as np
import librosa as rosa
import matplotlib.pyplot as plt


'''
Calculate HNGD for a wave (a,sr) from start to start+M.
if M=None, M will be sr//200 (10ms).
if win_length=None, win_length will be M*40 (400ms).
'''
def HNGD_p(a,sr,start,M=None,win_length=None):
    n=len(a)
    P=((sr//100)-1)//2 #2P+1 corresponds to length of 10ms
    if M==None:
        M=sr//200
    if win_length==None:
        win_length=M*40
    
    
    #ZFF
    #print("processing ZFF")
    s=[0]*n
    for i in range(n-1,-1,-1):
        s[i]=a[i]-(a[i-1] if i>0 else 0)
    
    ##### no usage #####
    '''
    这些量在论文里有计算，但是从论文提供的算法来看从来没有用到过。
    '''
    y1=[0]*n
    for i in range(n-1,-1,-1):
        y1[i]=(2*y1[i-1] if i>0 else 0)-(y1[i-2] if i>1 else 0)+s[i]
    y2=[0]*n
    for i in range(n-1,-1,-1):
        y2[i]=(2*y2[i-1] if i>0 else 0)-(y2[i-2] if i>1 else 0)+y1[i]
    s[0]=y2[0]
    for i in range(1,n):
        s[i]=s[i-1]+y2[i]
    y=[0]*n
    for i in range(n):
        y[i]=y2[i]-((s[i+P] if i+P<n else s[n-1])-(s[i-P-1] if i-P-1>=0 else 0))/(2*P+1)
    ##### no usage #####
    
    
    
    #ZTW
    #print("processing ZTW")
    def w1(n):
        if n==0: return 0
        return 1/(4*math.sin(math.pi*n/(2*win_length))**2)
    def w2(n):
        if n>=M: return 0
        return 4*math.cos(math.pi*n/(2*M))**2
    def window(s,win):
        x=[0]*len(s)
        for i in range(len(s)):
            x[i]=s[i]*win(i)
        return x
    s0=[(s[start+i] if i<M else 0) for i in range(win_length)]
    x=window(s0,w1)
    x=window(x,w1)
    x=window(x,w2)
    
    
    
    #NGD
    #print("processing NGD")
    X=np.fft.fft(np.array(x))
    Y=np.fft.fft(np.array([i*x[i] for i in range(len(x))]))
    g=[X[i].real*Y[i].real+X[i].imag*Y[i].imag for i in range(win_length)]
    
    
    
    #DNGD
    #print("processing DNGD")
    e=[2*g[i]-(g[i+1] if i+1<win_length else g[win_length-1])-(g[i-1] if i-1>=0 else g[0]) for i in range(win_length)]
    
    
    
    #HNGD
    #print("processing HNGD")
    E=np.fft.fft(e)
    Eh=[0]
    Eh+=[(complex(0,-1)*E[i] if i<=win_length//2 else complex(0,1)*E[i]) for i in range(1,win_length)]
    eh=np.fft.ifft(Eh)
    
    ##### MAY BE WRONG #####
    '''
    此处把“复数的平方”理解为“复数的模长的平方”，但论文中没有提到这一点。
    可能出现问题。
    '''
    hngd=[math.sqrt(e[i]**2+eh[i].real**2+eh[i].imag**2) for i in range(win_length)]
    ##### MAY BE WRONG #####
    hngd=hngd[1:win_length//2]
    
    #print("finished")
    return hngd

'''
Calculate HNGD for a wave (a,sr) from start to end with step length M.
if M=None, M will be sr//200 (10ms).
if win_length=None, win_length will be M*40 (400ms).
'''
def HNGD(a,sr,start,end,M=None,win_length=None):
    if M==None:
        M=sr//200
    hngd=[]
    for i in range(start,end-M,M):
        hngd+=[HNGD_p(a,sr,i,M,win_length)]
    return hngd

'''
f="./train/s1.wav"
a,sr=rosa.load(f)
hngd=HNGD(a,sr,5000,12000)
fig=plt.figure()
for i in range(len(hngd)):
    ax=fig.add_subplot(1,len(hngd),i+1)
    ax.plot(hngd[i],list(range(len(hngd[i]))))
plt.show()
'''
