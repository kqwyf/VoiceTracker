import math
import numpy as np
import librosa as rosa
import matplotlib.pyplot as plt

f="./train/s1.wav"

a,sr=rosa.load(f)
n=len(a)
P=((sr//100)-1)//2 #2P+1 corresponds to length of 10ms
M=sr//200
START=0



#ZFF
print("processing ZFF")
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
print("processing ZTW")
win_length=M*40
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
s0=[(s[START+i] if i<M else 0) for i in range(win_length)]
x=window(s0,w1)
x=window(x,w2)



#NGD
print("processing NGD")
X=np.fft.fft(np.array(x))
Y=np.fft.fft(np.array([i*x[i] for i in range(len(x))]))
g=[X[i].real*Y[i].real+X[i].imag*Y[i].imag for i in range(win_length)]



#DNGD
print("processing DNGD")
e=[g[i] for i in range(len(g))]
for i in range(len(e)-1,-1,-1):
    e[i]-=e[i-1] if i>0 else 0
for i in range(len(e)-1,-1,-1):
    e[i]-=e[i-1] if i>0 else 0
for i in range(len(e)):
    e[i]=-e[i]



#HNGD
print("processing HNGD")
E=np.fft.fft(e)
lenE=len(E)

##### MAY BE WRONG #####
'''
论文中在求E_h(\omega)时，\omega的取值范围是(-\pi,\pi)，
但实际上E(\omega)是DFT(g[n])，\omega的范围应为(0,2\pi)。
此处把论文中原式的(-\pi,0)映射到了(\pi,2\pi)
'''
Eh=[(complex(0,-1)*E[i] if i<lenE//2 else complex(0,1)*E[i]) for i in range(lenE)]
##### MAY BE WRONG #####

gh=np.fft.ifft(Eh)

##### MAY BE WRONG #####
'''
此处把“复数的平方”理解为“复数的模长的平方”，但论文中没有提到这一点。
可能出现问题。
'''
hngd=[math.sqrt(g[i]**2+gh[i].real**2+gh[i].imag**2) for i in range(lenE)]
##### MAY BE WRONG #####
hngd=hngd[0:lenE//2]

print("finished")
#plt.plot(list(range(n)),a)
plt.plot(list(range(len(hngd))),hngd)
plt.grid()
plt.show()
