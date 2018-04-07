import librosa as rosa
import matplotlib.pyplot as plt

a, sr = rosa.load('./train/s1.wav')
n = len(a)
plt.figure()
plt.plot(range(n), a)

# 一阶差分 s[n]
n = n - 1
s = [0] * n
for i in range(n):
    s[i] = a[i + 1] - a[i]
plt.figure()
plt.plot(range(n), s)

# 一次ZFR y1[n]
y1 = [0] * n
y1[0] = s[0]
y1[1] = s[1]
for i in range(2, n):
    y1[i] = (2 * y1[i - 1] if i >= 1 else 0) - (y1[i - 2] if i >= 2 else 0) + s[i]
plt.figure()
plt.plot(range(n), y1)

# 二次ZFR y2[n]
y2 = [0] * n
y2[0] = y1[0]
y2[1] = y1[1]
for i in range(2, n):
    y2[i] = (2 * y2[i - 1] if i >= 1 else 0) - (y2[i - 2] if i >= 2 else 0) + y1[i]
plt.figure()
plt.plot(range(n), y2)

# ZFF信号 y[n]
P = ((sr // 100) - 1) // 2
y = [0] * n
for i in range(P, n - P):
    y[i] = y2[i] - sum(y2[i - P: i + P + 1]) / (2 * P + 1)
y = y[P: n - P]
n = n - 2 * P
plt.figure()
plt.plot(range(n), y)
plt.show()
