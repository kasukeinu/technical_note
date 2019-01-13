#%%
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def plotGraph(y,labels):
    for i in range(len(y)):
        plt.plot(range(len(y[i])),y[i],label=labels[i])
    plt.legend()
    plt.show()

#%% [markdown]
#  正弦波に高周波ノイズを加えてハイパス・ローパスをかけてみる

#%%
# 時系列のサンプルデータ作成
n = 512                         # データ数
dt = 0.01                       # サンプリング間隔
f = 1                           # 周波数
fn = 1/(2*dt)                   # ナイキスト周波数
t = np.linspace(1, n, n)*dt-dt
# y = np.sin(2*np.pi*f*t)+t+0.5*np.random.randn(t.size)

y = np.sin(2*np.pi*f*t)+0.5*np.random.randn(t.size)

# パラメータ設定
fp = 3                          # 通過域端周波数[Hz]
fs = 2                       # 阻止域端周波数[Hz]
gpass = 1                       # 通過域最大損失量[dB]
gstop = 40                      # 阻止域最小減衰量[dB]
# 正規化
Wp = fp/fn
Ws = fs/fn
Wp2 = Ws
Ws2 = Wp

# ローパスフィルタで波形整形
# バターワースフィルタ
N, Wn = signal.buttord(Wp, Ws, gpass, gstop)
b1, a1 = signal.butter(N, Wn, "High")
y1 = signal.filtfilt(b1, a1, y)

N2, Wn2 = signal.buttord(Wp2, Ws2, gpass, gstop)
b2, a2 = signal.butter(N2, Wn2, "Low")
y2 = signal.filtfilt(b2, a2, y)

plotGraph([y,y1,y2],['row','High','Low'])


#%% [markdown]
#  正弦波にバイアスを加えてハイパス・ローパスをかけてみる

#%%
# 時系列のサンプルデータ作成
n = 512                         # データ数
dt = 0.01                       # サンプリング間隔
f = 1                           # 周波数
fn = 1/(2*dt)                   # ナイキスト周波数
t = np.linspace(1, n, n)*dt-dt
y = np.sin(2*np.pi*f*t)+t

# y = np.sin(2*np.pi*f*t)+0.5*np.random.randn(t.size)

# パラメータ設定
fp = 3                          # 通過域端周波数[Hz]
fs = 2                       # 阻止域端周波数[Hz]
gpass = 1                       # 通過域最大損失量[dB]
gstop = 40                      # 阻止域最小減衰量[dB]
# 正規化
Wp = fp/fn
Ws = fs/fn
Wp2 = Ws
Ws2 = Wp

# ローパスフィルタで波形整形
# バターワースフィルタ
N, Wn = signal.buttord(Wp, Ws, gpass, gstop)
b1, a1 = signal.butter(N, Wn, "High")
y1 = signal.filtfilt(b1, a1, y)
N2, Wn2 = signal.buttord(Wp2, Ws2, gpass, gstop)
b2, a2 = signal.butter(N2, Wn2, "Low")
y2 = signal.filtfilt(b2, a2, y)

plotGraph([y, y1, y2], ['row', 'High', 'Low'])


#%% [markdown]
#  正弦波に高周波ノイズ+バイアスを加えてハイパス・ローパスをかけてみる

#%%
# 時系列のサンプルデータ作成
n = 512                         # データ数
dt = 0.01                       # サンプリング間隔
f = 1                           # 周波数
fn = 1/(2*dt)                   # ナイキスト周波数
t = np.linspace(1, n, n)*dt-dt
y = np.sin(2*np.pi*f*t)+t+0.5*np.random.randn(t.size)


# パラメータ設定
fp = 3                          # 通過域端周波数[Hz]
fs = 2                       # 阻止域端周波数[Hz]
gpass = 1                       # 通過域最大損失量[dB]
gstop = 40                      # 阻止域最小減衰量[dB]
# 正規化
Wp = fp/fn
Ws = fs/fn
Wp2 = Ws
Ws2 = Wp

# ローパスフィルタで波形整形
# バターワースフィルタ
N, Wn = signal.buttord(Wp, Ws, gpass, gstop)
b1, a1 = signal.butter(N, Wn, "High")
y1 = signal.filtfilt(b1, a1, y)
N2, Wn2 = signal.buttord(Wp2, Ws2, gpass, gstop)
b2, a2 = signal.butter(N2, Wn2, "Low")
y2 = signal.filtfilt(b2, a2, y)

plotGraph([y, y1, y2], ['row', 'High', 'Low'])


#%% [markdown]
#  正弦波だけの信号にハイパス・ローパスをかけてみる

#%%
# 時系列のサンプルデータ作成
n = 512                         # データ数
dt = 0.01                       # サンプリング間隔
f = 1                           # 周波数
fn = 1/(2*dt)                   # ナイキスト周波数
t = np.linspace(1, n, n)*dt-dt
y = np.sin(2*np.pi*f*t)


# パラメータ設定
fp = 3                          # 通過域端周波数[Hz]
fs = 2                       # 阻止域端周波数[Hz]
gpass = 1                       # 通過域最大損失量[dB]
gstop = 40                      # 阻止域最小減衰量[dB]
# 正規化
Wp = fp/fn
Ws = fs/fn
Wp2 = Ws
Ws2 = Wp

# ローパスフィルタで波形整形
# バターワースフィルタ
N, Wn = signal.buttord(Wp, Ws, gpass, gstop)
b1, a1 = signal.butter(N, Wn, "High")
y1 = signal.filtfilt(b1, a1, y)
N2, Wn2 = signal.buttord(Wp2, Ws2, gpass, gstop)
b2, a2 = signal.butter(N2, Wn2, "Low")
y2 = signal.filtfilt(b2, a2, y)

plotGraph([y, y1, y2], ['row', 'High', 'Low'])


#%% [markdown]
#  正弦波+早い正弦波にハイパス・ローパスをかけてみる

#%%
# 時系列のサンプルデータ作成
n = 512                         # データ数
dt = 0.01                       # サンプリング間隔
f = 1                           # 周波数
fn = 1/(2*dt)                   # ナイキスト周波数
t = np.linspace(1, n, n)*dt-dt
ya = np.sin(2*np.pi*f*t)
yb = np.sin(10*np.pi*f*t)
y=ya+yb

# パラメータ設定
fp = 3                          # 通過域端周波数[Hz]
fs = 2                       # 阻止域端周波数[Hz]
gpass = 1                       # 通過域最大損失量[dB]
gstop = 40                      # 阻止域最小減衰量[dB]
# 正規化
Wp = fp/fn
Ws = fs/fn
Wp2 = Ws
Ws2 = Wp

# ローパスフィルタで波形整形
# バターワースフィルタ
N, Wn = signal.buttord(Wp, Ws, gpass, gstop)
b1, a1 = signal.butter(N, Wn, "High")
y1 = signal.filtfilt(b1, a1, y)
N2, Wn2 = signal.buttord(Wp2, Ws2, gpass, gstop)
b2, a2 = signal.butter(N2, Wn2, "Low")
y2 = signal.filtfilt(b2, a2, y)

plotGraph([y, y1, y2, ya, yb], ['row', 'High', 'Low', 'Low Freq', 'High Freq'])
plotGraph([y, y1, yb], ['row', 'High', 'High Freq'])
plotGraph([y, y2, ya], ['row', 'Low',  'Low Freq'])


#%%
