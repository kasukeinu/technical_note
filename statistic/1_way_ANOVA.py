#%%
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

#%% [markdown]

## one way ANOVA
# 参考：http://swdrsker.hatenablog.com/entry/2017/06/13/184206


#%%
a1 = np.random.normal(10, 5, size=90)
a2 = np.random.normal(10, 10, size=120)
a3 = np.random.normal(10, 20, size=100)

plt.figure(facecolor='w')
bins = np.arange(-30, 50, 5)
plt.hist(a1, bins=bins, alpha=0.3)
plt.hist(a2, bins=bins, alpha=0.3)
plt.hist(a3, bins=bins, alpha=0.3)
result = stats.f_oneway(a1, a2, a3)
print(result.pvalue)
plt.show()


#%%
