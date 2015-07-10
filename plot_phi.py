# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
d = np.load("eigs.pkl")
phi = d['phi']
df = pd.DataFrame(phi/phi[:,0][:,None])
# df = pd.DataFrame(phi)
df.ix[:,:20].plot(subplots=True, layout=(-1,3))
plt.show()
 
