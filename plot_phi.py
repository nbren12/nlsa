"""Plot eigs.pkl files

Usage:
  plot_eigs.py <eigs>

"""
# coding: utf-8
from docopt import docopt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

args = docopt(__doc__)

d = np.load(args['<eigs>'])
phi = d['phi']
df = pd.DataFrame(phi/phi[:,0][:,None])
# df = pd.DataFrame(phi)
df.ix[:,:20].plot(subplots=True, layout=(-1,3))
plt.show()
 
