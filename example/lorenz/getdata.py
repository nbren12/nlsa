"""Generate data for Lorenz96 system
"""
import numpy as np
from scipy.integrate import odeint
import xarray as xr


def f(x, t, F):
    """Lorenz 96 right hand side"""
    return np.roll(x, 1) * (np.roll(x, -1) - np.roll(x, 2)) - x + F


tout = np.r_[0:100:2e-2]
n = 40
xinit = np.random.rand(n) * .0001

print("Generating output")
xout = odeint(f, xinit, tout, args=(8,))

ds = xr.DataArray(xout, [('time', tout), ('x', np.arange(n))])\
     .to_dataset(name='u')\
     .sel(time=slice(10,None))


print("Saving data to netcdf file")
ds.to_netcdf("lorenz96.nc")


print("Plotting output")
import matplotlib as mpl
import matplotlib.pyplot as plt
ds.u.sel(time=slice(10, 20)).plot.contourf(levels=21, cmap='YlGnBu_r')
plt.savefig("lorenz.png")
