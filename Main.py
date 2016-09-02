"""
This program loads the data from a netcdf4 file and performs the assigned functions.
I have used some parts of the code from the xray tutorial at -
http://nbviewer.jupyter.org/github/nicolasfauchereau/metocean/blob/master/notebooks/xray.ipynb

Author: Mrugen Anil Deshmukh (mad5js)
Date Created: 1 April 2016
"""

import xray
import numpy as np
from mpl_toolkits.basemap import Basemap as bm
from matplotlib import pyplot as plt
import math

# This piece of code has been used directly from the tutorial at -
# http://nbviewer.jupyter.org/github/nicolasfauchereau/metocean/blob/master/notebooks/xray.ipynb
m = bm(projection='cyl',llcrnrlat=-90,urcrnrlat=90,\
            llcrnrlon=0,urcrnrlon=360,\
            lat_ts=0,resolution='c')

# The following function has been borrowed from the given tutorial as well
def plot_field(X, lat, lon, vmin, vmax, step, cmap=plt.get_cmap('jet'), ax=False, title=False, grid=False):
    if not ax: 
        f, ax = plt.subplots(figsize=(10, 10))
    m.ax = ax
    im = m.contourf(lons, lats, X, np.arange(vmin, vmax+step, step), latlon=True, cmap=cmap, extend='both', ax=ax)
    m.drawcoastlines()
    if grid: 
        m.drawmeridians(np.arange(0, 360, 60), labels=[0,0,0,1])
        m.drawparallels([-90, 0, 90], labels=[1,0,0,0])
    m.colorbar(im)
    if title: 
        ax.set_title(title)
       

filename = 'monthlytemps.nc'
# Load the data set
ds = xray.open_dataset(filename)

# The given data set contains a lot of absurd values (around 10e36)
# I've used the following logic to determine the time instances where
# calculating mean of 'tax_mon' is possible, which turns out to be the 
# first 120 months in the data.
valid = []
invalid = []
for i in range(len(ds['time'])):
    x = ds.sel(time = i)['tmax_mon'].mean()
    y = math.isinf(x)
    if y == True:
        invalid.append(i)
    else:
        valid.append(i)
        
valid = np.sort(valid)

# I have created a new data set which contains data for all time instances 
# where 'tmax_mon' does not have absurd values
ds2 = ds.sel(time = valid) 

# tmax and tmin are the maximum and minimum values of 'tmax_mon' over the entire data set
tmax = ds2['tmax_mon'].max()
tmin = ds2['tmax_mon'].min()

# Used to create a meshgrid necessary for the pretty plots
lat = ds2['latitude']
lon = ds2['longitude']
lons, lats = np.meshgrid(lon, lat)

    # Part 1
# Here I have considered that the simlation would only consist of data without
# the absurd 'tmax_mon' values. Hence, only first 120 months considered
y = int(len(ds2['time'])/12)
print("Number of years of simulation:",y)

    # Part 2
# Here I have created a new 2D list where each element is the mean 'tmax_mon'
# value over all time instances for a particular set of coordinates. Then I have 
# used this 2D list to plot for the mean surface temperature for the data set 
ds3 = []
for i in range(len(ds2['latitude'])):
    y = []    
    for j in range(len(ds2['longitude'])):        
        x = ds2.sel(lat = (i), lon = (j))['tmax_mon'].mean()
        y.append(x)
    ds3.append(y)

plot_field(ds3, lats, lons, tmin - 1, tmax + 1, 1,grid=True)
plt.title('Mean surface temperature over the dataset')

    # Part 3
# Plots the surface temperature for the first month
plot_field(ds2.sel(time=(valid[0]))['tmax_mon'], lats, lons, tmin - 1, tmax + 1, 1, grid=True)
plt.title('Surface temperature at the beginning of the simulation')

# Plots the surface temoerature for the last month (data set without the absurd values)
plot_field(ds2.sel(time=(valid[-1]))['tmax_mon'], lats, lons, tmin - 1, tmax + 1, 1, grid=True)
plt.title('Surface temperature at the end of simulation')

    # Part 4
# Coordinates for Mauna Loa used are 19.4721N, 155.5922W
# As these exact values are not present in the data, I have used the closest value
# that is greater than the given value to plot
a1 = np.where(ds2['latitude'] >= 19.4721)
a2 = np.where(ds2['longitude'] >= -155.5922)

y1 = []
for j in range(len(ds2['time'])):
    x1 = ds2.sel(lat = a1[0][0], lon = a2[0][0],time = j )['tmax_mon'].mean()
    y1.append(x1)

plt.figure()
plt.plot(ds2['time'], y1)
plt.title('Surface temperature as a function of time in Mauna Loa')
plt.xlabel('Time (month)')
plt.ylabel('Temperature')

    # Part 5
# Coordinates for Siberia used are 70.73806N, 95.0E
# As these exact values are not present in the data, I have used the closest value
# that is greater than the given value to plot
k1 = np.where(ds2['latitude'] >= 70.73806)
k2 = np.where(ds2['longitude'] >= 95.0)

y2 = []
for j1 in range(len(ds2['time'])):
    x2 = ds2.sel(lat = k1[0][0], lon = k2[0][0], time = j1)['tmax_mon'].mean()
    y2.append(x2)

plt.figure()    
plt.plot(ds2['time'], y2)
plt.title('Surface temperature as a function of time in Siberia')
plt.xlabel('Time (month)')
plt.ylabel('Temperature')
    