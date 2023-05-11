# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 15:41:07 2020

@author: grundch
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import cv2

from lib.evaluator import Evaluator
from lib.FileReader import FileReader

#%% Configuration of saving and plotting

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', serif='lm')

fsize = (12/2.54, 7/2.54)
SAVE_FIGURE = False
SAVE_FORMAT = 'png'

#%% Specify parameters
pxSize = 0.1 # Scan resolution in μm
numPeaks = 8 # = number of edges
heightPeaks = 0.8 # approx. value, maybe play around a bit

#%%load data
DIR = os.getcwd()
i=1

reader = FileReader()
for filename in os.listdir(DIR):
    if filename.endswith('.npy'):
        file = np.load(filename)
    try:
        header, data = reader.loadFile(filename)
    except:
        pass

#%% Evaluate
evaluate = Evaluator(pxSize, numPeaks, heightPeaks)
HalfMaxes, gausses, meanCurve = evaluate.evaluate(data)

#%% Plot
x = np.arange(-len(meanCurve)/2, len(meanCurve)/2)*pxSize
fig = plt.figure(1, figsize=fsize)
for gauss in gausses:
    plt.plot(x, gauss, 'bx', markersize = 3)
plt.plot(x, meanCurve, 'r', linewidth=3.0)
plt.ylabel('Norm. Intensity [a.U.]')
plt.xlabel('Width [$\mu$m]')
plt.text(x[int(len(x)*2/3)], 0.9, 'FWHM = \n' + str(round(np.mean(HalfMaxes),2)) + u"\u00B1" + str(round(np.std(HalfMaxes)))+ ' nm')


#%% Save plot
if SAVE_FIGURE:
    plt.savefig('MeanCurve.pdf')

#%% Show plot wih lines
dataWithLines = evaluate.addLinesToImage(file)

fig = plt.figure(2)
pos = plt.imshow(dataWithLines,
                  cmap = 'hot',
                  interpolation='none')

#plt.title(str(Area_x) + ' x ' + str(Area_y) + ' \u03BC' +'m Scan')
cbar = fig.colorbar(pos)
cbar.set_label("Norm. Intensity [a.U.]")

yticks, ylabels = plt.yticks()
xticks, xlabels = plt.xticks()
labelsy = yticks[1:-1]*pxSize
labelsx = xticks[1:-1]*pxSize

plt.yticks(yticks[1:-1], labelsy)
plt.xticks(xticks[1:-1], labelsx)
plt.ylabel('y-pos [$\mu$m]')
plt.xlabel('x-pos [$\mu$m]')
        



