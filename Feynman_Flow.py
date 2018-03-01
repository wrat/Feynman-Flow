#!/usr/bin/env python

from scipy.ndimage import imread
from matplotlib import pyplot as plt
from matplotlib.pyplot import show
from scipy.ndimage.filters import convolve as filter2
#
from astropy import constants as const
import numpy as np
import math
from load_input.io import getimgfiles

plt.gray()

kernelX = np.array([[-1, 1],
                     [-1, 1]]) * .25 #kernel for computing d/dx
kernelY = np.array([[-1,-1],
                     [ 1, 1]]) * .25 #kernel for computing d/dy

def computeDerivatives(im1, im2):

    theta_x = np.zeros([im1.shape[0],im1.shape[1]])
    theta_y = np.zeros([im1.shape[0],im1.shape[1]])

    #Estimate derivatives
    alpha_x = filter2(im1,kernelX)
    beta_x =  filter2(im2,kernelX)

    alpha_y = filter2(im1,kernelY)
    beta_y  = filter2(im2,kernelY)
    
    for i in range(0,im1.shape[0]):
        for j in range(0,im1.shape[1]):
            alpha = im1[i][j]
            beta = im2[i][j]
            theta_x[i][j] = ( (alpha * beta_x[i][j]) - (beta * alpha_x[i][j]) ) / ( (alpha*alpha) + (beta * beta))
            theta_y[i][j] = ( (alpha * beta_y[i][j]) - (beta * alpha_y[i][j]) ) / ( (alpha*alpha) + (beta * beta))

    return theta_x,theta_y

def superconductivity_continuity(im1,im2):

    X = im1.shape[0]
    Y = im1.shape[1]
    Ax = np.zeros([im1.shape[0],im1.shape[1]])
    Ay = np.zeros([im1.shape[0],im1.shape[1]])
    for i in range(0,X):
        for j in range(0,Y):
            alpha = im1[i][j]
            beta = im2[i][j]
            intensity = (2*const.e.value) * (math.sqrt((alpha*alpha) + (beta * beta)))
            theta =  math.degrees(math.atan(beta/alpha))
            Ax[i][j] = math.sqrt(intensity) * math.cos(theta/2)
            Ay[i][j] = math.sqrt(intensity) * math.sin(theta/2)

    theta_x , theta_y = computeDerivatives(im1,im2)

    #Compute Flow
    Vx =  ( ( ( ( const.h.value * theta_x ) / 2 ) - ( (2*const.e.value) * Ax ) ) / const.m_e.value)
    Vy =  ( ( ( ( const.h.value * theta_y ) / 2 ) - ( (2*const.e.value) * Ay ) ) / const.m_e.value)

    return Vx , Vy

def demo(stem):
    flist,ext = getimgfiles(stem)
    for i in range(len(flist)-1):
        fn1 = "{0}.{1}{2}".format(stem,i,ext)
        im1 = imread(fn1,flatten=True).astype(float)  #flatten=True is rgb2gray

        fn2 = "{0}.{1}{2}".format(stem,i+1,ext)
        im2 = imread(fn2,flatten=True).astype(float)

        Vx,Vy = superconductivity_continuity(im1,im2)
        Motion_detection = np.zeros([im1.shape[0],im1.shape[1]])
        for i in range(0,im1.shape[0]):
            for j in range(0,im1.shape[1]):
                threshold = math.degrees(math.atan(Vy[i][j]/Vx[i][j]))
                if(threshold < -2.78):
                      Motion_detection[i][j] = 1
        plt.imshow(Motion_detection,interpolation='nearest')
        show()

if __name__ == '__main__':
    from argparse import ArgumentParser
    p = ArgumentParser(description='Pure Python Horn Schunck Optical Flow')
    p.add_argument('stem',help='path/stem of files to analyze')
    p = p.parse_args()
    demo(p.stem)
