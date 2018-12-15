import numpy as np
from numpy import conjugate
from PIL import Image
import scipy
from scipy import misc
from scipy import signal
import matplotlib.pyplot as plt
import math as mt
import cmath as cm
import sys
import os
import SobelFilter as a1

def Hough(init, threshold):
    
    #******************HERE STARTS THE HOUGH ALGOR******************
    thetas = np.deg2rad(np.arange(-90.0, 90.0))#180 degrees
    imWidth, imHeight = init.shape
    
    diag_len = int(np.sqrt(imWidth * imWidth + imHeight * imHeight))#maximum distance
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)#range of rho in accumulator
    diag_size=2*diag_len
    diag_size=np.uint64(diag_size)
    
    coss = np.cos(thetas)
    sins = np.sin(thetas)
    num_thetas = len(thetas)    

    accumulator = np.zeros((diag_size, num_thetas), dtype=np.uint64) #accumulator size
    y_indeces, x_indeces = np.nonzero(init)#edge points
    
    for i in range(len(x_indeces)):#loop for each edge point according to x
        x=x_indeces[i]
        y=y_indeces[i]
        for j in range(num_thetas):#for each theta degree add 1 for each edge you find
            rho = int(round(x*coss[j]+y*sins[j]))+diag_len
            accumulator[rho, j]+=1
    accumulatorNoThres=np.copy(accumulator)#this is used for showing the accumulator
    
    thress=(np.max(accumulator)*threshold) #HERE LIES THE THRESHOLD POINT 4lanes=40% side=40% sidecop=24%
    
    for i in range(len(accumulator)):
        for j in range(len(accumulator[0])):
            if (accumulator[i,j]<thress):
                accumulator[i,j]=0 #the thresholded accumulator
    
    #*************************HERE ENDS THE ACCUMULATOR CODE AND IT'S FINE***************    
    return accumulator, rhos, thetas, diag_len, accumulatorNoThres
    
def drawEdges(accum, inputImage, rhos, thetas, diag_len):
    origin=inputImage
    for k in range(len(accum)):#for each accumulator non-zero point, compute images y according to rho and theta
        for l in range(len(accum[0])):
            if(accum[k,l]>0):             
                rho = rhos[k]
                theta = thetas[l]
                for i in range(len(inputImage[0])):
                    if(np.sin(theta)!=0):
                        y = (rho - i*np.cos(theta))/np.sin(theta)
                        if(np.int64(np.abs(y))>(len(inputImage)/2+10) and np.int64(np.abs(y))<(len(inputImage)-2)): #y reaches only the half of the image
                            origin[np.int64(np.abs(y)), i, 0]=0
                            origin[np.int64(np.abs(y)), i, 1]=255
                            origin[np.int64(np.abs(y)), i, 2]=0    
                            
    #SECOND THOUGHT OF DRAWING LINES USING INTERPOLATION
    #px = rho*np.cos(np.rad2deg(theta))
        #py = rho*np.sin(np.rad2deg(theta))                    
        #p1_x = np.abs(px + diag_len*np.cos((theta)))
        #p1_y = np.abs(py + diag_len*np.sin((theta)))
        #p2_x = np.abs(px - diag_len*np.cos(np.rad2deg(theta)))
        #p2_y = np.abs(py - diag_len*np.sin(np.rad2deg(theta)))
        #if(p1_x<len(inputImage) and p2_x<len(inputImage) and p1_y<len(inputImage[0]) and p2_y<len(inputImage[0])):  
            #origin[np.int64(p1_y), np.int64(p1_x), 0]=255
            #origin[np.int64(p1_y), np.int64(p1_x), 1]=0
            #origin[np.int64(p1_y), np.int64(p1_x), 2]=0                        
            
            #origin[np.int64(p2_y), np.int64(p2_x), 0]=255
            #origin[np.int64(p2_y), np.int64(p2_x), 1]=0
            #origin[np.int64(p2_y), np.int64(p2_x), 2]=0                                
    return origin    


def main():
    if (len(sys.argv)<4):
        print ("Invalid imput\n")
        print ("Execution: SobelFilter.py <inputfile> <input image threshold> <hough accumulator threshold>\n")
        sys.exit()    
    else:
        originalImage = Image.open(str(sys.argv[1]))
        grayIm = a1.rgb2gray(originalImage)
        origin = np.array(originalImage)
        
        init, a, b, c = a1.detectEdges(grayIm, float(sys.argv[2]))#0.33=4lanes=roadside 0.25=sidecop
        
        init[0:np.int64(np.floor(len(init)/2)),:]=0 #the input image contains only the half of the edges, because I want to take info only from road lines
        accumulator, rhos, thetas, diag_len, accumMap=Hough(init, float(sys.argv[3]))#0.4=4lanes=roadside 0.24=sidecop
    
        output = drawEdges(accumulator, origin, rhos, thetas, diag_len)
    
        plt.subplot(1,3,1)
        plt.title('input to Hough', fontsize=8)
        plt.imshow(init, cmap="gray")    
        
        plt.subplot(1,3,2)
        plt.title('accumulator', fontsize=8)
        plt.imshow(accumMap, cmap="gray")    
        
        plt.subplot(1,3,3)
        plt.title('final image', fontsize=8)
        plt.imshow(output, cmap="gray")        
        
        plt.show()
        
        imToSave = np.copy(accumMap)
        imToSave = imToSave - np.min(imToSave)
        imToSave = np.float32(imToSave)
        imToSave = np.round( 255. * (imToSave / np.max(imToSave)) )
        imToSave = np.uint8(imToSave)
        stri="Hough"+str(sys.argv[1])
        Image.fromarray(imToSave).save(stri)
        
        imToSave = np.copy(output)
        imToSave = imToSave - np.min(imToSave)
        imToSave = np.float32(imToSave)
        imToSave = np.round( 255. * (imToSave / np.max(imToSave)) )
        imToSave = np.uint8(imToSave)
        stro="Output"+str(sys.argv[1])
        Image.fromarray(imToSave).save(stro)
if __name__ == "__main__": main()