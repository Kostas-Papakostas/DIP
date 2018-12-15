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
import getopt

def detectEdges(inputImage, thresshold):
    
    sobelX=[[1,0,-1],[2,0,-2],[1,0,-1]]
    sobelY=[[-1,-2,-1],[0,0,0],[1,2,1]]
    
    imSobeledX=signal.convolve2d(inputImage, sobelX, mode="same", boundary="wrap")
    imSobeledX=np.abs(imSobeledX)
    
    imSobeledY=signal.convolve2d(inputImage, sobelY, mode="same", boundary="wrap")
    imSobeledY=np.abs(imSobeledY)
    
    imRes=np.zeros_like(imSobeledX)
    
    imRes=np.sqrt(pow(imSobeledX,2)+pow(imSobeledY,2))
    sobelRes=np.sqrt(pow(imSobeledX,2)+pow(imSobeledY,2))
    thress=(np.max(imRes)*thresshold) #mgsV:15% mgsVBoss=14% gra=25% sample5=33%
    
    for i in range(0, len(imRes)):
        for j in range(0, len(imRes[0])):
            if(imRes[i,j]>thress):
                imRes[i,j]=255
            else:
                imRes[i,j]=0   
    
    return imRes, imSobeledX, imSobeledY, sobelRes

def rgb2gray(inputRGB):
    
    init=np.array(inputRGB)
    #print(init.ndim)
    redM=np.zeros((len(init), len(init[0])))
    greenM=np.zeros((len(init), len(init[0])))
    blueM=np.zeros((len(init), len(init[0])))
    
    for i in range(0, len(init)):
        for j in range(0, len(init[0])):
            redM[i,j]=init[i,j,0]
            greenM[i,j]=init[i,j,1]
            blueM[i,j]=init[i,j,2]        
    
    grayImage = (redM + greenM + blueM)/3
    grayImage = np.uint8(grayImage)    
    return grayImage

def main():
    if (len(sys.argv)<3):
        print ("No imput file\n")
        print ("Execution:Askisi.py <inputfile> <threshold percentage>\n")
        sys.exit()
    else:
        Initialize=Image.open(str(sys.argv[1]))
        test=np.array(Initialize)
        if(test.ndim==3):
            grayImage = rgb2gray(Initialize)
            imOut, sobelX, sobelY, sobelRes=detectEdges(grayImage, float(sys.argv[2]))
        elif(test.ndim==2):
            init=np.array(Initialize)
            imOut, sobelX, sobelY, sobelRes=detectEdges(init, float(sys.argv[2]))
    
    plt.subplot(2,2,1)
    plt.title('sobelX', fontsize=8)
    plt.imshow(sobelX, cmap="gray")
    
    plt.subplot(2,2,2)
    plt.title('sobelY', fontsize=8)
    plt.imshow(sobelY, cmap="gray")    
    
    plt.subplot(2,2,3)
    plt.title('sobelX+sobelY', fontsize=8)
    plt.imshow(sobelRes, cmap="gray")    
    
    plt.subplot(2,2,4)
    plt.title('sobel thresshold image', fontsize=8)
    plt.imshow(imOut, cmap="gray")        
    
    plt.show()
    string="Output"+str(sys.argv[1])
    imToSave = np.copy(imOut)
    imToSave = imToSave - np.min(imToSave)
    imToSave = np.float32(imToSave)
    imToSave = np.round( 255. * (imToSave / np.max(imToSave)) )
    imToSave = np.uint8(imToSave)
    Image.fromarray(imToSave).save(string)
    
if __name__ == "__main__": main()
