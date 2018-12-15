import numpy as np
from PIL import Image
from scipy import misc
import matplotlib.pyplot as plt
import math as mt

initialize=Image.open("MRI.png")
npimage=np.array(initialize)

imageMean = (len(npimage)/2)
print(imageMean)

transArray = np.zeros((3, len(npimage)*len(npimage)))
rotatedValues = np.zeros((1,len(npimage)*len(npimage)))

rotated = np.zeros((3, len(npimage)*len(npimage)))

def initTransformationTable():
    table=np.zeros((3,3))
    for i in range(len(table)-1):
        for j in range(len(table[0])):
            table[i,j]=0
            print(table)
            element=input("please type the "+str(i)+","+str(j)+" element: ")
            table[i,j]=float(element)
    print(table)
    table[2,0]=table[0,2]
    table[2,1]=table[1,2]
    
    table[0,2]=0
    table[1,2]=0
    table[2,2]=1
    
    return table

def inverseTable(transCore, transformedCoords):
    invertedTable=np.zeros_like(transArray)
    
    invertedTemp=np.linalg.inv(transCore)
    for i in range(len(transformedCoords[0])):
        for j in range(3):
            invertedTable[j,i] = np.dot(invertedTemp[:,j],transformedCoords[:,i])
    return invertedTable

def imageTransformation():
    j=0
    for k in range(len(npimage)):
        for l in range(len(npimage[0])):
            transArray[0][j] = k-round(imageMean)    #periexei to Y me vasi to kentro tis eikonas 
            transArray[1][j] = l-round(imageMean)    #periexei to X me vasi to kentro tis eikonas 
            transArray[2][j] = 1             #oli i grammi einai 1
            rotatedValues[0][j] = npimage[k][l]   #periexei apla tis times twn keliwn
            j+=1
    return transArray

def nearestNeighbor(invertedTransformation):#theloume ton pianaka metatropis k ton antistrofo tou
    imagetemp=np.copy(npimage)
    NNimage=np.zeros_like(npimage)

    for j in range(len(transArray[0])):
        imY=int(round(invertedTransformation[0][j])+imageMean)
        imX=int(round(invertedTransformation[1][j])+imageMean)
        nnY=int(round(transArray[0][j])+imageMean)
        nnX=int(round(transArray[1][j])+imageMean)
        if(all(a >= 0 and a<len(npimage) and a<len(npimage[0]) 
               for a in (imX, imY, nnX, nnY))):
            NNimage[nnX][nnY]=npimage[imX][imY]
   
    return NNimage

def implementBilinear(invertedTransformation):
    BLimage=np.zeros((len(npimage),len(npimage[0])))
    BLimage[0:npimage.shape[0],0:npimage.shape[1]] = npimage
    BLimage2=np.copy(BLimage)
    
    for j in range(len(invertedTransformation[0])):
        imY=int(round(invertedTransformation[0][j])+imageMean)
        imX=int(round(invertedTransformation[1][j])+imageMean)        
        BLY=int(round(transArray[0][j])+imageMean)
        BLX=int(round(transArray[1][j])+imageMean)
        
        if(all(a >= 0 and a<len(npimage) and a<len(npimage[0]) 
           for a in (imX, imY, BLX, BLY))):
            BLimage[BLX,BLY]=npimage[imX,imY]
    
    BLimage3=np.copy(BLimage)
    BLimage4=np.copy(BLimage)
    BLimage5=np.copy(BLimage)
    for i in range(1,len(BLimage)-1):
        for j in range(1,len(BLimage[0])-1):
            
            #I use 5 linear interpolations one for each of the upper left and right neighbors of the point 
            #and one for the lowest left and right neighbors of the point
            #and the last one for the point we want to interpolate
            
            #f(x,0) = (1-x)f(0,0)+xf(1,0), 
            #f(x,1) = (1-x)f(0,1) + xf(1,1)            
            BLimage4[i,j-1]=((i-1)*(BLimage3[i-1][j-1])+(i)*BLimage3[i,j-1])
            BLimage4[i,j+1]=((i-1)*(BLimage3[i-1][j+1])+(i)*BLimage3[i,j+1])
            
            #f(0,y) = (1-y)f(0,0) + yf(0,1), 
            #f(1,y) = (1-y)f(1,0) + y(f(1,1)            
            BLimage4[i-1,j]=((j-1)*(BLimage3[i-1,j-1])+(j)*BLimage3[i-1,j])
            BLimage4[i+1,j]=((j-1)*(BLimage3[i+1,j-1])+(j)*BLimage3[i+1,j])
            
            #f(x,y) = ((1-y)f(0,0) + yf(0,1))(1-x) + ((1-y)f(1,0) + y(f(1,1)))x
            BLimage5[i,j]=(BLimage4[i-1][j])*(i-1) + (BLimage4[i+1,j])*(i) + (BLimage4[i][j-1])*(j-1) + (BLimage4[i,j+1])*(j)

    
    return BLimage5

def main():
    
    coreTable=initTransformationTable()
    interpolate=input("please choose the interpolation type NN:(Nearest Neighbor) or BL:(Bilinear) or Nope:(Nothing):")
    transTable=imageTransformation()    
    inverted=inverseTable(coreTable, transTable)  
        
    if(interpolate == "NN"):
        resultImage=nearestNeighbor(inverted)
    elif(interpolate=="BL"):
        resultImage=implementBilinear(inverted)     
    imToSave = np.copy(resultImage)
    imToSave = imToSave - np.min(imToSave)
    imToSave = np.float32(imToSave)
    imToSave = np.round( 255. * (imToSave / np.max(imToSave)) )
    imToSave = np.uint8(imToSave)
    Image.fromarray(imToSave).save('res.png')    

    plt.imshow(resultImage, cmap="gray")
    plt.show()
    
if __name__ == "__main__": main()