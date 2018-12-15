import numpy as np
from PIL import Image
from scipy import misc
import matplotlib.pyplot as plt

name=input("please type image's name and type: ")
size=input("please type convolution's core size 3:(3x3) 5:(5x5): ")
size=int(size)
name="MRI.png"
core=np.zeros((size,size))
for z in range(size):
    for w in range(size):
        #core[z][w]=(1/(size*size))   IN CASE YOU DON'T WANT TO TYPE POINT BY POINT
        element=input("please insert the core element "+str(z)+","+str(w)+" ATTENTION: NOT A FRACTION, A NUMBER: ")
        core[z][w]=float(element)

initialize=Image.open(name)
npimage=np.array(initialize)


zeroPadding=(len(npimage)+size-1)
paddedTmp=np.zeros((zeroPadding,zeroPadding))
paddedRes=np.zeros((zeroPadding,zeroPadding))
offset=int(np.ceil(size/2)-1)

paddedTmp[offset:npimage.shape[0]+offset,offset:npimage.shape[1]+offset] = npimage
for i in range(offset, (len(paddedRes)-offset)):
    for j in range(offset, (len(paddedRes)-offset)):
        sum=0
        for m in range(0,size):
            for k in range(0,size):
                sum+=paddedTmp[i+(-zeroPadding+m)][j+(-zeroPadding+k)]*core[m][k]
        paddedRes[i][j]=sum

imToSave = np.copy(paddedRes)
imToSave = imToSave - np.min(imToSave)
imToSave = np.float32(imToSave)
imToSave = np.round( 255. * (imToSave / np.max(imToSave)) )
imToSave = np.uint8(imToSave)
Image.fromarray(imToSave).save('filteredMRI.png')

plt.subplot(1,2,1)
plt.imshow(paddedRes, cmap="gray")
plt.subplot(1,2,2)
plt.imshow(paddedTmp, cmap="gray")
plt.show()
