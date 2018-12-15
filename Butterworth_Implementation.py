import numpy as np
from PIL import Image
import scipy
from scipy import misc
from scipy import signal
import matplotlib.pyplot as plt
import math as mt
from numpy.fft import fft2
from numpy.fft import ifft2
from numpy.fft import fftshift
from numpy.fft import ifftshift

initialize=Image.open("degraded.png")

#DYO BUTTERWORTH ZWNOPERAATA FILTRA ENA ME D02=60 KAI W=25
#KAI ENA ME D02=115 KAI W=25

def Butterworth_implementation(image, d0=None, W=None, n=None):
    npimage=np.array(image)
    F = fft2(image)
    Fshifted = fftshift(F)    
    H = np.zeros_like(F)
    spectrum_center = [x // 2 for x in F.shape]
    plt.subplot(2,2,1).set_title("Basic Fourier")
    plt.imshow(np.log(1+np.abs(Fshifted)),cmap="gray")    
    if(d0==None):
        d0=120
    if(W==None):
        W=6
    if(n==None):
        n=2
    for u in range(0, len(F)):
        for v in range(0, len(F[0])):
            duv=mt.sqrt(mt.pow(np.abs(u-spectrum_center[0]),2)+mt.pow(np.abs(v-spectrum_center[1]),2))
            duv2=mt.pow(duv,2)
            d02=mt.pow(d0,2)               
            if(duv2-d02!=0):
                H[u,v]=1/(1+mt.pow((((duv*W)/(duv2-d02))),2*n))                      
    
    Ffiltered_shifted = Fshifted * H
    
    plt.subplot(2,2,2).set_title("Mask")
    plt.imshow(np.log(1+np.abs(H)),cmap="gray")          

    
    plt.subplot(2,2,3).set_title("Masked Fourier")
    plt.imshow(np.log(1+np.abs(Ffiltered_shifted)),cmap="gray")      
    
    Ffiltered = ifftshift(Ffiltered_shifted)
    
    f_filtered = np.real((ifft2(Ffiltered)))
    
    print(np.log(1+np.abs(np.max(f_filtered))))
    
    plt.subplot(2,2,4).set_title("Filtered Image")        
    plt.imshow(np.log(1+np.abs(f_filtered)), cmap="gray")
    plt.show()
    imToSave = np.copy(f_filtered)
    imToSave = imToSave - np.min(imToSave)
    imToSave = np.float32(imToSave)
    imToSave = np.round( 255. * (imToSave / np.max(imToSave)) )
    imToSave = np.uint8(imToSave)
    string = "filtered_but.png"
    Image.fromarray(imToSave).save(string)   

    return f_filtered

def main():
    butters=Butterworth_implementation(initialize)
    filtered=Butterworth_implementation(butters, d0=60, W=6, n=2)
    
    imToSave = np.copy(filtered)
    imToSave = imToSave - np.min(imToSave)
    imToSave = np.float32(imToSave)
    imToSave = np.round( 255. * (imToSave / np.max(imToSave)) )
    imToSave = np.uint8(imToSave)
    Image.fromarray(imToSave).save('filtered_res.png')
    
if __name__ == "__main__": main()
