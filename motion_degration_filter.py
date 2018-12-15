import numpy as np
from numpy import conjugate
from PIL import Image
import scipy
from scipy import misc
from scipy import signal
import matplotlib.pyplot as plt
import math as mt
import cmath as cm
from numpy.fft import fft2
from numpy.fft import ifft2
from numpy.fft import fftshift
from numpy.fft import ifftshift

initialize=Image.open("degraded_2o.png")#image you want to degrade

def motion_degration(image, a_p=0,b_p=0.85, T_p=2, SNR_p=0.825, d=0.04):
    
    F = fft2(image)
    Fshifted = fftshift(F)
    
    plt.figure().suptitle('psf') #figure name
    
    H = np.ones_like(F)
    spectrum_center = [x // 2 for x in F.shape]
    plt.subplot(2,2,1).set_title("Basic Fourier")
    plt.imshow(np.log(1+np.abs(Fshifted)),cmap="gray")    
    
    a=a_p       #degrees of the filter, here is 0 to apply it vertically
    b=b_p       #higher value means small filter
    T=T_p       #shows the density of the lines
    for u in range(0, len(F)):
        for v in range(240, len(F[0])-240):
            differ=mt.pi*((u-spectrum_center[0])*a)+((v-spectrum_center[1])*b)
            expose=cm.exp(-1j*differ)
            sinus=mt.sin(differ)*expose
            if(differ!=0):
                H[u,v]=(T/differ)*sinus
                
    SNR=SNR_p 
    
    W=(np.conj(H))/(np.power(np.abs(H),2)+(1/SNR))
    Ffiltered_shifted = (Fshifted)*W
    
    plt.subplot(2,2,2).set_title("Mask")
    plt.imshow(np.log(1+np.abs(H)),cmap="gray")   

    
    plt.subplot(2,2,3).set_title("Masked Fourier")
    plt.imshow(np.log(1+np.abs(Ffiltered_shifted)),cmap="gray")      
    
    Ffiltered = ifftshift(Ffiltered_shifted)
    
    f_filtered = np.real((ifft2(Ffiltered)))

    plt.subplot(2,2,4).set_title("Filtered Image")        
    plt.imshow(np.log(1+np.abs(f_filtered)), cmap="gray")
    plt.show()

    return f_filtered



def sinc_implementation(image, deg=0, SNR_p=0.925, d_p=0.005):
    
    F = fft2(image)
    Fshifted = fftshift(F)
    
    plt.figure().suptitle('sinc') #figure name
    
    if(deg!=0):
        rotImage=scipy.ndimage.rotate(image,deg)
    else:
        rotImage=image
    
    F = fft2(rotImage)
    Fshifted = fftshift(F)
    H = np.ones_like(F)
    spectrum_center = [x // 2 for x in F.shape]
    plt.subplot(2,2,1).set_title("Basic Fourier")
    plt.imshow(np.log(1+np.abs(Fshifted)),cmap="gray")    
    
    

    d=d_p #more float digits means the top of the sin is larger, and higher number means smoother transition
    if(deg>0):
        for u in range(120, len(F)-120):
            for v in range(0,len(F[0])):
                H[u,v]=np.sinc(np.pi*(u-spectrum_center[0])*d)
    
    elif(deg==0):
        for u in range(180, len(F)-180):
            for v in range(0,len(F[0])):
                H[u,v]=np.sinc(np.pi*(u-spectrum_center[0])*d)
    
    SNR=SNR_p #higher value means more aggressive filter
    
    W=(np.conj(H))/(np.power(np.abs(H),2)+(1/SNR))
    Ffiltered_shifted = (Fshifted)*W
    
    plt.subplot(2,2,2).set_title("Mask")
    plt.imshow(np.log(1+np.abs(H)),cmap="gray")          

    
    plt.subplot(2,2,3).set_title("Masked Fourier")
    plt.imshow(np.log(1+np.abs(Ffiltered_shifted)),cmap="gray")      
    
    Ffiltered = ifftshift(Ffiltered_shifted)
    
    f_filtered = np.real((ifft2(Ffiltered)))
    
    
    if(deg==0):
        f_filtered2=np.copy(f_filtered)
    if(deg==116):
        f_filtered=scipy.ndimage.rotate(f_filtered,np.negative(deg))
        f_filtered2=f_filtered[209:603,155:686]
    if(deg!=116 and deg!=0):
        f_filtered=scipy.ndimage.rotate(f_filtered,np.negative(deg))
        f_filtered2=f_filtered[197:596,147:678]    
    
    plt.subplot(2,2,4).set_title("Filtered Image")        
    plt.imshow(np.log(1+np.abs(f_filtered2)), cmap="gray")
    plt.show()
    
    return f_filtered2

def main():
    init=np.array(initialize)
    
    motion=sinc_implementation(init, deg=116, d_p=0.009, SNR_p=15)
    initDif=init-motion
    #plt.imshow(initDif,cmap="gray")      In case you wanna see each part
    #plt.show()    
    
    filtered=sinc_implementation(motion, d_p=0.0089)   
    signalPrime=initDif-filtered
    #plt.imshow(signalPrime,cmap="gray")  In case you wanna see each part    
    #plt.show()
    
    signal=motion_degration(filtered, T_p=150, b_p=50)
    signalRes=signalPrime-signal    
    signalRes=255-signalRes
    signalRes=np.uint8(signalRes)
    
    plt.imshow(signalRes,cmap="gray")
    plt.show()    
    
    imToSave = np.copy(signalRes)
    imToSave = imToSave - np.min(imToSave)
    imToSave = np.float32(imToSave)
    imToSave = np.round( 255. * (imToSave / np.max(imToSave)) )
    imToSave = np.uint8(imToSave)
    Image.fromarray(imToSave).save('filtered_res.png')
    
if __name__ == "__main__": main()
