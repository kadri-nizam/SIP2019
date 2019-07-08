#program to smooth spectra taking into account the flux uncertainties (ivar)
#Elisa Toloba, UOP, August 24th 2018
#-------------------------------
import pdb
import numpy as np
from astropy.io import fits
#-------------------------------
def gauss_ivar(lamb1, spec1, ivar1, sig):

    lamb2=lamb1
    n1=len(lamb1)
    n2=len(lamb2)
    f=np.arange(n1)
    spec2=np.empty(n2)
    ivar2=np.empty(n2)

    dlam=np.repeat(sig, n2)
    dlambda1=np.diff(lamb1)
    maxsigma=4.
    halfwindow=int(np.ceil(1.1*maxsigma*max(dlam)/min(dlambda1)))
    for i in range(n2):
        if f[i]-halfwindow <= 0:
            low=0
        else:
            low=f[i]-halfwindow

        if f[i]+halfwindow < (n1-1) and f[i]+halfwindow > 0:
            high=f[i]+halfwindow
        else:
            high=int(n2-1)
        if low < n1 and low < high:
            w=np.array(np.where(abs(lamb1[low:high+1]-lamb2[i])<dlam[i]*maxsigma))+low
            cw=len(w[0])            
        #the where command in the idl program is saving the positions for the wavelengths that comply the condition within brackets, cw just counts how many positions there are, so length of w. The addition of low is increasing the number of the indices in w by the amount indicated by low.
        #For some reason we have to add high+1 to make it the same as the idl program, this is something internal about how the indices behave, in idl if you say w[0:10] it shows you the first 11 elements, while in python it shows you the first 10.
            if cw > 0:
                gauss=np.exp(-1.*(lamb1[w] - lamb2[i])**2/(2.0*dlam[i]**2))
                temp=ivar1[w]*gauss
                temp2=np.sum(temp)
                spec2[i]=np.sum(spec1[w]*temp)/temp2
                ivar2[i]=temp2**2/np.sum(temp*gauss)

    return spec2, ivar2
#--------------------------------

