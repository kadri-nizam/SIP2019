"""Name: py_specrebin.py
Author: K. M. Hamren
Date: Jan. 2014
Purpose: A python version of x_specrebin, modified such that input and output fluxes are the same
Examples:

1. Put in just the required arguments, return new rebinned flux array
f2 = rebinspec(l,f,l2)

2. Put in required arguments + variance, return new rebinned flux array and a rebinned variance array
f2, var2 = rebinspec(l,f,l2,var = var)

3. Put in required arguments + ivar, return new rebinned flux array and a rebinned ivar
f2, ivar2 = rebinspec(l,f,l2,ivar = ivar)
"""
import pdb
import numpy as np
import warnings

warnings.simplefilter("ignore", RuntimeWarning)


def spec_interp(wv,fx,nwwv,*args):
    #Required arguments:
    #   - wv: old wavelength array
    #   - fx: flux to be rebinned
    #   - nwwv: new wavelength array
    #
    #Optional argument: variance

    #Optional var argument
    npix = len(wv)
    if len(args) == 0:
        var = np.ones(npix)
        nwvarFlag = False
    else:
        var = args[0]
        nwvarFlag = True

    nwpix = len(nwwv)

    #Calculate wavelength endpoints for each pixel

    wvl = (wv + np.roll(wv,1))/2.
    wvh = (wv + np.roll(wv,-1))/2.
    wvl[0] = wv[0] - (wv[1] - wv[0])/2.
    wvh[npix-1] = wv[npix-1] + (wv[npix-1]-wv[npix-2])/2.

    #Calculate endpoints of the final array
    bwv = np.zeros(nwpix+1)
    bwv[0:nwpix] = (nwwv+np.roll(nwwv,1))/2.
    bwv[0] = nwwv[0] - (nwwv[1] - nwwv[0])/2.
    bwv[nwpix] = nwwv[nwpix-1]+(nwwv[nwpix-1] - nwwv[nwpix - 1])/2.

    #Create tmp arrays for final array

    nwfx = np.zeros(nwpix)
    nwvar = np.zeros(nwpix)
    nwunitfx = np.zeros(nwpix)

    #Loop through the arrays
    for q in range(npix):

        #No overlap
        if (wvh[q] <= bwv[0]) | (wvl[q] >= bwv[nwpix]):
            continue

        #Find pixel that bw is within
        if wvl[q] <= bwv[0]:
            i1 = [0]
        else:
            i1 = np.argwhere((wvl[q] <= np.roll(bwv,-1)) & (wvl[q] > bwv))[0]

        if wvh[q] > bwv[nwpix]:
            i2 = [nwpix-1]
        else:
            i2 = np.argwhere((wvh[q] <= np.roll(bwv,-1)) & (wvh[q] > bwv))[0]

        j1 = i1[0]
        j2 = i2[0]

        #Now Sum up
        for kk in range(j1,j2+1):
            #Rejected pixesl do not get added in
            if var[q] > 0.:
                frac = ( np.min([wvh[q],bwv[kk+1]]) - np.max([wvl[q],bwv[kk]]) ) / (wvh[q]-wvl[q])
                nwfx[kk] = nwfx[kk]+frac*fx[q]
                nwunitfx[kk] = nwunitfx[kk]+frac*1.0

                #Variance
                if nwvarFlag:
                    if (var[q] <= 0.) | (nwvar[kk] == -1):
                       nwvar[kk] = -1
                    else:
                       nwvar[kk] = nwvar[kk]+frac*var[q]

    if nwvarFlag:
        fxOut = nwfx/nwunitfx
        varOut = nwvar*nwunitfx
        
        return fxOut,varOut
    else:
        fxOut = nwfx/nwunitfx
        return fxOut


def rebinspec(*args,**kwargs):
    #Required arguments:
    #   - wv: old wavelength array
    #   - fx: flux to be rebinned
    #   - nwwv: new wavelength array
    #
    #Optional arguments:
    #   - var = var, input and output variance
    #   - ivar = ivar, input and output ivar

    if len(args) != 3:
        print('Proper syntax is: out = rebinspec(wv, fx, nwwv, **kwargs)')
        return np.nan

    else:
        wv, fx, nwwv = args

        var = kwargs.get('var',None)
        ivar = kwargs.get('ivar',None)

        if (var is not None) & (ivar is None):
            nwfx,nwvar = spec_interp(wv,fx,nwwv,var)

            return nwfx, nwvar
        elif (var is None) & (ivar is not None):
            var = 1./ivar
            nwfx,nwvar_1 = spec_interp(wv,fx,nwwv,var)
            nwvar_1[nwvar_1 == 0.0] = -10.0
            nwivar = 1.0/nwvar_1
            nwivar[nwivar < 0.0] = 0.0
            
            return nwfx, nwivar
        else:
            nwfx = spec_interp(wv,fx,nwwv)

            return nwfx
        

        
        
        

    
