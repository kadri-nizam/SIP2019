#program to run the V6.7.16 June/2019 of pPXF over the VUGC spectra.
#Elisa Toloba, UoP, June 28th 2019
#---------------------------------------
from __future__ import print_function

from astropy.io import fits
from astropy.io import ascii
import pdb
import numpy as np
import glob
from time import perf_counter as clock

from py_specrebin import rebinspec
from smooth_gauss import gauss_ivar 

import ppxf as ppxf_package
from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util

import matplotlib.pyplot as plt

from astropy.convolution import Gaussian1DKernel, convolve
#---------------------------------------
def ppxf_VUGC():

    #READ THE TEMPLATES
    files=glob.glob('/Users/etoloba/Dropbox/python/ppxf/DEIMOS_600/template*fits') #remove the a band template
    templates_names=[]
    aband_temp_name=[]
    for i in range(len(files)):        
        if 'aband' not in files[i]:
            templates_names.append(files[i])
        else:
            aband_temp_name.append(files[i])

    ntemp=len(templates_names)

    head=fits.open(templates_names[0])[0].header
    lam_temp=head['CRVAL1']+head['CDELT1']*np.arange(head['NAXIS1'])

    #READ THE CONFIGURATION FILE
    c = 299792.458
    conf=ascii.read('DEIMOS_spr19.conf')
    nobj=len(conf) #number of objects

    for i in range(nobj): #run ppxf for each object
        #read the object (deimos file red and blue sides independently)
        file=fits.open('../data/'+conf['Object'][i])
        v0=conf['Vel'][i]
        z=v0/c

        lambmin=conf['LambMin'][i]
        lambmax=conf['LambMax'][i]

        stddev=conf['stddev'][i]
        #-----------------------------------------------------
        #READ THE OBJECT
        specb=file[1].data
        specr=file[2].data
        if np.max(specr['LAMBDA'])<8000: specr=file[3].data
        if np.max(specr['LAMBDA'])<8000: break

        #generate the final wavelength array for the object (make it the same as the templates)
        lambmin_blue=np.min(specb['lambda'][0])
        lambmax_red=np.max(specr['lambda'][0])
        dlam=specb['lambda'][0][1]-specb['lambda'][0][0]
        lam_gal=np.arange(lambmin_blue,lambmax_red,dlam)
        
        specrebinb, ivarrebinb=rebinspec(specb['LAMBDA'][0], specb['SPEC'][0], lam_gal, ivar=specb['IVAR'][0])
        specrebinr, ivarrebinr=rebinspec(specr['LAMBDA'][0], specr['SPEC'][0], lam_gal, ivar=specr['IVAR'][0])
        skyrebinb =rebinspec(specb['LAMBDA'][0], specb['SKYSPEC'][0], lam_gal)
        skyrebinr =rebinspec(specr['LAMBDA'][0], specr['SKYSPEC'][0], lam_gal)
        specrebinb=np.nan_to_num(specrebinb)#change nan to 0
        specrebinr=np.nan_to_num(specrebinr)
        ivarrebinb=np.nan_to_num(ivarrebinb)
        ivarrebinr=np.nan_to_num(ivarrebinr)
        skyrebinb=np.nan_to_num(skyrebinb)
        skyrebinr=np.nan_to_num(skyrebinr)
        flux=specrebinb+specrebinr
        ivar=ivarrebinb+ivarrebinr
        noise=1/np.sqrt(ivar)
        sky=skyrebinb+skyrebinr

        #remove NaN, inf
        noise[(noise!=noise)]=0.
        noise[(noise>10e6)]=999.
        noise[(noise<10e-6)]=0.

        #remove negative values of flux
        flux[(flux<0)]=0.

        #save values for the a-band calculation
        aband_flux=flux
        aband_noise=noise
        aband_lam=lam_gal

        #cut the object
        ind=np.searchsorted(lam_gal,lambmin,side='right')
        lambmin=lam_gal[ind]
        ind=np.searchsorted(lam_gal,lambmax,side='left')
        lambmax=lam_gal[ind]

        mask_gal=((lam_gal>=lambmin) & (lam_gal<=lambmax))
        flux=flux[mask_gal]
        ivar=ivar[mask_gal]
        noise=noise[mask_gal]
        sky=sky[mask_gal]
        lam_gal=lam_gal[mask_gal]

        #a-band region:
        ind=np.searchsorted(lam_gal,6800,side='right')
        lambmin_aband=lam_gal[ind]
        ind=np.searchsorted(lam_gal,7750,side='left')
        lambmax_aband=lam_gal[ind]

        mask_aband=((aband_lam>=lambmin_aband) & (aband_lam<=lambmax_aband))
        aband_flux=aband_flux[mask_aband]
        aband_noise=aband_noise[mask_aband]
        aband_lam=aband_lam[mask_aband]
        lamRange_aband=[np.min(aband_lam),np.max(aband_lam)]

        aband_frac=aband_lam[1]/aband_lam[0]
        aband_velscale=np.log(aband_frac)*c #resolution in km/s 
        
        #-----------------------------------------------------
        #get the wavelength region for the templates:
        lamRange_temp=[np.min(lam_temp), np.max(lam_temp)]
        
        #a-band template:
        flux_aband_temp=fits.open(aband_temp_name[0])[0].data
        aband_temp=util.log_rebin(lamRange_temp, flux_aband_temp[0], velscale=aband_velscale)[0]

        #remove NaN, inf
        aband_temp[(aband_temp!=aband_temp)]=0.
        aband_temp[(aband_temp>10e6)]=999.
        aband_temp[(aband_temp<10e-6)]=0.
        
        #-----------------------------------------------------
        #log rebin the a-band region of the object
        lin_aband_lam=aband_lam
        lin_aband_flux=aband_flux
        lin_aband_noise=aband_noise
        
        logaband_flux, aband_logLam, kkk=util.log_rebin(lamRange_aband, aband_flux, velscale=aband_velscale)
        logaband_noise=util.log_rebin(lamRange_aband, aband_noise, velscale=aband_velscale)[0]
        aband_flux=logaband_flux/np.median(logaband_flux)
        aband_noise=logaband_noise/np.median(logaband_flux)
        ppxf_aband_noise=np.full_like(aband_flux, 0.0166) #assume constant noise (we'll use the real noise in the MC only)
        aband_flux[(aband_flux!=aband_flux)]=0.
        
        #put the region in between the a and b bands to flux=0:
        btw=np.where((aband_logLam>np.log(7000))&(aband_logLam<np.log(7500)))
        aband_flux[btw]=0
        
        #goodpixels=util.determine_goodpixels(logLam, lamRange, z) #only when you need to remove emission lines
        aband_goodpixels=np.array(range(len(aband_logLam)))
        
        #remove those pixels without flux from the a band flux:
        remove=np.where(aband_flux==0)[0] 
        index=[]

        for j in range(len(remove)):
            for k in range(len(aband_goodpixels)):
                if remove[j]==aband_goodpixels[k]:
                    index.append(k)

        aband_goodpixels=np.delete(aband_goodpixels,index)
        
        #-----------------------------------------------------
        #A-BAND MEASUREMENT:
        vel=0.
        start=[vel,3*aband_velscale]
        dv = np.log(lam_temp[0]/aband_lam[0])*c    # km/s
        t=clock()
        
        if sum(aband_flux)>0.:
            aband_pp=ppxf(aband_temp, aband_flux, aband_noise, aband_velscale, start,
                    goodpixels=aband_goodpixels, plot=True, moments=2,
                    degree=12, vsyst=dv, clean=False,lam=np.exp(aband_logLam), quiet=1)

            v_aband=aband_pp.sol[0]
            s_aband=aband_pp.sol[1]
        else:
            v_aband=0.
            s_aband=0.
        print(conf['Object'][i],'Vel=',round(v_aband,2),' sigma=',round(s_aband,2))
        #-----------------------------------------------------
        #MC SIMULATIONS:
        nsimul=100
        v_aband_MC=np.zeros((nsimul))
        s_aband_MC=np.zeros((nsimul))
        if sum(aband_flux)>0.: #there may be some slits where the blue chip crassed and we don't have flux for the a band
            start_MC=aband_pp.sol
            for j in range(nsimul):
                print(conf['Object'][i],'A-band Monte Carlo:',j)
                flux_aband_MC=np.zeros((len(aband_flux)))
                for k in range(len(aband_flux)):
                    #flux_aband_MC[k]=aband_flux[k]+np.random.normal(aband_flux[k],aband_noise[k])
                    flux_aband_MC[k]=aband_flux[k]+np.random.normal(aband_flux[k],abs(aband_noise[k]))

                pp_aband_MC=ppxf(aband_temp, flux_aband_MC/np.median(flux_aband_MC), ppxf_aband_noise, aband_velscale, start_MC,
                    goodpixels=aband_goodpixels, plot=True, moments=2,
                    degree=12, vsyst=dv, clean=False, lam=np.exp(aband_logLam), quiet=1)
                v_aband_MC[j]=pp_aband_MC.sol[0]
                s_aband_MC[j]=pp_aband_MC.sol[1]
        else:
            v_aband_MC=0.
            s_aband_MC=0.
            
        print('Median velocity:', round(np.median(v_aband_MC),2), 'Median sigma:', round(np.median(s_aband_MC),2))
        print('Velocity errors:', round(np.median(v_aband_MC)-np.percentile(v_aband_MC,16),2), round(np.percentile(v_aband_MC,84)-np.median(v_aband_MC),2))
        print('Sigma errors:', round(np.median(s_aband_MC)-np.percentile(s_aband_MC,16),2), round(np.percentile(s_aband_MC,84)-np.median(s_aband_MC),2))
        print('Elapsed time in PPXF: %.2f s' % (clock() - t))

        #-----------------------------------------------------
        #A-BAND CORRECTION

        #keep only the parameter values (drop the uncertainties)
        deltaLambPolyVals = np.load('galaxy_LSF_output/COSMOS.deltaLambdaFit.npy')[:,0]
        deltaLambFunc = np.poly1d(deltaLambPolyVals) #this is the LSF 
        lsf = deltaLambFunc(lam_gal) #returns the value of the parabola at each wavelength
        delta_lamb=v_aband*7600*lsf/c/deltaLambFunc(7600) #we use 7600 because it's the wavelength for the Aband, where we calculate the correction
        lam_gal2=lam_gal+delta_lamb #Now, the step between consequtive pixels is not constant
        flux2=flux
        
        flux_nw, ivar_nw=rebinspec(lam_gal2, flux, lam_gal, ivar=1/(noise**2))
        sky_nw=rebinspec(lam_gal2, sky, lam_gal)
        flux=np.nan_to_num(flux_nw)#change nan to 0
        ivar=np.nan_to_num(ivar_nw)
        sky=np.nan_to_num(sky_nw)
        noise=1/np.sqrt(ivar)

        #remove NaN, inf
        noise[(noise!=noise)]=0.
        noise[(noise>10e6)]=999.
        noise[(noise<10e-6)]=0.

        #remove negative values of flux
        flux[(flux<0)]=0.
        sky[(sky<0)]=0.
        
        #log rebin the object
        lamRange_gal=[np.min(lam_gal),np.max(lam_gal)]
        frac=lam_gal[1]/lam_gal[0]
        velscale=np.log(frac)*c #resolution in km/s (deltaLambda/lambda)
        lin_lam=lam_gal
        lin_flux=flux
        lin_noise=noise
       
        logflux, logLam, kk=util.log_rebin(lamRange_gal, flux, velscale=velscale)
        lognoise=util.log_rebin(lamRange_gal, noise, velscale=velscale)[0]
        logsky=util.log_rebin(lamRange_gal, sky, velscale=velscale)[0]
        logflux[(logflux!=logflux)]=0.
        flux=logflux/np.median(logflux[(logflux>0)])
        noise=lognoise/np.median(logflux[(logflux>0)])
        ppxf_noise=np.full_like(flux, 0.0166) #assume constant noise (we'll use the real noise in the MC only)
        logsky[(logsky!=logsky)]=0.
        sky=logsky/np.median(logsky[(logsky>0)])
        
        #remove the a-band and b-band by putting them to flux=0: (this is not needed here because the templates don't have these bands in them)
        aband=np.where((logLam>np.log(7550))&(logLam<np.log(7700)))[0] #lambda affected
        bband=np.where((logLam>np.log(6800))&(logLam<np.log(7000)))[0] #b-band 6800-7000, earlier beginning to remove the cup shape
        flux[aband]=0.
        flux[bband]=0.
        sky[aband]=0.
        sky[bband]=0.

        #goodpixels=util.determine_goodpixels(logLam, lamRange, z) #only when you need to remove emission lines
        goodpixels=np.array(range(len(logLam)))

        #remove the chip gap and other pixels without flux:
        remove=np.where(flux==0)[0] 
        index=[]

        for j in range(len(remove)):
            for k in range(len(goodpixels)):
                if remove[j]==goodpixels[k]:
                    index.append(k)

        goodpixels=np.delete(goodpixels,index)

        #rebin in log and save the templates together
        flux_temp0=fits.open(templates_names[0])[0].data
        flux_temp0=flux_temp0[0]
        loglam=util.log_rebin(lamRange_temp, flux_temp0, velscale=velscale)[0]
        templates=np.empty((len(loglam), ntemp))
    
        for j in range(ntemp):
            flux_temp=fits.open(templates_names[j])[0].data
            newflux=util.log_rebin(lamRange_temp, flux_temp[0], velscale=velscale)[0]
            templates[:,j]=newflux

        #remove NaN, inf
        templates[(templates!=templates)]=0.
        templates[(templates>10e6)]=999.
        templates[(templates<10e-6)]=0.
        
        #-----------------------------------------------------
        #RUNNING PPXF
        vel=c*np.log(1+z)
        start=[vel,3*velscale]
        dv = np.log(lam_temp[0]/lam_gal[0])*c    # km/s
        

        pp=ppxf(templates, flux, ppxf_noise, velscale, start,
                goodpixels=goodpixels, plot=True, moments=2, #sky=sky,
                degree=12, vsyst=dv, clean=False,lam=np.exp(logLam), quiet=1)

        print(conf['Object'][i],'Vel=',round(pp.sol[0],2),' sigma=',round(pp.sol[1],2))

        #-----------------------------------------------------
        #MC SIMULATIONS:
        nsimul=100
        v_MC=np.zeros((nsimul))
        s_MC=np.zeros((nsimul))
        start_MC=pp.sol
        for j in range(nsimul):
            print(conf['Object'][i],'Monte Carlo:',j)
            flux_MC=np.zeros((len(flux)))
            for k in range(len(flux)):
                #flux_MC[k]=flux[k]+np.random.normal(flux[k],noise[k])
                flux_MC[k]=flux[k]+np.random.normal(flux[k],abs(noise[k]))
              
            pp_MC=ppxf(templates, flux_MC/np.median(flux_MC), ppxf_noise, velscale, start_MC,
                goodpixels=goodpixels, plot=True, moments=2,#sky=sky,
                degree=12, vsyst=dv, clean=False, lam=np.exp(logLam), quiet=1)
            v_MC[j]=pp_MC.sol[0]
            s_MC[j]=pp_MC.sol[1]
        print('Median velocity:', round(np.median(v_MC),2), 'Median sigma:', round(np.median(s_MC),2))
        print('Velocity errors:', round(np.median(v_MC)-np.percentile(v_MC,16),2), round(np.percentile(v_MC,84)-np.median(v_MC),2))
        print('Sigma errors:', round(np.median(s_MC)-np.percentile(s_MC,16),2), round(np.percentile(s_MC,84)-np.median(s_MC),2))
        
        noise[aband]=0.
        noise[bband]=0.
        print('S/N per pixel:', round(np.median(flux)/np.median(noise),2))
       
        #-----------------------------------------------------
        #SAVE THE RESULTS IN AN ASCII FILE

        f=open('../ppxf/results/'+conf['Object'][i]+'.dat','w')
        f.write('#S/N  V  V_MC  Vpe  Vne  S  S_MC  Spe  Sne  Aband  Aband_MC  Aband_pe  Aband_ne \n')
        f.write( str(round(np.median(flux)/np.median(noise),2))+'  '+str(round(pp.sol[0],2))+'  '+str(round(np.median(v_MC),2))+'  '+str(round(np.percentile(v_MC,84)-np.median(v_MC),2))+'  '+str(round(np.median(v_MC)-np.percentile(v_MC,16),2))+'  '+str(round(pp.sol[1],2))+'  '+str(round(np.median(s_MC),2))+'  '+str(round(np.percentile(s_MC,84)-np.median(s_MC)))+'  '+str(round(np.median(s_MC)-np.percentile(s_MC,16),2))+'  '+str(round(v_aband,2))+'  '+str(round(np.median(v_aband_MC),2))+'  '+str(round(np.percentile(v_aband_MC,84)-np.median(v_aband_MC),2))+'  '+str(round(np.median(v_aband_MC)-np.percentile(v_aband_MC,16),2))+'\n')
        f.close()
        
        #-----------------------------------------------------
        #SAVE PLOTS
        from matplotlib.backends.backend_pdf import PdfPages #make pdfs with multiple pages
        
        with PdfPages('../ppxf/figures/'+conf['Object'][i]+'.pdf') as pdf:
            
            mn, mx = np.min(pp.bestfit[pp.goodpixels]), np.max(pp.bestfit[pp.goodpixels])
            resid=mn+pp.galaxy-pp.bestfit
            mn1=np.min(resid[pp.goodpixels])
            ll, rr = np.min(pp.lam), np.max(pp.lam)

            g=Gaussian1DKernel(stddev=stddev)
            sm_flx,sm_ivar=gauss_ivar(lin_lam,lin_flux,1/lin_noise**2,stddev) #smoothed flux weighted by ivar
            sm_flx=sm_flx/np.median(sm_flx)
            
            plt.figure()
            plt.plot(pp.lam[pp.goodpixels], resid[pp.goodpixels],marker='d',color='LimeGreen',mec='LimeGreen', markersize=4, linestyle='None')
            plt.plot(lin_lam,sm_flx,'k')
            plt.plot(pp.lam,convolve(pp.bestfit, g, boundary='extend'),'r',linewidth=2)
            plt.plot(pp.lam[pp.goodpixels],pp.goodpixels*0+mn, '.k', ms=1)
            w=np.flatnonzero(np.diff(pp.goodpixels) > 1)
            if w.size > 0:
                for wj in w:
                    j = slice(pp.goodpixels[wj], pp.goodpixels[wj+1] + 1)
                    plt.plot(pp.lam[j], resid[j], 'b')
                w = np.hstack([0, w, w + 1, -1])  # Add first and last point
            else:
                w = [0, -1]
            for gj in pp.goodpixels[w]:
                plt.plot(pp.lam[[gj, gj]], [mn, pp.bestfit[gj]], 'LimeGreen')
            
            plt.xlim([ll, rr]+ np.array([-0.02, 0.02])*(rr-ll))
            plt.ylim([mn1, mx]+np.array([-0.05, 0.05])*(mx-mn1))
            plt.ylabel("Counts")
            plt.xlabel("$\lambda$ ($\AA$)")
            plt.title('S/N='+str(round(np.median(flux)/np.median(noise),2)))
            plt.tight_layout
            pdf.savefig()
            plt.close()

            #make zoom-ins:

            z=pp.sol[0]/c
                        
            if np.min(pp.lam) < 4800:
                lmin=4800
            else:
                lmin=np.min(pp.lam)
            npix=(5250-lmin) #Hbeta and MgT region

            if lmin < 5250:
                lamfin_p1=lmin+npix
                mx_p1=np.max(pp.galaxy[pp.goodpixels][(pp.lam[pp.goodpixels]<lamfin_p1)])
                mn1_p1=np.min(resid[pp.goodpixels][(pp.lam[pp.goodpixels]<lamfin_p1)])

                plt.figure()
                plt.plot(pp.lam[pp.goodpixels]/(1+z), resid[pp.goodpixels],marker='d',color='LimeGreen',mec='LimeGreen', markersize=4, linestyle='None')
                plt.plot(lin_lam/(1+z),sm_flx,'k')
                plt.plot(pp.lam/(1+z),convolve(pp.bestfit, g, boundary='extend'),'r',linewidth=1)
                plt.plot(pp.lam[pp.goodpixels]/(1+z),pp.goodpixels*0+mn, '.k', ms=1)
                w=np.flatnonzero(np.diff(pp.goodpixels) > 1)
                if w.size > 0:
                    for wj in w:
                        j = slice(pp.goodpixels[wj], pp.goodpixels[wj+1] + 1)
                        plt.plot(pp.lam[j]/(1+z), resid[j], 'b')
                    w = np.hstack([0, w, w + 1, -1])  # Add first and last point
                else:
                    w = [0, -1]
                for gj in pp.goodpixels[w]:
                    plt.plot(pp.lam[[gj, gj]]/(1+z), [mn, pp.bestfit[gj]], 'LimeGreen')

                plt.plot([4861,4861],[mn1_p1-0.05*(mx_p1-mn1_p1),mx_p1+0.05*(mx_p1-mn1_p1)],'k', linestyle=':')#Hbeta
                plt.plot([5015,5015],[mn1_p1-0.05*(mx_p1-mn1_p1),mx_p1+0.05*(mx_p1-mn1_p1)],'k', linestyle=':')#Fe5015
                plt.plot([5167,5167],[mn1_p1-0.05*(mx_p1-mn1_p1),mx_p1+0.05*(mx_p1-mn1_p1)],'k', linestyle=':')#Mg1
                plt.plot([5173,5173],[mn1_p1-0.05*(mx_p1-mn1_p1),mx_p1+0.05*(mx_p1-mn1_p1)],'k', linestyle=':')#Mg2
                plt.plot([5184,5184],[mn1_p1-0.05*(mx_p1-mn1_p1),mx_p1+0.05*(mx_p1-mn1_p1)],'k', linestyle=':')#Mg3
                plt.xlim([lmin, lamfin_p1]+ np.array([-0.02, 0.02])*(lamfin_p1-lmin))
                plt.ylim([mn1_p1, mx_p1]+np.array([-0.05, 0.05])*(mx_p1-mn1_p1))
                plt.ylabel("Counts")
                plt.xlabel("$\lambda$ ($\AA$)")
                plt.tight_layout
                pdf.savefig()
                plt.close()

            if sum(aband_flux)>0. and min(pp.lam[pp.goodpixels])> 6100. and min(pp.lam[pp.goodpixels])< 6750.:
                lmin2=6100
                npix=(6750-lmin2)
                lamfin_p2=lmin2+npix

                mx_p2=np.max(pp.galaxy[pp.goodpixels][(pp.lam[pp.goodpixels]>lmin2)&(pp.lam[pp.goodpixels]<lamfin_p2)])
                mn1_p2=np.min(resid[pp.goodpixels][(pp.lam[pp.goodpixels]>lmin2)&(pp.lam[pp.goodpixels]<lamfin_p2)])
            
                plt.figure()
                plt.plot(pp.lam[pp.goodpixels]/(1+z), resid[pp.goodpixels],marker='d',color='LimeGreen',mec='LimeGreen', markersize=4, linestyle='None')
                plt.plot(lin_lam/(1+z),sm_flx,'k')
                plt.plot(pp.lam/(1+z),convolve(pp.bestfit, g, boundary='extend'),'r',linewidth=1)
                plt.plot(pp.lam[pp.goodpixels]/(1+z),pp.goodpixels*0+mn, '.k', ms=1)
                w=np.flatnonzero(np.diff(pp.goodpixels) > 1)
                if w.size > 0:
                    for wj in w:
                        j = slice(pp.goodpixels[wj], pp.goodpixels[wj+1] + 1)
                        plt.plot(pp.lam[j]/(1+z), resid[j], 'b')
                    w = np.hstack([0, w, w + 1, -1])  # Add first and last point
                else:
                    w = [0, -1]
                for gj in pp.goodpixels[w]:
                    plt.plot(pp.lam[[gj, gj]]/(1+z), [mn, pp.bestfit[gj]], 'LimeGreen')

                plt.plot([6563,6563],[mn1_p2-0.05*(mx_p2-mn1_p2),mx_p2+0.05*(mx_p2-mn1_p2)],'k', linestyle=':')#Halpha
                plt.xlim([lmin2, lamfin_p2]+ np.array([-0.02, 0.02])*(lamfin_p2-lmin2))
                plt.ylim([mn1_p2, mx_p2]+np.array([-0.05, 0.05])*(mx_p2-mn1_p2))
                plt.ylabel("Counts")
                plt.xlabel("$\lambda$ ($\AA$)")
                plt.tight_layout
                pdf.savefig()
                plt.close()

            lmin3=8300
            npix=(8800-lmin3)
            lamfin_p3=lmin3+npix

            mx_p3=np.max(pp.galaxy[pp.goodpixels][(pp.lam[pp.goodpixels]>lmin3)&(pp.lam[pp.goodpixels]<lamfin_p3)])
            mn1_p3=np.min(resid[pp.goodpixels][(pp.lam[pp.goodpixels]>lmin3)&(pp.lam[pp.goodpixels]<lamfin_p3)])
            
            plt.figure()
            plt.plot(pp.lam[pp.goodpixels]/(1+z), resid[pp.goodpixels],marker='d',color='LimeGreen',mec='LimeGreen', markersize=4, linestyle='None')
            plt.plot(lin_lam/(1+z),sm_flx,'k')
            plt.plot(pp.lam/(1+z),convolve(pp.bestfit, g, boundary='extend'),'r',linewidth=1)
            plt.plot(pp.lam[pp.goodpixels]/(1+z),pp.goodpixels*0+mn, '.k', ms=1)
            w=np.flatnonzero(np.diff(pp.goodpixels) > 1)
            if w.size > 0:
                for wj in w:
                    j = slice(pp.goodpixels[wj], pp.goodpixels[wj+1] + 1)
                    plt.plot(pp.lam[j]/(1+z), resid[j], 'b')
                w = np.hstack([0, w, w + 1, -1])  # Add first and last point
            else:
                w = [0, -1]
            for gj in pp.goodpixels[w]:
                plt.plot(pp.lam[[gj, gj]]/(1+z), [mn, pp.bestfit[gj]], 'LimeGreen')

            plt.plot([8498,8498],[mn1_p3-0.05*(mx_p3-mn1_p3),mx_p3+0.05*(mx_p3-mn1_p3)],'k', linestyle=':')#Ca1
            plt.plot([8542,8542],[mn1_p3-0.05*(mx_p3-mn1_p3),mx_p3+0.05*(mx_p3-mn1_p3)],'k', linestyle=':')#Ca2
            plt.plot([8662,8662],[mn1_p3-0.05*(mx_p3-mn1_p3),mx_p3+0.05*(mx_p3-mn1_p3)],'k', linestyle=':')#Ca1
            plt.xlim([lmin3, lamfin_p3]+ np.array([-0.02, 0.02])*(lamfin_p3-lmin3))
            plt.ylim([mn1_p3, mx_p3]+np.array([-0.05, 0.05])*(mx_p3-mn1_p3))
            plt.ylabel("Counts")
            plt.xlabel("$\lambda$ ($\AA$)")
            plt.tight_layout
            pdf.savefig()
            plt.close()

            # velocity MC histogram
            plt.figure()
            n, bins, patches=plt.hist(v_MC, facecolor='LightBlue')
            plt.plot([np.median(v_MC),np.median(v_MC)], [0,np.max(n)+0.1*np.max(n)], 'k', linestyle='--')
            plt.plot([np.percentile(v_MC,16),np.percentile(v_MC,16)], [0,np.max(n)+0.1*np.max(n)], 'k', linestyle=':')
            plt.plot([np.percentile(v_MC,84),np.percentile(v_MC,84)], [0,np.max(n)+0.1*np.max(n)], 'k', linestyle=':')
            plt.title(str(round(np.median(v_MC),2))+'( +'+str(round(np.percentile(v_MC,84)-np.median(v_MC),2))+' -'+str(round(np.median(v_MC)-np.percentile(v_MC,16),2))+')')
            plt.ylim(0., np.max(n)+0.1*np.max(n))
            plt.ylabel("Number")
            plt.xlabel("V (km/s)")
            plt.tight_layout
            pdf.savefig()
            plt.close()

            # sigma MC histogram
            plt.figure()
            n, bins, patches=plt.hist(s_MC, facecolor='Plum')
            plt.plot([np.median(s_MC),np.median(s_MC)], [0,np.max(n)+0.1*np.max(n)], 'k', linestyle='--')
            plt.plot([np.percentile(s_MC,16),np.percentile(s_MC,16)], [0,np.max(n)+0.1*np.max(n)], 'k', linestyle=':')
            plt.plot([np.percentile(s_MC,84),np.percentile(s_MC,84)], [0,np.max(n)+0.1*np.max(n)], 'k', linestyle=':')
            plt.title(str(round(np.median(s_MC),2))+'( +'+str(round(np.percentile(s_MC,84)-np.median(s_MC),2))+' -'+str(round(np.median(s_MC)-np.percentile(s_MC,16),2))+')')
            plt.ylim(0., np.max(n)+0.1*np.max(n))
            plt.ylabel("Number")
            plt.xlabel("$\sigma$ (km/s)")
            plt.tight_layout
            pdf.savefig()
            plt.close()

            #A-band fit

            sm_aband_flx,sm_aband_ivar=gauss_ivar(lin_aband_lam,lin_aband_flux,1/lin_aband_noise**2,stddev) #smoothed flux weightedby ivar
            sm_aband_flx=sm_aband_flx/np.median(sm_aband_flx)
            sm_aband_flx[btw]=0
        
            if sum(aband_flux)>0:
                mn, mx = np.min(aband_pp.bestfit[aband_pp.goodpixels]), np.max(aband_pp.bestfit[aband_pp.goodpixels])
                resid=mn+aband_pp.galaxy-aband_pp.bestfit
                mn1=np.min(resid[aband_pp.goodpixels])
                ll, rr = np.min(aband_pp.lam), np.max(aband_pp.lam)

                plt.figure()
                plt.plot(aband_pp.lam[aband_pp.goodpixels], resid[aband_pp.goodpixels],marker='d',color='LimeGreen',mec='LimeGreen', markersize=4, linestyle='None')
                plt.plot(lin_aband_lam,sm_aband_flx,'k')
                plt.plot(aband_pp.lam,convolve(aband_pp.bestfit, g, boundary='extend'),'r',linewidth=2)
                plt.plot(aband_pp.lam[aband_pp.goodpixels],aband_pp.goodpixels*0+mn, '.k', ms=1)
                w=np.flatnonzero(np.diff(aband_pp.goodpixels) > 1)
                if w.size > 0:
                    for wj in w:
                        j = slice(aband_pp.goodpixels[wj], aband_pp.goodpixels[wj+1] + 1)
                        plt.plot(aband_pp.lam[j], resid[j], 'b')
                    w = np.hstack([0, w, w + 1, -1])  # Add first and last point
                else:
                    w = [0, -1]
                for gj in aband_pp.goodpixels[w]:
                    plt.plot(aband_pp.lam[[gj, gj]], [mn, aband_pp.bestfit[gj]], 'LimeGreen')
                
                plt.xlim([ll, rr]+ np.array([-0.02, 0.02])*(rr-ll))
                plt.ylim([mn1, mx]+np.array([-0.05, 0.05])*(mx-mn1))
                plt.ylabel("Counts")
                plt.xlabel("$\lambda$ ($\AA$)")
                plt.tight_layout
                pdf.savefig()
                plt.close()

                # A-band velocity MC histogram
                plt.figure()
                n, bins, patches=plt.hist(v_aband_MC, facecolor='LightBlue')
                plt.plot([np.median(v_aband_MC),np.median(v_aband_MC)], [0,np.max(n)+0.1*np.max(n)], 'k', linestyle='--')
                plt.plot([np.percentile(v_aband_MC,16),np.percentile(v_aband_MC,16)], [0,np.max(n)+0.1*np.max(n)], 'k', linestyle=':')
                plt.plot([np.percentile(v_aband_MC,84),np.percentile(v_aband_MC,84)], [0,np.max(n)+0.1*np.max(n)], 'k', linestyle=':')
                plt.title(str(round(np.median(v_aband_MC),2))+'( +'+str(round(np.percentile(v_aband_MC,84)-np.median(v_aband_MC),2))+' -'+str(round(np.median(v_aband_MC)-np.percentile(v_aband_MC,16),2))+')')
                plt.ylim(0., np.max(n)+0.1*np.max(n))
                plt.ylabel("Number")
                plt.xlabel("V (km/s)")
                plt.tight_layout
                pdf.savefig()
                plt.close()

                # A-band sigma MC histogram
                plt.figure()
                n, bins, patches=plt.hist(s_aband_MC, facecolor='Plum')
                plt.plot([np.median(s_aband_MC),np.median(s_aband_MC)], [0,np.max(n)+0.1*np.max(n)], 'k', linestyle='--')
                plt.plot([np.percentile(s_aband_MC,16),np.percentile(s_aband_MC,16)], [0,np.max(n)+0.1*np.max(n)], 'k', linestyle=':')
                plt.plot([np.percentile(s_aband_MC,84),np.percentile(s_aband_MC,84)], [0,np.max(n)+0.1*np.max(n)], 'k', linestyle=':')
                plt.title(str(round(np.median(s_aband_MC),2))+'( +'+str(round(np.percentile(s_aband_MC,84)-np.median(s_aband_MC),2))+' -'+str(round(np.median(s_aband_MC)-np.percentile(s_aband_MC,16),2))+')')
                plt.ylim(0., np.max(n)+0.1*np.max(n))
                plt.ylabel("Number")
                plt.xlabel("$\sigma$ (km/s)")
                plt.tight_layout
                pdf.savefig()
                plt.close()
        pdb.set_trace()
#------------------------------------------------------------------------------

if __name__ == '__main__':

    ppxf_VUGC()
    import matplotlib.pyplot as plt
    #plt.pause(0.01)
    #plt.show()

