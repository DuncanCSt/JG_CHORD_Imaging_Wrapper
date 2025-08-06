import numpy as np
import os
import ctypes
import time
from numpy.ctypeslib import ndpointer
from wrapper_functions import *
import reverb as rev

omega = 2*np.pi/(3600*24) ## in per s

if __name__ == "__main__": ## this 'if' statement prevents this code from running if not
                            # executed as a script
    start_time = time.time()

    ### IMAGING DETAILS ### (fill this out)

    run_name = 'Basic_Example_4D' ## will appear in .fits name

    foldername = '250731' ## this will be the name of the folder the .fits file is put in

    print('Run:' + run_name) ## for slurm diagnostics

    imcenter = (48,26) ## RA, dec of the centre of the image. This also sets the WCS
    cellsize = 0.5/60 ## image pixel sidelength in deg
    imsize = 500 ## pixels on a side
    imparams = np.array([imcenter[0],imcenter[1],cellsize,imsize])
    u,w = rev.gen_image_u(imcenter,cellsize,imsize) ## first argument is ra/dec in deg as tuple, then cellsize, then imsize. Angles in deg

    ### OBSERVATION SPECIFICS ### (set these; some are calculated for you from others)

    M = 22 ## how many antennae in NS direction
    N = 24 ## how many antennae in EW direction
    L1 = 8.5 ## antenna spacing delta in NS direction (in m)
    L2 = 6.3 ## antenna spacing delta in EW direction (in m)
    chord_lat = 49.3 ## chord zenith declination (deg)
    ant_diam = 6.0 ## antenna diameter (in m)
    dphi = 0.3 ## degrees of RA per time step
    dtau = dphi*np.pi/180/omega ## integration time (in s; calculated for you if you set dphi)
    centre_phi_RA_deg = 48 ## central RA, in deg, for the set of integrations
    N_times = 1 ## number of integrations to do
    initial_phi_offset = (N_times-1)/2*dphi ## Calculated for you. Don't worry about this

    survey_dec = 26 ## Sets CHORD survey declination (degrees)
    nu1 = 1425e6 ## first channel frequency (Hz)
    nu2 = 1500e6  ## last channel frequency (Hz)
    nchannels = 16 ## number of channels
    dnu = (nu2-nu1)/nchannels ## (computed for you)
    eta = 1 ## antenna power collection efficiency
    SEFD = 6000 ## per antenna system equivalent flux density (in Jy)

    frequencies = np.linspace(nu1,nu2,nchannels)
    wavelengths = 3e8/frequencies

    ### WHAT TIMES (in days) TO EVALUATE THE MAP AT? ###

    Times = np.arange(3500,7000,1000) ## must be called 'Times'

    ### WHAT SOURCE FIELDS TO INCLUDE?

    include_bright_background = True ## include the FIRST-derived, 'bright' background (>1mJy)
    include_faint_background = True ## include the randomly generated 'faint' background (10uJy - 1mJy)
    apply_background_variation = True ## apply realistic brightness variation statistics
    include_transients = False ## include the transient library. This is specifically a functionality
                                # for my research (Josh Goodeve)

    ### SOURCE VARIABILITY AMPLITUDES AND PROBABILITIES (These will not be used if backgrounds are not set to have variation) ###

    varbins = np.array([0,0.02,0.1]) ## fractional brightness fluctuation standard deviation. 0.1 Corresponds to a 10% RMS flux density fluctuation
    varprobs = np.array([318/370,40/370,12/370]) ## correspond probabilities. Must sum to 1

    ### THIS IS WHERE YOU WOULD APPEND YOUR OWN SOURCES ###

    ### now grab the data for the transients:

    if include_transients:

        transient_data = np.load('JettedTDE_2400.npz')
        transient_spectra = transient_data['data']/1000 ## divide by 1000 to get Jy from mJy
        transient_ra = transient_data['ra']
        transient_dec = transient_data['dec']

        new_source_us = rev.u_vec(transient_ra,transient_dec)
        condition = np.abs(transient_dec - imcenter[1]) < 10
        new_source_us = new_source_us[condition]

    ### CHORD SETUP ### (this plugs in the information you entered above to Hans' classes)

    chord_thetas = np.asarray([np.deg2rad(90-survey_dec)], dtype=ctypes.c_float)
    cp = chordParams(thetas = unpackArraytoStruct(chord_thetas),
                    centre_phi = np.deg2rad(centre_phi_RA_deg),
                    initial_phi_offset = np.deg2rad(initial_phi_offset),
                     m1=M, m2=N, L1=L1, L2=L2, CHORD_zenith_dec = chord_lat, D = ant_diam,
                    delta_tau = dtau, time_samples=N_times)

    ### NOW, THE AUTOMATED STUFF. YOU SHOULDN'T NEED TO TOUCH THINGS BELOW HERE ##########################################################
    source_us_orig = np.zeros((1,3))
    source_us_orig[0][0] = 1 ## so that this is actually a unit vector
    spectra_orig = np.zeros((1,len(frequencies)))

    if include_bright_background:

        ## leave this block alone, it reads in the sky background file ##
        background_data = np.load('SKYMODEL_RA_dec_F_sidx.npz')
        RA = background_data['RA']
        Dec = background_data['dec']
        FJy = background_data['FmJy']/1000
        spec_idx = background_data['spec_idx']
        bright_source_us,F,s = rev.return_close_sources(centre_phi_RA_deg,survey_dec,initial_phi_offset,N_times,dphi,RA,Dec,FJy,spec_idx)
        source_us_orig = np.concatenate((source_us_orig,bright_source_us),axis = 0)
        spectra_orig = np.concatenate((spectra_orig,rev.get_spectra(frequencies,F,s)),axis = 0) ## generates spectra for all background objects

    if include_faint_background:

        ### generate the faint background sources:
        phi1 = centre_phi_RA_deg - 0.5*N_times*dphi - 2/np.cos(np.deg2rad(imcenter[1]))
        phi2 = centre_phi_RA_deg + 0.5*N_times*dphi + 2/np.cos(np.deg2rad(imcenter[1]))

        if np.abs(phi2-phi1) > 360:
            phi2 = 360
            phi1 = 0
        faint_u, faint_spectra = rev.gen_faint_background(120,-1.85,0.1,1,
        phi1,
        phi2,
        imcenter[1]-2,
        np.min((imcenter[1]+2,90)),frequencies)
        source_us_orig = np.concatenate((source_us_orig,faint_u),axis = 0)
        spectra_orig = np.concatenate((spectra_orig,faint_spectra),axis = 0) ## generates spectra for all background objects

    if apply_background_variation:

        ### import the variation library:

        varlib = np.load('variation_library_300.npz')['varlib']

        ### assign each background source a variability time series and fractional variance:

        varindices, varfractions, varstarts = rev.assign_variability(source_us_orig,varbins,varprobs)

    MAP = np.zeros((imsize,imsize,len(Times),len(frequencies)))
    NOISE = MAP.copy()

    for j in range(len(Times)):

        t0 = time.time()

        T = Times[j]
        print('loop: %d/%d' %(j,len(Times)))

        if apply_background_variation:

            ## apply the variance to background sources
            spectra_var = rev.apply_variability(spectra_orig,T,varlib,varindices,varstarts,varfractions)

        else:

            spectra_var = spectra_orig.copy()

        if include_transients:

            new_spectra = transient_spectra[T]
            #new_spectra = np.array([[new_spectra[i,0]] for i in range(len(new_spectra))]) <- switches to channel 0

            condition = np.abs(transient_dec - imcenter[1]) < 10

            new_spectra = new_spectra[condition]

            source_us = np.concatenate((source_us_orig,new_source_us),axis = 0)
            spectra = np.concatenate((spectra_var,new_spectra),axis = 0)

        else:

            source_us = source_us_orig
            spectra = spectra_var

        print('Sim will include %d sources' %(len(spectra)))

        t1 = time.time()

        print("generating sources took", t1-t0, " seconds")

        ### RUN THE CODE ###

        dirtymap = dirtymap_simulator_wrapper (u.astype(ctypes.c_float), wavelengths.astype(ctypes.c_float), source_us, spectra, 1e-9, cp)
        dirtymap = dirtymap.reshape(imsize,imsize,len(frequencies))

        dirtymap /= M**2
        dirtymap /= N**2 ## for normalization. THIS IS IMPORTANT TO GET THE MAP BACK IN JY!

        t2 = time.time()

        print("Dirtymap simulator took", t2-t1, " seconds")

        ### MAKE CORRESPONDING NOISE ###

        noise = dirtymap.copy()
        for i in range(len(frequencies)):
            noise[:,:,i] = rev.make_some_noise(M,N,L1,L2,chord_lat,survey_dec,N_times,dnu,dtau,SEFD,eta,frequencies[i],imsize,cellsize,ant_diam,applybeam = False)
        t3 = time.time()

        print("Generating noise took", t3-t2, " seconds")

        ### RECOVER THE BEAM ###

        if j == 0: ## because this only has to be done once

            A_beam,B_beam = rev.recover_net_beam(u, centre_phi_RA_deg, initial_phi_offset, dphi, N_times, frequencies, survey_dec, antenna_diam = ant_diam)

        ### apply the normalized raw beam to my own noise:

        for i in range(len(frequencies)):

            noise[:,:,i] *= B_beam[:,:,i] ## maximum will be divided out later!

        ## normalize the beam and dirtymap ##:
        for i in range(len(frequencies)):
            maxx = np.max(A_beam[:,:,i])
            A_beam[:,:,i] /= maxx
            dirtymap[:,:,i] /= maxx
            noise[:,:,i] /= maxx
            (noise[:,:,i])[A_beam[:,:,i]<0.25] = np.nan
            (dirtymap[:,:,i])[A_beam[:,:,i]<0.25] = np.nan

        t4 = time.time()
        print("Generating the beam took", t4-t3, " seconds")

        MAP[:,:,j,:] = dirtymap
        NOISE[:,:,j,:] = noise

    if not os.path.exists(f"output/{foldername}"):
        os.makedirs(f"output/{foldername}")
    rev.writetofits_4D(f'output/{foldername}/{run_name}',w,MAP,NOISE,A_beam)

    end_time = time.time()

    seconds = end_time - start_time

    hours = 0
    minutes = 0

    while seconds >= 60:

        minutes += 1
        seconds -= 60

    while minutes >= 60:

        hours += 1
        minutes -= 60

    print('Total time: %dh%dm%ds' %(hours,minutes,seconds))
