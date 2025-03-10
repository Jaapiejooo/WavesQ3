import numpy as np                   #importing numpy package for scientific computing
import matplotlib.pyplot as plt      #importing matplotlib package for plots
from scipy.signal import welch
import matplotlib
from matplotlib import animation
#from IPython.display import display, Image

from scipy.fftpack import fft    # importing Fourier transform package
from scipy.stats import chi2     # importing confidence interval package

#plt.rcParams['figure.figsize'] = (15, 8)   # set the width and height of plots in inches
#plt.rcParams.update({'font.size': 13})     # change this value to your desired font size

def read_point_output(name_table):
    '''reads the wave gauge output type 
            input: name_table is the name of the output table (e.g. name_table="gauge1.tbl" in the original SWASH input file)
            outputs:
                t: vector containing the time in s
                eta: vector containing the surface elevation in m
                ux: vector containing the depth averaged horizontal velocity in m/s
    '''
    
    gauge = np.loadtxt(name_table,skiprows=7)

    t   = gauge[:,0] #time vector (s)
    eta = gauge[:,1] #water level (m)
    ux  = gauge[:,2] #depth-averaged velocity in the x-direction (m/s)
    #uy = gauge[:,3] #depth-averaged velocity in the y-direction (0 at all time because we are looking at a 1D case)
    
    return t, eta, ux #, uy

def read_grid_output(name_table,nx):
    ''' time, eta, ux, uy = read_singlepointoutput(name_table)
    read grid output
    inputs: name_table : name of the table which needs to be read (e.g., grid_output.tbl)
          nx: number of cells according to the file 
    outputs:
          time: time vector (s) of length nt  
          x: cross-shore location (m) of length nx 
          eta: array of size nx*nt containing the surface elevation (m)
                                  with respect to still water level 
          zbed: bed level (m) of length nx
          ux: array of size nx*nt containing the depth averaged horizontal velocity (m)
          h: array of size nx*nt containing the instantaneous water depth (m)'''

    
    grp = np.loadtxt(name_table,skiprows=0) #load file
    
    ns,no=grp.shape  #check dimensions of the file
                     
    nt = round(ns/nx)            # number of time steps saved in the output 
                                 # (note that the sampling interval for the output file may be different from the dt used in the calculations)
    A = np.reshape(grp,(nx,nt,no),order="F")  #we reshape the array grp as a 3D matrix of dimension nx, nt, no, where no is the number of outputed variables
    
    time =  A[0,:,0]  
    x    =  A[:,0,1]  
    eta  =  A[:,:,2] 
    zbed = -A[:,0,3]
    ux   =  A[:,:,4]
    h    =  A[:,:,6]


    return time, x, eta, zbed, ux, h

def frequency_filter(data, Fs, f_low, f_high):
    ''' frequency_filter is a simple spectral filter in which the unwanted frequencies (below f_low and above f_high) 
        are set to zero before coming back to the time-domain
            input: data timeseries you want to filter
                   F_s the sampling frequency of this timeseries (Hz)
                   f_low and f_high are the limits of the band pass filter (Hz)
            output: data_filtered band pass filtered timeseries (same unit as the input timeseries)
    '''
    
    N = len(data)

    fft_data = np.fft.fft(data)  # fourier transform of the signal

    freq_vector = np.fft.fftfreq(N, d=1/Fs) # corresponding (2-sided) frequency axis (includes positive and negative values)
    
    idx = np.where((abs(freq_vector) > f_high) | (abs(freq_vector) <= f_low)) # we select the indices to filter out
    
    fft_data[idx]=0.  # we set the the fourier coefficients corresponding to abs(f)>f_high and abs(f)<f_low to zero

    data_filtered = np.fft.ifft(fft_data).real  # we come back to the time domain with an inverse Fourier transform
    
    return data_filtered

def Analysis_all_locations(xbed,zbed,time,eta,nBlocks=20):    
    # Exercise 2 - Cross shore evolution of H1/3
    # This function calculates the significant wave height (H13_tot) at each of the 11
    # locations of the SandyDuck dataset 

    # ---------------------------
    #       Initialisation
    # ---------------------------

    # load time-series of surface elevation (with respect to mean water level)
    data = eta
    dt = time[1]-time[0]
    Fs = 1/dt                            # sampling frequency [Hz]

    # load bathymetry

    # other useful variables
    zeta_tide = .5          # tidal level in meters when the data was collected
    positions = np.arange(0,len(eta))
    #positions = np.array([49,167,239,264,289,310,327,340,364,388,405]) # cross-shore position of the 11 pressure sensors 
                                                                       # positions[0] is the location of P1, positions[1] 
                                                                       #corresponds to P2, etc.

    # initialisation of the arrays which will contain the significant wave
    # heights and rms wave heights at each location (now filled with zeros, but will be modified within the loop)
    H13_tot = np.zeros((len(positions),1))
    H_mean = np.zeros((len(positions),1)) 
    Hrms_tot = np.zeros((len(positions),1))
    T13_tot = np.zeros((len(positions),1))
    T_mean = np.zeros((len(positions),1))
    Hi = []
    Ti = []
    E_all = []
    f_all =[]  

    # ---------------------------
    #        Calculations
    # ---------------------------

    for i in range(len(positions)): # loop on the index of the sensors 
                        # (i=0 corresponds to the most offshore sensor P1, i=10 to the most onshore sensor, P11)

        # we define eta_i as the i^th column of the array data = surface elevation at sensor P{i+1}
        # trim data
        t_start = 30
        idx_start = int(t_start/dt)       
        eta_i = data[i,idx_start:]

        #eta_i = frequency_filter(eta_i, Fs, f_low=0.03, f_high=2)
        E, f, _, _ = wave_spectrum(eta_i,nBlocks,Fs)
        E_all.append(E)
        f_all.append(f)

        # and then conduct a wave-by-wave analysis
        #print(f'loop {i}') 
        Hind,Tind = zero_crossing(eta_i,Fs)
        Hi.append(Hind)
        Ti.append(Tind)

        # we calculate the significant wave height and store it in the i^th element of the vector H13_tot
        try:
            H13_tot[i] = significant_wave_height(Hind)
            H_mean[i] = Hind.mean()
            T13_tot[i] = period_13(Hind,Tind)
            T_mean[i] = Tind.mean()
        except: 
            H13_tot[i] = np.nan
            H_mean[i] = np.nan
            T13_tot[i] = np.nan
            T_mean[i] = np.nan

        # we do the same for the root mean square wave height
        #Hrms_tot[i] = rms_wave_height(Hind)


    return H13_tot, T13_tot, H_mean, T_mean, Hi, Ti, np.array(E_all), np.array(f_all)

def zero_crossing(data,Fs):
    # performs zero crossing analysis of wave data
    #
    # inputs   data: detrended time-series of water elevation in m. 
    #          Fs: sampling frequency of data in Hz
    #
    # outputs  H_ind: individual wave heights in m (array)
    #          T_ind: individual wave periods in s (array)
    
    # 1. Preliminary calculations
    # ---------------------------
    
    # time vector for the water elevation [s]
    time = np.linspace(0,(len(data)-1)/Fs,len(data))
    # Before performing the analysis, we remove the zero values
    d0 = data[data != 0]
    t0 = time[data != 0]
    
    # 2. Zero-crossing analysis 
    # -------------------------

    # We identify the indices at which surface elevation changes sign
    crossing1 = np.squeeze(np.where(d0[0:-1]*d0[1:] < 0))
    # If the elevation is negative at t=0, the first crossing
    # is a zero-upward-crossing -> it is rejected
    try:
        if d0[0] < 0.0:
            crossing2 = np.delete(crossing1,0)
            crossing = crossing2[0::2]
        else:
            crossing = crossing1[0::2]  # these are the zero-down-crossing

        # 3. Calculation individual wave characteristics
        # ----------------------------------------------
        elevation_crest = np.zeros(len(crossing)-1)
        elevation_trough = np.zeros(len(crossing)-1)
   
        if len(crossing)>=2: # Calculate wave period and height if at least one wave has been identified (=2 crossings)
            for i in range(len(crossing)-1):
                elevation_crest[i] = np.max(d0[crossing[i]:crossing[i+1]]) # crest elevation
                elevation_trough[i] = np.min(d0[crossing[i]:crossing[i+1]]) # trough elevation
            T_ind = np.diff(t0[crossing]) #period = time difference between two successive down-zero-crossing
            H_ind = elevation_crest - elevation_trough # individual wave heigh
            if np.shape(T_ind) != np.shape(H_ind):
                T_ind=[]
                H_ind=[]
        else: # if no waves are detected, returns empty vectors
            T_ind=[]
            H_ind=[]

    except:
        T_ind=[]
        H_ind=[]

    return H_ind, T_ind

def rms_wave_height(H_ind):
    # Hrms = rms_wave_height(Hind)
    # input    H_ind array containing the individual wave heights in m 
    # output   Hrms root mean square wave height in m.
    Hrms = np.sqrt(np.sum(H_ind**2)/len(H_ind)) # calculation of Hrms (NB: sqrt is a numpy built-in function that calculates the root square)
    return Hrms

def significant_wave_height(H_ind):
    # Compute the signicant wave height 
    # input: H_ind is vector containing the individual wave heights of a given record (m) 
    # output: H13 significant wave height (m)
    
    
    # 1. sort the wave heights from highest to lowest
    H_sort = sorted(H_ind, reverse=True)
    
    # 2. define a new vector H_1to13 containing the highest third of the waves of the record
    # (=first n13 elements of the sorted vector)
    n_waves = len(H_ind)             # number of waves in the signal
    n13 = int(np.round(n_waves/3.0)) # the round function insures that n13 is an integer
    # --- ADD A LINE OF CODE HERE ---
    H_1to13 = H_sort[:n13]
    
    # 3. calculate H13 (output of the function)
    # --- ADD A LINE OF CODE HERE ---
    H13 = np.sum(H_1to13)/n13 
    
    return H13

def period_13(H_ind,T_ind):
    # Compute the signicant wave period 
    # input: H_ind, T_ind are vectors containing the individual wave heights and periods of a given record (m,s) 
    # output: T13 significant wave period (s)
    
    
    # 1. sort the wave heights from highest to lowest
    sorted_indexes = np.argsort(H_ind)[::-1]
    H_sort = H_ind[sorted_indexes]
    T_sort = T_ind[sorted_indexes]
    
    
    # 2. define a new vector T_1to13 containing the period corresponding to the third highest waves of the record
    # (=first n13 elements of the sorted vector)
    n_waves = len(H_ind)             # number of waves in the signal
    n13 = int(np.round(n_waves/3.0)) # the round function insures that n13 is an integer
    # --- ADD A LINE OF CODE HERE ---
    T_1to13 = T_sort[:n13]
    
    
    # 3. calculate T13 (output of the function)
    # --- ADD A LINE OF CODE HERE ---
    T13 = np.sum(T_1to13)/n13
    
    return T13

def wave_spectrum(data,nBlocks,Fs):
    ''' Compute variance density spectrum from a given time-series and its 
    90% confidence intervals. 
    The time-series is first divided into nBlocks blocks (of length nfft = [total length]/nBlocks) before being 
    Fourier-transformed 
    Note that this is one of the simplest ways to estimate the variance density spectrum 
    (no overlap between the blocks, and use of a rectangular window) - see for instance 
    scipy.signal.welch for more advanced spectral calculations.

    INPUT
      data    timeseries 
      nBlocks  number of blocks
      Fs     sampling frequency of the timeseries (Hz)
    
    OUTPUT
      E       (one-sided) variance spectral density. If data is in meters, E is in m^2/Hz
      f       frequency axis (Hz)
      confLow and confUpper     Lower and upper 90% confidence interval; 
                                (Multiplication factors for E)  '''
    # 1. PRELIMINARY CALCULATIONS
    # ---------------------------
    n = len(data)                # length of the time-series
    nfft = int(n/nBlocks)
    nfft = int(nfft - (nfft%2))
    data_new = data[0:nBlocks*nfft] # (we work only with the blocks which are complete)

    # we organize the initial time-series into blocks of length nfft 
    dataBlock = np.reshape(data_new,(nBlocks,nfft))  # each column of dataBlock is one block
    
    # 2. CALCULATION VARIANCE DENSITY SPECTRUM
    # ----------------------------------------

    # definition frequency axis
    df = Fs/nfft      # frequency resolution of the spectrum df = 1/[Duration of one block]
    f = np.arange(0,Fs/2+df,df)   # frequency axis (Fs/2 = Fnyquist = max frequency)
    fId = np.arange(0,len(f))

    # Calculate the variance for each block and for each frequency
    fft_data = fft(dataBlock,n = nfft,axis = 1)/nfft    # Fourier transform of the data
    fft_data = 2*fft_data[:,fId]                        # Only one side needed
     
    E = np.abs(fft_data)**2/2                  # E(i,b) = ai^2/2 = variance at frequency fi for block b 
    # We finally average the variance over the blocks, and divide by df to get the variance DENSITY spectrum
    E = np.mean(E, axis = 0)/df
    
    # 3. CONFIDENCE INTERVALS
    # -----------------------
    edf = round(nBlocks*2)   # Degrees of freedom 
    alpha = 0.1              # calculation of the 90% confidence interval

    confLow = edf/chi2.ppf(1-alpha/2,edf)    # see explanations on confidence intervals given in lecture 3 
    confUpper  = edf/chi2.ppf(alpha/2,edf)
    
    return E,f,confLow,confUpper
