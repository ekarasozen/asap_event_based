import numpy as np
def whitenoise(tro,trd,t,type,amplitude,min_freq=1,max_freq=3):
    #np.random.seed(42)
    trg = tro.copy() #garbage signal
    tro_temp = tro.copy() 
    tro_temp.trim(t - 1, t + 5)
    #print(tro.std())
    #print(tro_temp.std())
    if type==1: #white noise, all stations
        noe=tro.stats.npts 
        wn = np.random.normal(0,nl,noe) 
        trd.data = trd.data + wn 
    if type==2: #band-limited white noise
        nl=amplitude*tro.std()
        nos=tro.stats.sampling_rate 
        n = tro.data.size 
        timestep = 1/(nos) 
        f = np.fft.fft(trd.data) 
        freqs = np.fft.fftfreq(n, d=timestep) 
        idx = np.where(np.logical_and(freqs>=min_freq, freqs<=max_freq))[0]
        wn = np.random.normal(0,1,len(idx))
        f[idx] += wn*nl
        finv = np.fft.ifft(f).real
        trd.data = finv
    if type==3: #band-limited noise, Mike's version
        trg.data = np.random.normal(0 ,1 , len(trg.data))
        #trg0 = trg.copy()               # copy of unfiltered garbage (for plots)
        trg.filter('bandpass', freqmin=min_freq, freqmax=max_freq, corners=5, zerophase=True)
#        trg1 = trg.copy()               # copy of filtered garbage (for plots)
        trd.data = trd.data + (trg.data*amplitude*tro_temp.std())  # ---- degraded data -----
    return trd.data
