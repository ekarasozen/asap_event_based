import numpy as np
from scipy.signal import butter, lfilter
def addnoise(st,type,amplitude,min_freq=1,max_freq=3):
    np.random.seed(42)
    nl=amplitude*st.std()
    noe=st.stats.npts #number of elements
    nos=st.stats.sampling_rate #sampling rate
    #min_freq = 1 #minimum frequency to add noise
    #max_freq = 3 #maximum frequency to add noise
    n = st.data.size #to define the size of the frequency array 
    timestep = 1/(nos) # to define the time step of the frequency array
    if type==1: #white noise, all stations
        wn = np.random.normal(0,nl,noe) #white noise
        st.data = st.data + wn 
    if type==2: #band-limited white noise
        f = np.fft.fft(st.data) #convert signal to frequency domain:
        #define the frequency array based on the signal size and time step
        #so that minimum - maximum frequency criteria can be applied:
        freqs = np.fft.fftfreq(n, d=timestep) 
        #apply minimum - maximum frequency criteria to the frequency array indexes:
        idx = np.where(np.logical_and(freqs>=min_freq, freqs<=max_freq))[0]
        #define white noise to be applied, size should be same with the index array:
        wn = np.random.normal(0,nl,len(idx))
        #add white noise to the frequency domain where the index criteria satisfy
        #with minimum - maximum frequency criteria, rest of the frequency array should remain same: 
        f[idx] += wn
        #convert frequency array to time series:
        finv = np.fft.ifft(f).real
        st.data = finv
    return st.data
