import numpy as np
def whitenoise(st,t,type,amplitude,min_freq=1,max_freq=3):
    #np.random.seed(42)
    st_temp = st.copy() #[o]riginal signal
    st_temp.trim(t - 1, t + 5)
    nl=amplitude*st_temp.std()
    print(st.std())
    print(st_temp.std())
    noe=st.stats.npts 
    nos=st.stats.sampling_rate 
    n = st.data.size 
    timestep = 1/(nos) 
    if type==1: #white noise, all stations
        wn = np.random.normal(0,nl,noe) 
        st.data = st.data + wn 
    if type==2: #band-limited white noise
        f = np.fft.fft(st.data) 
        freqs = np.fft.fftfreq(n, d=timestep) 
        idx = np.where(np.logical_and(freqs>=min_freq, freqs<=max_freq))[0]
        wn = np.random.normal(0,nl,len(idx))
        f[idx] += wn
        finv = np.fft.ifft(f).real
        st.data = finv
    return st.data
