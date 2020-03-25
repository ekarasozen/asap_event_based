import numpy as np
import mlpy.wavelet as wave
import math
from obspy.imaging.cm import obspy_sequential

# For now, wavelet type, omega0 and dj are set to default
# type = Morlet, omega0=6, d=0.05
# Based on Torrence & Compo '97 and obspy example

def scales(st):
    N = len(st.data)
    dt = st.stats.delta
    dj = 0.05
    omega0 = 6
    scales = wave.autoscales(N, dt=dt, dj=dj, wf='morlet', p=omega0)
    return scales

def param (st,scales):
    omega0 = 6
    t = np.arange(st.stats.npts) / st.stats.sampling_rate
    freq = (omega0 + np.sqrt(2.0 + omega0 ** 2)) / (4 * np.pi * scales[1:])
    return t, freq

def cwt(st,scales):
    dt = st.stats.delta
    dj = 0.05
    signal = st.data
    omega0 = 6
   # scales = scales(st)
    # approximate scales through frequencies
    X = wave.cwt(x=signal, dt=dt, scales=scales, wf='morlet', p=omega0)
#    idx = np.where(np.logical_and(freq>=0.1, freq<=0.4))
#     X[idx] = 0
#     X = (X).real
    return X

def icwt(X, st):
    N = len(st.data)
    dt = st.stats.delta
    dj = 0.05
    omega0 = 6
    scales = wave.autoscales(N, dt=dt, dj=dj, wf='morlet', p=omega0)
    #From Torrence & Compo'97 Table 2
    Cr=0.776 #reconstruction factor 
    Wo=(math.pi)**(-1/4) #Psi0(0)
    #missing part of the eqn 11 is defined here as Sc=Scale
    Sc=(dj*math.sqrt(dt))/(Cr*Wo)
    IX = wave.icwt(X=(X*Sc), dt=dt, scales=scales, wf='morlet', p=omega0)
    #np.savetxt('wavelet_8.out', Y, delimiter=',', newline='n')
    #np.savetxt('inv_wavelet_8.out', IX, delimiter=' ', newline='n')
    return IX