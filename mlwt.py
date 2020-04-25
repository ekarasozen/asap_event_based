import sys
sys.path.append("/Users/ezgikarasozen/Documents/Research/Array_processing/asap_w_obspy_git/")
#import param
import numpy as np
import mlpy.wavelet as wave
import math
from obspy.imaging.cm import obspy_sequential

#filename = input("Parameters file: ")
#exec(open(filename).read())

# For now, wavelet type, omega0 and dj are set to default
# type = Morlet, omega0=6, d=0.05
# Based on Torrence & Compo '97 and obspy example
#still not sure about renaming st to tr. 

#dj = param.dj
#omega0 = param.omega0
#wf = param.wf

def scales(st):
    L = len(st.data)
    dt = st.stats.delta
    scales = wave.autoscales(L, dt=dt, dj=dj, wf=wf, p=omega0)
    return scales

def param (st,scales):
    t = np.arange(st.stats.npts) / st.stats.sampling_rate
    #from table 1 in Torrence & Compo '97'
    freq = (omega0 + np.sqrt(2.0 + omega0 ** 2)) / (4 * np.pi * scales[0:])
    return t, freq

def cwt(st,scales):
    dt = st.stats.delta
    signal = st.data
    # approximate scales through frequencies
    X = wave.cwt(x=signal, dt=dt, scales=scales, wf=wf, p=omega0)
    return X

def icwt(X, st):
    L = len(st.data)
    dt = st.stats.delta
    scales = wave.autoscales(L, dt=dt, dj=dj, wf=wf, p=omega0)
    #From Torrence & Compo'97 Table 2
    Cr=0.776 #reconstruction factor 
    Wo=(math.pi)**(-1/4) #Psi0(0)
    #missing part of the eqn 11 is defined here as Sc=Scale
    Sc=(dj*math.sqrt(dt))/(Cr*Wo)
    IX = wave.icwt(X=(X*Sc), dt=dt, scales=scales, wf=wf, p=omega0)
    return IX