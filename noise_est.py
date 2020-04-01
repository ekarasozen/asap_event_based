import sys
sys.path.append("/Users/ezgikarasozen/Documents/Research/Array_processing/asap_w_obspy_git/")
from getdata import event
from getdata import taup
from getdata import prep
import parameters
import wavelet

import matplotlib.pyplot as plt
import numpy as np
import mlpy.wavelet as wave


event_list = parameters.event_list



event_id = ['https://earthquake.usgs.gov/fdsnws/event/1/query?eventid=' + s for s in event_list]
st = event(ev_id=event_id[0], network_list=['IM','XM'], station_code="BC", pick="P", channel="SHZ", start_time=20, end_time=60)
st = prep(st,filter_type="bandpass", freqmin=1, freqmax=3)
nsum2 = np.empty((0, 100))
x=st[0].data
Fs=st[0].stats.sampling_rate #sampling rate
Is=60 # noise segment length
block_pts= 3*Fs #number of points in each window
shift_per = 0.25
overlap = (1-shift_per)/block_pts
offset = block_pts-int(overlap)
hanwin = np.hanning(block_pts)
max_m = ((Is*Fs-block_pts)/offset)+1
print(max_m)
for m in range(1, int(max_m)):
    ibegin = int((m-1)*offset)
    iend = int((m-1)*offset+block_pts)
    a = np.matrix(x[ibegin:iend])
    at = a.getH()
    nwin = np.multiply(at,hanwin)   
    print(nwin.real) 
    dt_n = 0.05
    dj_n = 0.05
    L = len(nwin)
    scales_n = wave.autoscales(L, dt=dt_n, dj=dj_n, wf='morlet', p=6)
    #N = wave.cwt(x=nwin, dt=dt_n, scales=scales_n, wf='morlet', p=6)
    #nsum2= (np.abs(N) **2)
#print(nsum2)

    
#    nwin = np.transpose(x[ibegin:iend])*hanwin
    #nsum2 = nsum2 + abs().^2;
#nes = nsum2/max_m
