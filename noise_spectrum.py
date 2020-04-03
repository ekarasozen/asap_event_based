import sys
sys.path.append("/Users/ezgikarasozen/Documents/Research/Array_processing/asap_w_obspy_git/")
import wavelet
import numpy as np


def time_slice(st,ibegin,iend):
    #nsum2 = np.empty((0, 100))
    #Fs=st[0].stats.sampling_rate #sampling rate
    #Is=60 # noise segment length
    #block_pts= 3*Fs #number of points in each window
    #shift_per = 0.25
    #overlap = (1-shift_per)/block_pts
    #offset = block_pts-int(overlap)
    #hanwin = np.hanning(block_pts)
    #max_m = ((Is*Fs-block_pts)/offset)+1
    a = st.data[ibegin:iend]
    #at = a.getH() #if transpose is needed
    #nwin = np.multiply(at,hanwin)  if hanning window is needed  
    scales = wavelet.scales(st)
    t, freq = wavelet.param(st,scales)
    cwt_n = wavelet.cwt(st,scales)
    N = np.mean(cwt_n,axis=1)
    return(N)
    
    #nsum2 = nsum2 + abs().^2;
    #nes = nsum2/max_m
