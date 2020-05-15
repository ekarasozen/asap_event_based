'''
A module for functions that help assess the performance of spectral subtraction
'''

import numpy as np
from obspy.signal.cross_correlation import correlate
from obspy.signal.cross_correlation import xcorr_max
from scipy.signal import hilbert, chirp



def waveform_metrics(tro,trd,trp,picktime):
    '''
    Produce a set of metrics to assess the quality of spectral subtraction. This 
    function returns a single (somewhat bloated) dictionary object containing:
        maxcorr_long    max cross-correlation value in 30s time window
        lag_long        lag time at which this max cc occurs
        maxcorr_short   max cross-correlation value in 6s time window
        lag_short       lag time at which this max cc occurs
        snro_long       SNR of original trace, 30s windows before/after picktime
        snrd_long       SNR of degraded trace
        snrp_long       SNR of processed trace
        snro_short      SNR of original trace, 5s windows before/after picktime
        snrd_short      SNR of degraded trace
        snrp_short      SNR of processed trace

    Note that SNR is defined here as the ratio of the standard deviation
    of the waveform in two different windows. This is different than how SNR 
    is defined in the spectral subtraction routines. We use the simplistic 
    definition here because it is intuitive.
    '''
    
    tro_6  = tro.copy()
    trp_6  = trp.copy()
    tro_30 = tro.copy()
    trp_30 = trp.copy()
    tro_pre = tro.copy()
    tro_post = tro.copy()
    trd_pre = trd.copy()
    trd_post = trd.copy()
    trp_pre = trp.copy()
    trp_post = trp.copy()
    
    # cross-correlation, long window
    tro_30 = tro_30.trim(picktime-2, picktime+28).taper(max_percentage=100, max_length=1)
    trp_30 = trp_30.trim(picktime-2, picktime+28).taper(max_percentage=100, max_length=1)
    cclong = correlate(tro_30, trp_30, 20)
    cclong_shift, cclong_max = xcorr_max(cclong)
    
    # cross-correlation, short window
    tro_6 = tro_6.trim(picktime-1, picktime+5).taper(max_percentage=10, max_length=.5)
    trp_6 = trp_6.trim(picktime-1, picktime+5).taper(max_percentage=10, max_length=.5)
    ccshort = correlate(tro_6, trp_6, 100)
    ccshort_shift, ccshort_max = xcorr_max(ccshort)

    # SNR, long window
    wlen = 30
    tro_pre  = tro_pre.trim(picktime-1-wlen, picktime-1)   
    tro_post = tro_post.trim(picktime+1, picktime+1+wlen)
    trd_pre  = trd_pre.trim(picktime-1-wlen, picktime-1)
    trd_post = trd_post.trim(picktime+1, picktime+1+wlen)
    trp_pre  = trp_pre.trim(picktime-1-wlen, picktime-1)
    trp_post = trp_post.trim(picktime+1, picktime+1+wlen)
    snro_long = tro_post.std() / tro_pre.std()  
    snrd_long = trd_post.std() / trd_pre.std() 
    snrp_long = trp_post.std() / trp_pre.std()
    
    # SNR, long window
    wlen = 5   
    tro_pre  = tro_pre.trim(picktime-1-wlen, picktime-1)
    tro_post = tro_post.trim(picktime+1, picktime+1+wlen)
    trd_pre  = trd_pre.trim(picktime-1-wlen, picktime-1)
    trd_post = trd_post.trim(picktime+1, picktime+1+wlen)
    trp_pre  = trp_pre.trim(picktime-1-wlen, picktime-1)
    trp_post = trp_post.trim(picktime+1, picktime+1+wlen)
    snro_short = tro_post.std() / tro_pre.std() 
    snrd_short = trd_post.std() / trd_pre.std() 
    snrp_short = trp_post.std() / trp_pre.std()



    # create output
    metrics = {
        "maxcorr_long":   round(cclong_max,2),
        "lag_long":       cclong_shift*tro.stats.delta,
        "maxcorr_short":  round(ccshort_max,2),
        "lag_short":      ccshort_shift*tro.stats.delta,
        "snro_long":      round(snro_long,1),
        "snrd_long":      round(snrd_long,1),
        "snrp_long":      round(snrp_long,1),
        "snro_short":     round(snro_short,1),
        "snrd_short":     round(snrd_short,1),
        "snrp_short":     round(snrp_short,1)
    }
    
    return metrics
    
def hilb_metrics(trd,trp):
    #This can ben added to metrics if needed. 
    # Hilbert transform difference
    trd_h = hilbert(trd)
    trp_h = hilbert(trp)
    hilb_div = np.angle((trd_h / trp_h), deg=True) 
    max_hilb = np.max(np.abs(hilb_div))
    mean_hilb = np.mean(hilb_div)
    #hilbert_sub = np.unwrap(np.angle(trd_h) - np.angle(trp_h)) #this is same with  np.angle(trd_h / trp_h)

    return hilb_div, max_hilb, mean_hilb
    