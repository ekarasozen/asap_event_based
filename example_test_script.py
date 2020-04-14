import sys
sys.path.append("/Users/ezgikarasozen/Documents/Research/Array_processing/asap_w_obspy_git/")
from getdata import *
from addnoise import whitenoise
import param
import spectral_subtraction as ss
import mlwt ###mlpy
#import pycwt as wavelet  ###pycwt
import plot
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from obspy import Stream
from obspy.imaging.cm import obspy_sequential

event_list = param.event_list
network_list = param.network_list
station_code = param.station_code
pick_type = param.pick_type
channel = param.channel
start_time = param.start_time
end_time = param.end_time
filter_type = param.filter_type
filter_freqmin = param.filter_freqmin
filter_freqmax = param.filter_freqmax
noise_type = param.noise_type
noise_amplitude = param.noise_amplitude
noise_freqmin = param.noise_freqmin
noise_freqmax = param.noise_freqmax
ibegin = param.ibegin
iend = param.iend


event_id = ['https://earthquake.usgs.gov/fdsnws/event/1/query?eventid=' + s for s in event_list]
for e, lab in enumerate(event_id):
    st = event(ev_id=event_id[e], network_list=network_list, station_code=station_code, pick=pick_type, channel=channel, start_time=start_time, end_time=end_time)
    st = prep(st,filter_type=filter_type, freqmin=filter_freqmin, freqmax=filter_freqmax)
    tro = st[0].copy() #[o]riginal signal
    scales = mlwt.scales(tro) ###mlpy
    t, freq = mlwt.param(tro,scales) ###mlpy
    Xo = mlwt.cwt(tro,scales) ###mlpy
    #t = np.arange(tro.stats.npts) / tro.stats.sampling_rate ###pycwt
    #dt = tro.stats.delta ###pycwt
    #s0 = 0.096801331                               # smallest scale ###pycwt
    #dj = 0.05                               # scale spacing ###pycwt
    #J = (np.log2(len(tro) * dt / s0)) / dj  # number of scales-1 ###pycwt
    #mother = wavelet.Morlet(6)              # See https://github.com/regeirk/pycwt/blob/master/pycwt/wavelet.py ###pycwt
    #Xo, scales, freq, coi, fft, fftfreqs = wavelet.cwt(tro.data, dt, dj, s0, J, mother)    ###pycwt
    amp_Xo = abs(Xo)
    IXo = mlwt.icwt(Xo, tro) ###mlpy
    #IXo = wavelet.icwt(Xo, scales, dt, dj=0.05, wavelet='morlet') ###pycwt
    fig1 = plt.figure()
    fig1 = plot.all(t, tro, Xo, freq, IXo, fig1, event_list[e], figname="original")
    trd = st[0].copy() #[d]egraded version of a signal (noisy real world data, or has garbage added)
    trd.data = whitenoise(trd,type=noise_type,amplitude=noise_amplitude,min_freq=noise_freqmin,max_freq=noise_freqmax) ###mlpy
    scales_d = mlwt.scales(trd) ###mlpy
    t_d, freq_d = mlwt.param(trd,scales_d) ###mlpy
    Xd = mlwt.cwt(trd,scales_d) ###mlpy
    #t_d = np.arange(trd.stats.npts) / trd.stats.sampling_rate ###pycwt
    #Xd, scales_d, freq_d, coi_d, fft_d, fftfreqs_d = wavelet.cwt(trd.data, dt, dj, s0, J, mother)    ###pycwt
    amp_Xd = abs(Xd)
    IXd = mlwt.icwt(Xd, trd) ###mlpy
    #IXd = wavelet.icwt(Xd, scales_d, dt, dj=0.05, wavelet='morlet') ###pycwt
    fig2 = plt.figure()
    fig2 = plot.all(t_d, trd, Xd, freq_d, IXd, fig2, event_list[e], figname="degraded")
    trn = trd.copy() #[n]oise sample with no signal (typically used to determine what to remove)
    t0 = trn.stats.starttime
    trn.trim(t0, t0+60) #this option might be better
    #trn.detrend("linear")
    #trn.detrend("demean")
    #trn = trn.data[ibegin:iend] #not going to use this probably, parameters file needs to be changed 
    scales_n = mlwt.scales(trd) ###mlpy
    t_n, freq_n = mlwt.param(trn,scales_n) ###mlpy
    Xn = mlwt.cwt(trn,scales_n) ###mlpy
    #Xn, scales_n, freq_n, coi_n, fft_n, fftfreqs_n = wavelet.cwt(trn.data, dt, dj, s0, J, mother)    ###pycwt
    amp_Xn = abs(Xn) 
    Xna = np.mean(Xn,axis=1)
    trp = trd.copy() # [p]rocessed version of degraded signal to remove the noise
    amp_Xp = ss.simple_subtraction(Xd,Xna,2)
    phase_Xd = np.angle(Xd) 
    Xp = amp_Xp*(np.exp(1.j*phase_Xd))
    IXp = mlwt.icwt(Xp, trd) ###mlpy
    #IXp = wavelet.icwt(Xp, scales_d, dt, dj=0.05, wavelet='morlet') ###pycwt
    fig3 = plt.figure()
    fig3 = plot.all(t_d, trp, Xp, freq_d, IXp, fig3, event_list[e], figname="processed")
    fig4 = plt.figure()
    fig4 = plot.spectra(amp_Xo, amp_Xd, amp_Xn, amp_Xp, freq_d, fig4, event_list[e], figname="spectra_comparison")
    fig5 = plt.figure()
    fig5 = plot.scales_freq(freq_d, scales_d, fig5, event_list[e], figname="frequency_scale") 