import sys
sys.path.append("/Users/west/Repositories/asap_w_obspy/")
from getdata import *
from addnoise import whitenoise
import spectral_subtraction as ss
import mlwt ###mlpy
import pycwt as wavelet  ###pycwt
import myplot
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from obspy import Stream
from obspy.imaging.cm import obspy_sequential

filename = input("Parameters file: ")
exec(open(filename).read())

event_id = ['https://earthquake.usgs.gov/fdsnws/event/1/query?eventid=' + s for s in event_list]
for e, lab in enumerate(event_id):
    st = event(ev_id=event_id[e], network_list=network_list, station_code=station_code, pick=pick_type, channel=channel, start_time=start_time, end_time=end_time)
    st = prep(st,filter_type=filter_type, freqmin=filter_freqmin, freqmax=filter_freqmax)
    tro = st[0].copy() #[o]riginal signal
    trd = st[0].copy() #[d]egraded version of a signal (noisy real world data, or has garbage added)
    trd.data = whitenoise(trd,type=noise_type,amplitude=noise_amplitude,min_freq=noise_freqmin,max_freq=noise_freqmax)
    trn = trd.copy() #[n]oise sample with no signal (typically used to determine what to remove)
    t0 = trn.stats.starttime
    trn.trim(t0+ibegin, t0+iend) #put this to parameters
    trn.detrend("linear")
    trn.detrend("demean")
    trp = trd.copy() # [p]rocessed version of degraded signal to remove the noise, seems like there is no need for this
    if cwt_type == "mlwt":
        scales = mlwt.scales(tro)
        t, freq = mlwt.param(tro,scales)
        Xo = mlwt.cwt(tro,scales)
        IXo = mlwt.icwt(Xo, tro)
        scales_d = mlwt.scales(trd)
        t_d, freq_d = mlwt.param(trd,scales_d)
        Xd = mlwt.cwt(trd,scales_d)
        amp_Xd = abs(Xd)
        IXd = mlwt.icwt(Xd, trd)
        scales_n = mlwt.scales(trd)
        t_n, freq_n = mlwt.param(trn,scales_n)
        Xn = mlwt.cwt(trn,scales_n)
        amp_Xn = abs(Xn) 
#        amp_Xp = ss.simple_subtraction(amp_Xd,amp_Xn,2)
        amp_Xp = ss.over_subtraction(amp_Xd,amp_Xn)
        phase_Xd = np.angle(Xd) 
        Xp = amp_Xp*(np.exp(1.j*phase_Xd))
        IXp = mlwt.icwt(Xp, trd)
    elif cwt_type == "pycwt":
        t = np.arange(tro.stats.npts) / tro.stats.sampling_rate
        dt = tro.stats.delta
        J = (np.log2(len(tro) * dt / s0)) / dj  # number of scales-1 
        mother = wavelet.Morlet(omega0)              # See https://github.com/regeirk/pycwt/blob/master/pycwt/wavelet.py 
        Xo, scales, freq, coi, fft, fftfreqs = wavelet.cwt(tro.data, dt, dj, s0, J, mother)    
        IXo = wavelet.icwt(Xo, scales, dt, dj=0.05, wavelet='morlet') 
        t_d = np.arange(trd.stats.npts) / trd.stats.sampling_rate 
        Xd, scales_d, freq_d, coi_d, fft_d, fftfreqs_d = wavelet.cwt(trd.data, dt, dj, s0, J, mother)    
        amp_Xd = abs(Xd)
        IXd = wavelet.icwt(Xd, scales_d, dt, dj=0.05, wavelet='morlet')
        Xn, scales_n, freq_n, coi_n, fft_n, fftfreqs_n = wavelet.cwt(trn.data, dt, dj, s0, J, mother)
        amp_Xn = abs(Xn)
        #amp_Xp = ss.simple_subtraction(amp_Xd,amp_Xn,2)
        #amp_Xp = ss.over_subtraction(amp_Xd,amp_Xn,2)
        amp_Xp = ss.nonlin_subtraction(amp_Xd,amp_Xn,2)
        #amp_Xp = ss.mulban_subtraction(amp_Xd,amp_Xn,trd,freq_d)
        phase_Xd = np.angle(Xd)
        Xp = amp_Xp*(np.exp(1.j*phase_Xd))
        IXp = wavelet.icwt(Xp, scales_d, dt, dj=0.05, wavelet='morlet')
    amp_Xo = abs(Xo)
    fig1 = plt.figure()
    fig1 = myplot.all(t, tro, Xo, freq, IXo, fig1, event_list[e], figname="original")
    fig2 = plt.figure()
    fig2 = myplot.all(t_d, trd, Xd, freq_d, IXd, fig2, event_list[e], figname="degraded")
    fig3 = plt.figure()
    fig3 = myplot.all(t_d, trd, Xp, freq_d, IXp, fig3, event_list[e], figname="processed")
    fig4 = plt.figure()
    fig4 = myplot.spectra(amp_Xo, amp_Xd, amp_Xn, amp_Xp, freq_d, fig4, event_list[e], figname="spectra_comparison")
    fig5 = plt.figure()
    fig5 = myplot.scales_freq(freq_d, scales_d, fig5, event_list[e], figname="frequency_scale") 