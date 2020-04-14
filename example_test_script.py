import sys
sys.path.append("/Users/ezgikarasozen/Documents/Research/Array_processing/asap_w_obspy_git/")
from getdata import *
from addnoise import whitenoise
import param
import spectral_subtraction as ss
import mlwt
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
    scales = mlwt.scales(tro)
    t, freq = mlwt.param(tro,scales)
    Xo = mlwt.cwt(tro,scales)
    amp_Xo = abs(Xo)
    IXo = mlwt.icwt(Xo, tro)
    fig1 = plt.figure()
    fig1 = plot.all(t, tro, Xo, freq, IXo, fig1, event_list[e], figname="original")
    trd = st[0].copy() #[d]egraded version of a signal (noisy real world data, or has garbage added)
    trd.data = whitenoise(trd,type=noise_type,amplitude=noise_amplitude,min_freq=noise_freqmin,max_freq=noise_freqmax)
    scales_d = mlwt.scales(trd)
    t_d, freq_d = mlwt.param(trd,scales_d)
    Xd = mlwt.cwt(trd,scales_d)
    amp_Xd = abs(Xd)
    IXd = mlwt.icwt(Xd, trd)
    fig2 = plt.figure()
    fig2 = plot.all(t_d, trd, Xd, freq_d, IXd, fig2, event_list[e], figname="degraded")
    trn = st[0].copy() #[n]oise sample with no signal (typically used to determine what to remove)
    t0 = trn.stats.starttime
    trn.trim(t0, t0+60) #this option might be better
    #trn.detrend("linear")
    #trn.detrend("demean")
    #trn = trn.data[ibegin:iend] 
    scales_n = mlwt.scales(trd)
    t_n, freq_n = mlwt.param(trn,scales_n)
    Xn = mlwt.cwt(trn,scales_n)
    Xna = np.mean(Xn,axis=1)
    trp = trd.copy() # [p]rocessed version of degraded signal to remove the noise
    amp_Xp = ss.simple_subtraction(Xd,Xna,2)
    phase_Xd = np.exp(1.j*(np.angle(Xd))) 
    scales_p = mlwt.scales(trd)
    t_p, freq_p = mlwt.param(trd,scales_p)
    Xp = amp_Xp*phase_Xd
    IXp = mlwt.icwt((Xp), trd) 
    amp_Xn = abs(Xn) 
    fig3 = plt.figure()
    fig3 = plot.all(t_d, trp, Xp, freq_d, IXp, fig3, event_list[e], figname="processed")
    fig4 = plt.figure()
    fig4 = plot.spectra(amp_Xo, amp_Xd, amp_Xn, amp_Xp, freq_d, fig4, event_list[e], figname="spectra_comparison")
    fig5 = plt.figure()
    fig5 = plot.scales_freq(freq_d, scales_d, fig5, event_list[e], figname="frequency_scale")