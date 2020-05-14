import sys
sys.path.append("/Users/ezgikarasozen/Documents/Research/Array_processing/asap/")
from getdata import *
from addnoise import whitenoise
import spectral_subtraction as ss
#import mlwt ###mlpy
import pycwt as wavelet  ###pycwt
import myplot
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from obspy import Stream
from obspy.imaging.cm import obspy_sequential
import hilbert 
import measure

filename = input("Parameters file: ")
exec(open(filename).read())

event_id = ['https://earthquake.usgs.gov/fdsnws/event/1/query?eventid=' + s for s in event_list]
for e, lab in enumerate(event_id):
    st, picktime = event(ev_id=event_id[e], network_list=network_list, station_code=station_code, pick=pick_type, channel=channel, start_time=start_time, end_time=end_time)
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
        t_d, freqs_d = mlwt.param(trd,scales_d)
        Xd = mlwt.cwt(trd,scales_d)
        amp_Xd = abs(Xd)
        IXd = mlwt.icwt(Xd, trd)
        scales_n = mlwt.scales(trd)
        t_n, freq_n = mlwt.param(trn,scales_n)
        Xn = mlwt.cwt(trn,scales_n)
        amp_Xn = abs(Xn) 
        amp_Xp, SNR, alpha, beta = ss.simple_subtraction(amp_Xd,amp_Xn, 2, 1, 1)
        amp_Xp1, SNR1, alpha1, beta = ss.simple_subtraction(amp_Xd,amp_Xn, 2, 2, 1)
        amp_Xp2, SNR2, alpha2, beta = ss.simple_subtraction(amp_Xd,amp_Xn, 2, 3, 1)
        phase_Xd = np.angle(Xd) 
        Xp = amp_Xp*(np.exp(1.j*phase_Xd))
        trp.data = mlwt.icwt(Xp, trd)
        test1, test2 = hilbert.hilbert_diff(trd, trp)
    elif cwt_type == "pycwt":
        t = np.arange(tro.stats.npts) / tro.stats.sampling_rate
        dt = tro.stats.delta
        J = (np.log2(len(tro) * dt / s0)) / dj  # number of scales-1 
        mother = wavelet.Morlet(omega0)              # See https://github.com/regeirk/pycwt/blob/master/pycwt/wavelet.py 
        Xo, scales, freq, coi, fft, fftfreqs = wavelet.cwt(tro.data, dt, dj, s0, J, mother)    
        IXo = wavelet.icwt(Xo, scales, dt, dj=0.05, wavelet='morlet') 
        t_d = np.arange(trd.stats.npts) / trd.stats.sampling_rate 
        Xd, scales_d, freqs_d, coi_d, fft_d, fftfreqs_d = wavelet.cwt(trd.data, dt, dj, s0, J, mother)    
        amp_Xd = abs(Xd)
        IXd = wavelet.icwt(Xd, scales_d, dt, dj=0.05, wavelet='morlet')
        Xn, scales_n, freq_n, coi_n, fft_n, fftfreqs_n = wavelet.cwt(trn.data, dt, dj, s0, J, mother)
        amp_Xn = abs(Xn)
        amp_Xna = np.mean(amp_Xn,axis=1)
        amp_Xp, SNR, alpha, beta = ss.simple_subtraction(amp_Xd,amp_Xn, 2, 1, 1)
        #amp_Xp, SNR, alpha, beta = ss.over_subtraction(amp_Xd,amp_Xn,2)
        #amp_Xp, alpha, rho, phi, beta, gamma = ss.nonlin_subtraction(amp_Xd,amp_Xn)
        #amp_Xp, SNR, alpha, beta, delta = ss.mulban_subtraction(amp_Xd,amp_Xn,trd,freqs_d)
        phase_Xd = np.angle(Xd)
        Xp = amp_Xp*(np.exp(1.j*phase_Xd))
        trp.data = wavelet.icwt(Xp, scales_d, dt, dj=0.05, wavelet='morlet')
        print('----> maximum imaginary value in "processed signal" is: ', np.max(np.imag(trp.data)))
        if np.max(np.imag(trp.data)) == 0:
            trp.data = np.real(trp.data)
        else:
            print('maximum imaginary value in "processed signal" is not zero, therefore not outputted')
            continue #this condition is not tested yet
        tr_SNR = trd.copy()
        tr_alpha = trd.copy()
        tr_SNR.data = SNR.flatten()
        tr_alpha.data = alpha.flatten()
        metrics = measure.waveform_metrics(tro,trd,trp,picktime)
        test1, test2 = hilbert.hilbert_diff(trd, trp)
    #np.savetxt(event_list[e] + '_SNR.out', SNR, delimiter=',', newline="\n")  
    #np.savetxt(event_list[e] + '_alpha.out', alpha, delimiter=',', newline="\n")   
    #np.savetxt(event_list[e] + '_phi.out', phi, delimiter=',', newline="\n")  
    #np.savetxt(event_list[e] + '_alpha_ns.out', alpha, delimiter=',', newline="\n")   
    #np.savetxt(event_list[e] + '_rho.out', rho, delimiter=',', newline="\n")  
    #np.savetxt(event_list[e] + '_Xn.out', Xn, delimiter=',', newline="\n")  
    amp_Xo = abs(Xo)
    fig1 = plt.figure()
    fig1 = myplot.wfs(t, tro, trd, trp, fig1, event_list[e], figname="wfs") 
    fig2 = plt.figure()
    fig2 = myplot.scals(t, tro, Xo, Xd, Xp, freq, fig2, event_list[e], figname="scals") 
    #fig1 = myplot.all(t, tro, Xo, freq, IXo, fig1, event_list[e], figname="original")
    fig4 = plt.figure()
    fig4 = myplot.spectra(amp_Xo, amp_Xd, amp_Xn, amp_Xp, freqs_d, fig4, event_list[e], figname="spectra_comparison")
    fig5 = plt.figure()
    fig5 = myplot.scales_freq(freqs_d, scales_d, fig5, event_list[e], figname="frequency_scale") 
    fig6 = plt.figure()
    fig6 = myplot.subtraction_performance(amp_Xd,amp_Xp,freqs_d,picktime,tro,trd,trp,tr_SNR,tr_alpha,metrics,alpha,beta,fig6, event_list[e], figname="subtraction_performance")
    fig7 = plt.figure()
    fig7 = myplot.sub_param_one_tf(amp_Xd, amp_Xna, freqs_d, fig7, event_list[e], timeframe=1490, figname="one_timeframe_alpha_beta")
    fig8 = plt.figure()
    fig8 = myplot.processed_signal_tf(amp_Xd, amp_Xna, freqs_d, fig8, event_list[e], x1=1485, now1=11, step1=1, x3=1390, now2=11, step2=20, figname="cons_timeframe_processed_signal")   
    fig9 = plt.figure()
    fig9 = myplot.alpha_comp_wfs(t, tro, trd, amp_Xd, amp_Xn, phase_Xd, scales_d, omega0, dj, fig9, event_list[e], figname="alpha_comparison_wfs")