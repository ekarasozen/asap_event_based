import os
import sys
sys.path.append("/Users/ezgikarasozen/Documents/Research/Array_processing/asap/")
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
    if t_type == "mlwt":
        dj = 0.05 #scale spacing
        omega0 = 6
        wf = 'morlet' #type of wavelet for mlpy, is there a way to do this for pycwt too? check
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
        tr_SNR = trd.copy()
        tr_alpha = trd.copy()
        tr_SNR.data = SNR.flatten()
        tr_alpha.data = alpha.flatten()
        metrics = measure.waveform_metrics(tro,trd,trp,picktime)
        hilb_div, max_hilb, mean_hilb = measure.hilb_metrics(trd,trp)
    elif t_type == "pycwt":
        dj = 0.05 #scale spacing
        s0 = 0.096801331 # smallest scale, required for pycwt (maybe not? check this), mlpy automatically calculates this
        omega0 = 6
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
        if ss_type == "simple":
            amp_Xp, SNR, alpha0, alpha,beta = ss.simple_subtraction(amp_Xd,amp_Xn, 2, 2, 1)
        elif ss_type == "over":
            amp_Xp, SNR, alpha0, alpha, beta = ss.over_subtraction(amp_Xd,amp_Xn,2,8,0.2)
        elif ss_type == "simple_over":
            amp_Xp, SNR, alpha0, alpha, beta = ss.smooth_over_subtraction(amp_Xd,amp_Xn,2,5,0.2)
        elif ss_type == "freq_over": 
            amp_Xp, SNR, alpha0, alpha, beta = ss.freq_over_subtraction(amp_Xd,amp_Xn,2,5,0.2)
        elif ss_type == "non_lin":
            amp_Xp, SNR, alpha, rho, phi, beta, gamma, alpha0 = ss.nonlin_subtraction(amp_Xd,amp_Xn, 0.1, 0.5)
        elif ss_type == "simple_non_lin":
            amp_Xp, SNR, alpha, rho, phi, beta, gamma, alpha0 = ss.simple_nonlin_subtraction(amp_Xd,amp_Xn, 1.0, 0.0)
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
        hilb_div, max_hilb, mean_hilb = measure.hilb_metrics(trd,trp)
    elif t_type == "stft":
        #t = np.arange(tro.stats.npts) / tro.stats.sampling_rate
        fs = 1/trd.stats.delta
        nperseg = 64
        fo, to, Xo = signal.stft(tro.data, fs, nperseg=nperseg)
        fd, td, Xd = signal.stft(trd.data, fs, nperseg=nperseg)
        fn, tn, Xn = signal.stft(trn.data, fs, nperseg=nperseg)
        amp_Xd = np.abs(Xd)
        amp_Xn = np.abs(Xn)
        amp_Xp, SNR, alpha, beta = ss.simple_subtraction(amp_Xd,amp_Xn, 2, 4, 0.3)
        phase_Xd = np.angle(Xd)
        Xp = amp_Xp * np.exp(1.j*phase_Xd)
        t_tmp, trp.data = signal.istft(Xp, fs)
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
        hilb_div, max_hilb, mean_hilb = measure.hilb_metrics(trd,trp)
    #np.savetxt(event_list[e] + '_amp_Xp.out', amp_Xp, delimiter=',', newline="\n")  
    amp_Xo = abs(Xo)
    outpath = event_list[e] + '/' + ss_type + '/'
    if not os.path.exists(outpath):
       os.makedirs(outpath)
    fig1 = plt.figure()
    fig2 = plt.figure()
    fig3 = plt.figure()
    fig4 = plt.figure()
    fig5 = plt.figure()
    fig6 = plt.figure()
    fig7 = plt.figure()
    fig1 = myplot.wfs(t, tro, trd, trp, outpath, fig1, event_list[e], figname="wfs") 
    fig2 = myplot.scals(t, tro, Xo, Xd, Xp, freq, outpath, fig2, event_list[e], figname="scals") 
    #fig2 = myplot.stft(td, fd, tro, trd, trp, t_tmp, amp_Xd, amp_Xp, fig2, event_list[e], figname="stft") #COMMENT OUT FOR STFT *********
    if ss_type == "simple":
        fig3 = myplot.subtraction_performance(amp_Xd,amp_Xp,freqs_d,picktime,tro,trd,trp,tr_SNR,tr_alpha,metrics,SNR,alpha,alpha0,beta,ss_type, outpath, fig3, event_list[e], phi="0", figname="subtraction_performance")
        fig4 = myplot.sub_param_one_tf(amp_Xd, amp_Xn, freqs_d, outpath, fig4, event_list[e], timeframe=1490, figname="one_timeframe_alpha_beta")
        fig5 = myplot.processed_signal_tf(amp_Xd, amp_Xn, freqs_d, outpath, fig5, event_list[e], x1=1485, now1=11, step1=1, x3=1390, now2=11, step2=20, figname="cons_timeframe_processed_signal")   
        fig6 = myplot.hilb_plot(t,hilb_div,max_hilb,mean_hilb,outpath,fig6,event_list[e],figname="hilbert_metrics")
    elif ss_type == "over":
        fig3 = myplot.subtraction_performance(amp_Xd,amp_Xp,freqs_d,picktime,tro,trd,trp,tr_SNR,tr_alpha,metrics,SNR,alpha,alpha0,beta,ss_type, outpath, fig3, event_list[e], phi="0", figname="subtraction_performance")
        fig4 = myplot.oversub_param_one_tf(amp_Xd, amp_Xn, freqs_d, ss_type, outpath, fig4, event_list[e], timeframe=1490, figname="one_timeframe_oversub")#for oversub
        fig6 = myplot.hilb_plot(t,hilb_div,max_hilb,mean_hilb,outpath,fig6,event_list[e],figname="hilbert_metrics")
    elif ss_type == "simple_over":
        fig3 = myplot.subtraction_performance(amp_Xd,amp_Xp,freqs_d,picktime,tro,trd,trp,tr_SNR,tr_alpha,metrics,SNR,alpha,alpha0,beta,ss_type, outpath, fig3, event_list[e], phi="0", figname="subtraction_performance")
        fig4 = myplot.oversub_param_one_tf(amp_Xd, amp_Xn, freqs_d, ss_type, outpath, fig4, event_list[e], timeframe=1490, figname="one_timeframe_oversub")#for oversub
        fig6 = myplot.hilb_plot(t,hilb_div,max_hilb,mean_hilb,outpath,fig6,event_list[e],figname="hilbert_metrics")
    elif ss_type == "frequency_over": 
        fig3 = myplot.subtraction_performance(amp_Xd,amp_Xp,freqs_d,picktime,tro,trd,trp,tr_SNR,tr_alpha,metrics,SNR,alpha,alpha0,beta,ss_type, outpath, fig3, event_list[e], phi="0", figname="subtraction_performance")
       # fig4 = myplot.oversub_param_one_tf(amp_Xd, amp_Xn, freqs_d, ss_type, outpath, fig4, event_list[e], timeframe=1490, figname="one_timeframe_oversub")#for oversub
        fig6 = myplot.hilb_plot(t,hilb_div,max_hilb,mean_hilb,outpath,fig6,event_list[e],figname="hilbert_metrics")
    elif ss_type == "non_lin":
        fig3 = myplot.subtraction_performance(amp_Xd,amp_Xp,freqs_d,picktime,tro,trd,trp,tr_SNR,tr_alpha,metrics,SNR,alpha,gamma,beta,ss_type, outpath,fig3, event_list[e], phi, figname="subtraction_performance") #instead of alpha0, gamma is used here. 
        fig4 = myplot.nonlin_param_one_tf(amp_Xd, amp_Xn, freqs_d, gamma, beta, outpath, fig4, event_list[e], timeframe=1490, figname="one_timeframe_nonlin") #NON LIN SS
        fig5 = myplot.nonlin_signal_smooth(amp_Xd, amp_Xn, freqs_d, outpath, fig5, event_list[e], timeframe=1490, figname="signal_smooth_nonlin") #NON LIN SS
        fig6 = myplot.hilb_plot(t,hilb_div,max_hilb,mean_hilb, outpath,fig6,event_list[e],figname="hilbert_metrics")
    elif ss_type == "simple_non_lin":
        fig3 = myplot.subtraction_performance(amp_Xd,amp_Xp,freqs_d,picktime,tro,trd,trp,tr_SNR,tr_alpha,metrics,SNR,alpha,alpha0,beta,ss_type, outpath,fig3, event_list[e], phi, figname="subtraction_performance")
        fig4 = myplot.simple_nonlin_param_one_tf(amp_Xd, amp_Xn, freqs_d, gamma, beta, outpath, fig4, event_list[e], timeframe=1490, figname="one_timeframe_simple_nonlin")
        #fig5 = myplot.nonlin_signal_smooth(amp_Xd, amp_Xn, freqs_d, outpath, fig5, event_list[e], timeframe=1490, figname="signal_smooth_nonlin") # NOT UTILIZED FOR SIMPLE NON LIN YET
        fig6 = myplot.hilb_plot(t,hilb_div,max_hilb,mean_hilb, outpath,fig6,event_list[e],figname="hilbert_metrics")
    #fig9 = myplot.alpha_comp_wfs(t, tro, trd, amp_Xd, amp_Xn, phase_Xd, scales_d, omega0, dj, outpath, fig9, event_list[e], figname="alpha_comparison_wfs")
    #fig10 = myplot.alpha_comp_scals(t, tro, trd, amp_Xo, amp_Xd, amp_Xn, phase_Xd, scales_d, freqs_d, omega0, dj, outpath, fig10, event_list[e], figname="alpha_comparison_scals")
    fig7 = myplot.spectra(amp_Xo, amp_Xd, amp_Xn, amp_Xp, freqs_d, outpath, fig7, event_list[e], figname="spectra_comparison")
