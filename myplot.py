import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.offsetbox as offsetbox
from obspy import Stream
from obspy.imaging.cm import obspy_sequential
import os
import sys
sys.path.append("/Users/ezgikarasozen/Documents/Research/Array_processing/asap/")
import spectral_subtraction as ss
import pycwt as wavelet  ###pycwt
import mlwt ###mlpy

#plot all waveforms    
def wfs(t, tro, trd, trp, outpath, fig, event_list, station_list, figname="wfs"): 
    ax11 = fig.add_axes([0.1, 0.75, 0.7, 0.2])
    ax12 = fig.add_axes([0.1, 0.55, 0.7, 0.2], sharex=ax11)
    ax13 = fig.add_axes([0.1, 0.35, 0.7, 0.2], sharex=ax11)
    ax14 = fig.add_axes([0.1, 0.15, 0.7, 0.2], sharex=ax11)
    ax11.plot(t, tro.data, 'k', linewidth=0.3, label='original')
    ax12.plot(t, trd.data, 'k', linewidth=0.3, linestyle='--', label='degraded')
    ax13.plot(t, trp.data, 'b', linewidth=0.3, label='processed')
    ax14.plot(t, tro.data, 'k', linewidth=0.3, label='original')
    ax14.plot(t, trd.data, 'k', linewidth=0.3, linestyle='--', label='degraded')
    ax14.plot(t, trp.data, 'b', linewidth=0.3, label='processed')
    ax11.legend()
    ax12.legend()
    ax13.legend()
    ax14.legend(loc='upper left', fontsize='medium')
    fig.autofmt_xdate()    
    fig.savefig(os.path.join(outpath, event_list + '_'+ station_list + '_' + figname + '.png'))

#plot all scaleograms
def scals(t, tr, Xo, Xd, Xp, freq, outpath, fig, event_list, station_list, figname="scals"): 
    maxamp_o = abs(Xo).max()/2           # these two lines just adjust the color scale
    maxamp_d = abs(Xd).max()/2           # these two lines just adjust the color scale
    maxamp_p = abs(Xp).max()/2           # these two lines just adjust the color scale
    minamp = 0
    tX, f = np.meshgrid(tr.times(), freq)
    ax11 = fig.add_axes([0.1, 0.75, 0.7, 0.2])
    ax111 = fig.add_axes([0.83, 0.75, 0.03, 0.2])
    ax12 = fig.add_axes([0.1, 0.45, 0.7, 0.2])
    ax121 = fig.add_axes([0.83, 0.45, 0.03, 0.2])
    ax13 = fig.add_axes([0.1, 0.15, 0.7, 0.2])
    ax131 = fig.add_axes([0.83, 0.15, 0.03, 0.2])
    img_o = ax11.pcolormesh(tX, f, np.abs(Xo), cmap=obspy_sequential, vmin=minamp, vmax=maxamp_o)
    img_d = ax12.pcolormesh(tX, f, np.abs(Xd), cmap=obspy_sequential, vmin=minamp, vmax=maxamp_d)
    img_p = ax13.pcolormesh(tX, f, np.abs(Xp), cmap=obspy_sequential, vmin=minamp, vmax=maxamp_p)
    ax11.set_title('original CWT amplitude')
    ax12.set_title('degraded CWT amplitude')
    ax12.set_ylabel("frequency [Hz]")
    ax13.set_title('processed CWT amplitude')
    ax13.set_xlim(t[0], t[-1])
    ax13.set_ylim(freq[-1], freq[0])
    ax13.set_xlabel("time after %s [s]" % tr.stats.starttime)
    fig.colorbar(img_o, cax=ax111)
    fig.colorbar(img_d, cax=ax121)
    fig.colorbar(img_p, cax=ax131)
    fig.autofmt_xdate()    
    fig.savefig(os.path.join(outpath, event_list + '_'+ station_list + '_' + figname + '.png'))

def subtraction_performance(amp_Xd,amp_Xp,freqs_d,picktime,tro,trd,trp,tr_SNR,tr_alpha,metrics,SNR,alpha,alpha0,beta,ss_type,outpath,fig,event_list,station_list,phi="0",figname="subtraction_parameters"):
    '''
    Creates a single uber-plot summarizing the performance of spectral subtraction.
    The many input variables for this plotting method should be pretty self-evident. This 
    method needs to be called *after* calculate_metrics.
        freqs_d:  the vector of frequencies output by the CWT process
        tr_SNR:   trace object containing the vector of SNR values
        tr_alpha: trace object containing the vector of alpha values
        metrics:  is an awkward dictionary of values and strings produced 
                  by the measure.waveform_metrics function
    '''
    
    alpha_beta_text = ' alpha0: ' + str(alpha0) + '    beta: ' + str(beta) 
    #alpha0 is gamma for simple non lin. had to do this to make sure this plotting code works for other subtraction techniques
    gamma_beta_text = ' gamma: ' + str(alpha0) + '    beta: ' + str(beta)  #need to input gamma instead of alpha0 when calling this function for non_lin ss. 
    insetstring1 = (" xc (o,p):     long=" + str(round(metrics["maxcorr_long"],2)) + "(" + 
                     str(round(metrics["lag_long"]*tro.stats.delta,2)) + "s)  short=" + 
                     str(round(metrics["maxcorr_short"],2)) + "(" + 
                     str(round(metrics["lag_short"]*tro.stats.delta,2)) + "s)")
    insetstring2 = (" snr (d->p): long=" + str(round(metrics["snrd_long"],1)) + "->" + 
                     str(round(metrics["snrp_long"],1)) + "    short="  + 
                     str(round(metrics["snrd_short"],1)) + "->" + 
                     str(round(metrics["snrp_short"],1)) )
    titlestring = ( '$\Delta$SNR ' + 
                     str(round(100*metrics["snrp_long"]/metrics["snrd_long"],0)) + '%        30s-correlation ' + 
                     str(round(metrics["maxcorr_long"],2)) + '(' + 
                     str(metrics["lag_long"]*tro.stats.delta) + 's)' )       
    windowstart = round((picktime - trd.stats.starttime) - 40)
    windowend   = round((picktime - trd.stats.starttime) + 40)
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1,figsize=(7.5,10))
    maxamp = abs(amp_Xd).max()/2           # these two lines just adjust the color scale
    minamp = 0
    t, f = np.meshgrid(trd.times(), freqs_d)
    im1 = ax1.pcolormesh(t, f, np.abs(amp_Xd), cmap=cm.hot, vmin=minamp, vmax=maxamp)
    ax1.set_ylabel('frequency (Hz)')
    ax1.axes.xaxis.set_visible(False)
    ax1.set_title(ss_type + ' degraded CWT amplitude')
    ax1.set_xlim(windowstart, windowend)
    if ss_type == "simple":
        im2 = ax2.plot(t[1,:], tr_alpha.data, 'r--', linewidth=1,label='alpha')
        im2 = ax2.plot(t[1,:], tr_SNR.data/10, 'r-', linewidth=1,label='SNR')
        ax2.set_ylabel('0.1*SNR(db) & alpha')
    elif ss_type == "over":
        im2 = ax2.plot(t[1,:], tr_alpha.data, 'r--', linewidth=1,label='alpha')
        im2 = ax2.plot(t[1,:], tr_SNR.data/10, 'r-', linewidth=1,label='SNR')
        ax2.set_ylabel('0.1*SNR(db) & alpha')
        ax2.text(windowstart, -1,alpha_beta_text, style='italic', fontsize=8)
    elif ss_type == "smooth_over":
        im2 = ax2.plot(t[1,:], tr_alpha.data, 'r--', linewidth=1,label='alpha')
        im2 = ax2.plot(t[1,:], tr_SNR.data/10, 'r-', linewidth=1,label='SNR')
        ax2.set_ylabel('0.1*SNR(db) & alpha')
        ax2.text(windowstart, -1,alpha_beta_text, style='italic', fontsize=8)
    elif ss_type == "frequency_over":
        alpha_mean = np.mean(alpha,axis=0)
        SNR_mean = np.mean(SNR,axis=0)
        #im2 = ax2.plot(t[1,:], tr_alpha.data, 'r--', linewidth=1,label='alpha')
#        im2 = ax2.plot(t[1,:], alpha[1,:], 'r--', linewidth=1,label='alpha')
        im2 = ax2.plot(t[1,:], alpha_mean, 'r--', linewidth=1,label='alpha_mean')
        im2 = ax2.plot(t[1,:], 0.1*SNR_mean, 'r-', linewidth=1,label='SNR_mean')
        ax2.set_ylabel('0.1*SNR(db) & alpha (mean)')
        ylimits = ax2.get_ylim()
        ax2.text(windowstart, ylimits[0],alpha_beta_text, style='italic', fontsize=8)
    elif ss_type == "non_lin":
        #phi_mean = np.mean((10*np.log10(phi ** 2)),axis=0)
        SNR_mean = np.mean(SNR,axis=0)
        #im2 = ax2.plot(t[1,:], phi_mean, 'r--', linewidth=1,label='phi_mean')
        im2 = ax2.plot(t[1,:], SNR_mean ** (1/2), 'r-', linewidth=1,label='rho_mean')
        #im2 = ax2.plot(t[1,:], SNR[1], 'r-', linewidth=1,label='SNR')
        ax2.set_ylabel('SNR_mean^(1/2)(db) = rho')
        ylimits = ax2.get_ylim()
        ax2.text(windowstart, ylimits[0],gamma_beta_text, style='italic', fontsize=8)
    elif ss_type == "simple_non_lin":
       # phi = 10*np.log10(phi ** 2)    
       # phi_mean = np.mean(phi,axis=0)
        SNR_mean = np.mean(SNR,axis=0)
      #  im2 = ax2.plot(t[1,:], phi_mean, 'r--', linewidth=1,label='phi_mean')
        im2 = ax2.plot(t[1,:], SNR_mean, 'r-', linewidth=1,label='rho_mean')
        #im2 = ax2.plot(t[1,:], SNR[1], 'r-', linewidth=1,label='SNR')
        ax2.set_ylabel('SNR_mean(db) = rho')
        #ylimits = ax2.get_ylim()
        #ax2.text(windowstart, ylimits[0],gamma_beta_text, style='italic', fontsize=8)
    #ax2.set_yticks([-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7])
    ax2.grid()
    #ax2.set_ylim(0.1* tr_SNR.data.min(), 1.05*tr_alpha.data.max())
    ax2.legend(loc='upper left')
    ax2.axes.xaxis.set_visible(False)
    ax2.set_title(ss_type + ' subtraction parameters')
    ax2.set_xlim(windowstart, windowend)

    im3 = ax3.pcolormesh(t, f, np.abs(amp_Xp), cmap=cm.hot, vmin=minamp, vmax=maxamp)
    ax3.set_ylabel('frequency (Hz)')
    ax3.axes.xaxis.set_visible(False)
    ax3.set_title(ss_type + ' processed CWT amplitude')
    ax3.set_xlim(windowstart, windowend)

    im4 = ax4.plot(t[1,:], tro.data, 'r-', linewidth=.75,label='original')
    im4 = ax4.plot(t[1,:], trd.data, 'r:', linewidth=.75,label='degraded')
    im4 = ax4.plot(t[1,:], trp.data, 'k-', linewidth=.5,label='processed')
    ax4.set_ylabel('counts')
    ax4.legend(loc='upper left')
    ax4.set_xlabel('time (s)')
    ax4.set_title(titlestring)
    ax4.set_xlim(windowstart, windowend)
    ylimits = ax4.get_ylim()
    yrange = ylimits[1]-ylimits[0]
    ax4.text(windowstart, ylimits[0]+.10*yrange,insetstring2, style='italic', fontsize=8)
    ax4.text(windowstart, ylimits[0]+.02*yrange,insetstring1, style='italic', fontsize=8)
    fig.savefig(os.path.join(outpath, event_list + '_'+ station_list + '_' + figname + '_' + ss_type + '.png'))
    return 

def hilb_plot(t,hilb_div,max_hilb,mean_hilb,outpath,fig,event_list,station_list,figname="hilbert_metrics"):
    fig, ax1 = plt.subplots(1,1,figsize=(5,5))
    ax1.scatter(t, hilb_div, s=15, facecolors='none', edgecolors='r', linewidth=0.5)
    text = 'max = ' + str(round(max_hilb,2)) + '\n' + 'mean = ' + str(round(mean_hilb,2))    
    ob = offsetbox.AnchoredText(text, loc=1, prop=dict(color='black', size=8)) #ax.text location kept changing
    ob.patch.set(boxstyle='round', color='gray', alpha=0.5)
    ax1.add_artist(ob)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Phase lag (degrees)')
    fig.savefig(os.path.join(outpath, event_list + '_'+ station_list + '_' + figname + '.png'))

     

def stft(td, fd, tro, trd, trp, t_tmp, amp_Xd, amp_Xp, outpath, fig, event_list,station_list, figname="stft"):
    fig, (ax1, ax2, ax3) = plt.subplots(3,1)
    maxamp = abs(amp_Xd).max()/2           # these two lines just adjust the color scale
    minamp = 0
    t, f = np.meshgrid(td, fd)
    im1 = ax1.pcolormesh(t, f, np.abs(amp_Xd), cmap=cm.hot, vmin=minamp, vmax=maxamp)
    im2 = ax2.pcolormesh(t, f, np.abs(amp_Xp), cmap=cm.hot, vmin=minamp, vmax=maxamp)
    im3 = ax3.plot(tro.times(), tro.data, 'r-', linewidth=.75,label='original')
    im3 = ax3.plot(trd.times(), trd.data, 'r:', linewidth=.75,label='degraded')
    im3 = ax3.plot(t_tmp, trp.data, 'k-', linewidth=.5,label='processed')
    ax3.legend()
    ax1.set_ylabel('frequency (Hz)')
    ax2.set_ylabel('frequency (Hz)')
    ax3.set_ylabel('counts')
    ax3.set_xlabel('time (s)')
    ax1.set_title('degraded STFT amplitude')
    ax2.set_title('processed STFT amplitude')
    ax3.set_title(' STFT amplitude')
    fig.savefig(os.path.join(outpath,event_list + '_'+ station_list + '_' + figname + '.png'))


    
##################################################################################
##################################################################################
##################################################################################
###########VARIOUS TEST PLOTS FOR SPECTRAL SUBTRACTION TECHNIQUES#################
##################################################################################
##################################################################################
##################################################################################

def nonlin_signal_smooth(amp_Xd, amp_Xn, freqs_d, outpath, fig, event_list, station_list,timeframe="100", figname="signal_smooth_nonlin"):
    m, n = amp_Xd.shape 
    phi = np.zeros((m,n))
    phis = np.zeros((m,n))
    rho = np.zeros((m,n))
    rhos = np.zeros((m,n))
    amp_Xds = np.zeros((m,n))
    amp_Xp = np.zeros((m,n))
    amp_Xps = np.zeros((m,n))
    alpha = np.zeros((n))
    amp_Xna = np.mean(amp_Xn,axis=1)
    amp_Xna1 = np.max(amp_Xn,axis=1)

    amp_Xds1 = np.zeros((m,n))
    amp_Xds2 = np.zeros((m,n))
    amp_Xds3 = np.zeros((m,n))
    amp_Xds4 = np.zeros((m,n))
    amp_Xds5 = np.zeros((m,n))
    amp_Xps1 = np.zeros((m,n))
    amp_Xps2 = np.zeros((m,n))
    amp_Xps3 = np.zeros((m,n))
    amp_Xps4 = np.zeros((m,n))
    amp_Xps5 = np.zeros((m,n))
    
    muy = 0.3 #should be between 0.1-0.5
    #mud = 0.7 #should be between 0.5-0.9
    gamma = 0.5
    beta = 0.1
    i = timeframe   #   snr=9.1  alpha=2.6        alpha0-3/20*snr[7800]     
    alpha = np.max(amp_Xn,axis=1) # I think this makes more sense

    #WITHOUT SMOOTHING Xd
    rho[:,i] = amp_Xd[:,i]/amp_Xna
    phi[:,i] = alpha / (1 + (gamma*rho[:,i]))
    amp_Xp[:,i] = amp_Xd[:,i] - phi[:,i]
    foundlows = np.where(amp_Xd[:,i] < (phi[:,i] + (beta*amp_Xna)))
    amp_Xp[foundlows,i] = beta*amp_Xd[foundlows,i]

    #WITH SMOOTHING Xd
    amp_Xds[:,i] = (muy)*amp_Xd[:,(i-1)]+(1-muy)*amp_Xd[:,i] #smoothed estimate of degraded signal, same should be for the noise.
    rhos[:,i] = amp_Xds[:,i]/amp_Xna
    phis[:,i] = alpha / (1 + (gamma*rhos[:,i]))
    amp_Xps[:,i] = amp_Xds[:,i] - phis[:,i]
    foundlows = np.where(amp_Xds[:,i] < (phis[:,i] + (beta*amp_Xna)))
    amp_Xps[foundlows,i] = beta*amp_Xds[foundlows,i]

    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(7.5,10))
    ax1.plot(freqs_d,amp_Xd[:,i],       'k-',lw=.75,label='Xd, not smoothed ')
    ax1.plot(freqs_d,amp_Xp[:,i],       'k--',lw=.75,label='Xp, with not smoothed Xd')
    ax1.plot(freqs_d,amp_Xds[:,i],       'r-',lw=.75,label='Xd, smoothed')
    ax1.plot(freqs_d,amp_Xps[:,i],       'r--',lw=.75,label='Xp, with smoothed Xd ')
    ax1.legend()
    ax1.set_title('Xd vs smoothed Xd, \u03bcy=0.3 for timeframe ' + str(i))

    #VARYING MUY and MUD 
    muy = 0.1 #should be between 0.1-0.5
    #WITH SMOOTHING Xd
    amp_Xds1[:,i] = (muy)*amp_Xd[:,(i-1)]+(1-muy)*amp_Xd[:,i] #smoothed estimate of degraded signal, same should be for the noise.
    rhos[:,i] = amp_Xds1[:,i]/amp_Xna
    phis[:,i] = alpha / (1 + (gamma*rhos[:,i]))
    amp_Xps1[:,i] = amp_Xds1[:,i] - phis[:,i]
    foundlows = np.where(amp_Xds1[:,i] < (phis[:,i] + (beta*amp_Xna)))
    amp_Xps1[foundlows,i] = beta*amp_Xds1[foundlows,i]

    muy = 0.3 #should be between 0.1-0.5
    #WITH SMOOTHING Xd
    amp_Xds2[:,i] = (muy)*amp_Xd[:,(i-1)]+(1-muy)*amp_Xd[:,i] #smoothed estimate of degraded signal, same should be for the noise.
    rhos[:,i] = amp_Xds2[:,i]/amp_Xna
    phis[:,i] = alpha / (1 + (gamma*rhos[:,i]))
    amp_Xps2[:,i] = amp_Xds2[:,i] - phis[:,i]
    foundlows = np.where(amp_Xds2[:,i] < (phis[:,i] + (beta*amp_Xna)))
    amp_Xps2[foundlows,i] = beta*amp_Xds2[foundlows,i]


    muy = 0.5 #should be between 0.1-0.5
    #WITH SMOOTHING Xd
    amp_Xds3[:,i] = (muy)*amp_Xd[:,(i-1)]+(1-muy)*amp_Xd[:,i] #smoothed estimate of degraded signal, same should be for the noise.
    rhos[:,i] = amp_Xds3[:,i]/amp_Xna
    phis[:,i] = alpha / (1 + (gamma*rhos[:,i]))
    amp_Xps3[:,i] = amp_Xds3[:,i] - phis[:,i]
    foundlows = np.where(amp_Xds3[:,i] < (phis[:,i] + (beta*amp_Xna)))
    amp_Xps3[foundlows,i] = beta*amp_Xds3[foundlows,i]

    muy = 0.7 #should be between 0.1-0.5
    #WITH SMOOTHING Xd
    amp_Xds4[:,i] = (muy)*amp_Xd[:,(i-1)]+(1-muy)*amp_Xd[:,i] #smoothed estimate of degraded signal, same should be for the noise.
    rhos[:,i] = amp_Xds4[:,i]/amp_Xna
    phis[:,i] = alpha / (1 + (gamma*rhos[:,i]))
    amp_Xps4[:,i] = amp_Xds4[:,i] - phis[:,i]
    foundlows = np.where(amp_Xds4[:,i] < (phis[:,i] + (beta*amp_Xna)))
    amp_Xps4[foundlows,i] = beta*amp_Xds4[foundlows,i]

    muy = 1.0 #should be between 0.1-0.5
    #WITH SMOOTHING Xd
    amp_Xds[:,i] = (muy)*amp_Xd[:,(i-1)]+(1-muy)*amp_Xd[:,i] #smoothed estimate of degraded signal, same should be for the noise.
    rhos[:,i] = amp_Xds5[:,i]/amp_Xna
    phis[:,i] = alpha / (1 + (gamma*rhos[:,i]))
    amp_Xps5[:,i] = amp_Xds5[:,i] - phis[:,i]
    foundlows = np.where(amp_Xds5[:,i] < (phis[:,i] + (beta*amp_Xna)))
    amp_Xps5[foundlows,i] = beta*amp_Xds5[foundlows,i]

    ax2.plot(freqs_d,amp_Xds1[:,i],       'b-',lw=.75,label='Xds, \u03bcy=0.1')
    ax2.plot(freqs_d,amp_Xps1[:,i],       'b--',lw=.75,label='Xps, \u03bcy=0.1')
    ax2.plot(freqs_d,amp_Xds2[:,i],       'r-',lw=.75,label='Xds, \u03bcy=0.3')
    ax2.plot(freqs_d,amp_Xps2[:,i],       'r--',lw=.75,label='Xps, \u03bcy=0.3')
    ax2.plot(freqs_d,amp_Xds3[:,i],       'k-',lw=.75,label='Xds, \u03bcy=0.5')
    ax2.plot(freqs_d,amp_Xps3[:,i],       'k--',lw=.75,label='Xps, \u03bcy=0.5')
    ax2.plot(freqs_d,amp_Xds4[:,i],       'g-',lw=.75,label='Xds, \u03bcy=0.7')
    ax2.plot(freqs_d,amp_Xps4[:,i],       'g--',lw=.75,label='Xps, \u03bcy=0.7')
    ax2.plot(freqs_d,amp_Xds5[:,i],       'y-',lw=.75,label='Xds, \u03bcy=0.9')
    ax2.plot(freqs_d,amp_Xps5[:,i],       'y--',lw=.75,label='Xps, \u03bcy=0.9')
    ax2.legend()
    ax2.set_title('Smoothed Xd with varying \u03bcy for timeframe ' + str(i))


    fig.savefig(os.path.join(outpath, event_list + '_'+ station_list + '_' + figname + '.png'))

def nonlin_param_one_tf(amp_Xd, amp_Xn, freqs_d, gamma, beta, outpath, fig, event_list, station_list, timeframe="100", figname="one_timeframe_nonlin"):
    m, n = amp_Xd.shape 
    muy = 0.3 #should be between 0.1-0.5
    #p = 2
    phi = np.zeros((m,n))
    phi0 = np.zeros((m,n))
    phi1 = np.zeros((m,n))
    phi2 = np.zeros((m,n))
    phi3 = np.zeros((m,n))
    phi4 = np.zeros((m,n))
    phi5 = np.zeros((m,n))
    rho = np.zeros((m,n))
    amp_Xds = np.zeros((m,n))
    amp_Xp = np.zeros((m,n))
    amp_Xp0 = np.zeros((m,n))
    amp_Xp1 = np.zeros((m,n))
    amp_Xp2 = np.zeros((m,n))
    amp_Xp3 = np.zeros((m,n))
    amp_Xp4 = np.zeros((m,n))
    amp_Xp5 = np.zeros((m,n))
    amp_Xp6 = np.zeros((m,n))
    amp_Xp7 = np.zeros((m,n))
    amp_Xp8 = np.zeros((m,n))
    amp_Xp9 = np.zeros((m,n))
    amp_Xp10 = np.zeros((m,n))
    amp_Xp5nobeta5 = np.zeros((m,n))
    alpha = np.zeros((n))
    amp_Xna = np.mean(amp_Xn,axis=1)
    #amp_XdP = amp_Xd**p                     # appended P is shorthand for **p
    #amp_XnaP = amp_Xna**p

    
    # EXPLORE NON LINEAR SPECTRAL SUBTRACTION PARAMETERS
    
    #######################################################################################################################################
    # PLOT ROLE OF PHI
    #######################################################################################################################################    
    # SELECT A SINGLE TIME FRAME TO TEST  
    i = timeframe   #   snr=9.1  alpha=2.6        alpha0-3/20*snr[7800]     
    alpha = np.max(amp_Xn,axis=1) # I think this makes more sense
    amp_Xds[:,i] = (muy)*amp_Xd[:,(i-1)]+(1-muy)*amp_Xd[:,i] 
    rho[:,i] = amp_Xds[:,i]/amp_Xna

    gamma = 0.0
    phi0[:,i] = alpha / (1 + (gamma*rho[:,i]))
    gamma = 0.1
    phi1[:,i] = alpha / (1 + (gamma*rho[:,i]))
    gamma = 0.3
    phi2[:,i] = alpha / (1 + (gamma*rho[:,i]))
    gamma = 0.5
    phi3[:,i] = alpha / (1 + (gamma*rho[:,i]))
    gamma = 0.7
    phi4[:,i] = alpha / (1 + (gamma*rho[:,i]))
    gamma = 1.0
    phi5[:,i] = alpha / (1 + (gamma*rho[:,i]))



    #COMPARE GAMMA FOR PHI CALCULATION 
    # 900C3F C70039 FF5733 FFC305
    fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize=(7.5,10))
    ax1.plot(freqs_d,phi0[:,i],       'k:',lw=.75,label='phi, gamma = 0.0')
    ax1.plot(freqs_d,phi1[:,i],       'k--',lw=.75,label='phi, gamma = 0.1')
    ax1.plot(freqs_d,phi2[:,i],       'b:',lw=.75,label='phi, gamma = 0.3')
    ax1.plot(freqs_d,phi3[:,i],       'b--',lw=.75,label='phi, gamma = 0.5')
    ax1.plot(freqs_d,phi4[:,i],       'r:',lw=.75,label='phi, gamma = 0.7')
    ax1.plot(freqs_d,phi5[:,i],       'r--',lw=.75,label='phi, gamma = 1.0')
    ax1.plot(freqs_d,alpha,       'g-',lw=.75,label='alpha')
    ax1.legend()
    #ax1.set(xlim=[.5, 10])
    ax1.set_title('timeframe ' + str(i))


    beta = 0.5
    amp_Xp0[:,i] = (amp_Xds[:,i]) - phi0[:,i]
    foundlows = np.where(amp_Xds[:,i] < (phi0[:,i] + (beta*amp_Xna)))
    amp_Xp0[foundlows,i] = beta*amp_Xds[foundlows,i]

    amp_Xp1[:,i] = (amp_Xds[:,i]) - phi1[:,i]
    foundlows = np.where(amp_Xds[:,i] < (phi1[:,i] + (beta*amp_Xna)))
    amp_Xp1[foundlows,i] = beta*amp_Xds[foundlows,i]

    amp_Xp2[:,i] = (amp_Xds[:,i]) - phi2[:,i]
    foundlows = np.where(amp_Xds[:,i] < (phi2[:,i] + (beta*amp_Xna)))
    amp_Xp2[foundlows,i] = beta*amp_Xds[foundlows,i]

    amp_Xp3[:,i] = (amp_Xds[:,i]) - phi3[:,i]
    foundlows = np.where(amp_Xds[:,i] < (phi3[:,i] + (beta*amp_Xna)))
    amp_Xp3[foundlows,i] = beta*amp_Xds[foundlows,i]

    amp_Xp4[:,i] = (amp_Xds[:,i]) - phi4[:,i]
    foundlows = np.where(amp_Xds[:,i] < (phi4[:,i] + (beta*amp_Xna)))
    amp_Xp4[foundlows,i] = beta*amp_Xds[foundlows,i]

    amp_Xp5[:,i] = (amp_Xds[:,i]) - phi5[:,i]
    amp_Xp5nobeta5[:,i] = amp_Xp5[:,i] 
    foundlows = np.where(amp_Xds[:,i] < (phi5[:,i] + (beta*amp_Xna)))
    amp_Xp5[foundlows,i] = beta*amp_Xds[foundlows,i]
        
    ax2.plot(freqs_d,amp_Xds[:,i],       'k-',lw=1.25,label='Xds')
    ax2.plot(freqs_d,amp_Xp0[:,i],       '-',lw=.75,label='gamma = 0.0',color='navy')
    ax2.plot(freqs_d,amp_Xp1[:,i],       '-',lw=.75,label='gamma = 0.1',color='darkblue')
    ax2.plot(freqs_d,amp_Xp2[:,i],       '-',lw=.75,label='gamma = 0.3',color='indigo')
    ax2.plot(freqs_d,amp_Xp3[:,i],       '-',lw=.75,label='gamma = 0.5',color='cyan')
    ax2.plot(freqs_d,amp_Xp4[:,i],       '-',lw=.75,label='gamma = 0.7',color='aqua')
    ax2.plot(freqs_d,amp_Xp5[:,i],       '-',lw=.75,label='gamma = 1.0',color='turquoise')
    ax2.plot(freqs_d,amp_Xp5nobeta5[:,i],':',lw=1.75,label='alpha0=9, no beta',color='red')
    ax2.plot(freqs_d,beta*amp_Xna,      'g:',lw=1.75,label='beta=' + str(beta) + '*Xna')
    ax2.plot(freqs_d,beta*amp_Xds[:,i],      'k:',lw=1.75,label='beta=' + str(beta) + '*Xds')
    ax2.legend()
    
    gamma = 0.5
    beta = 0.1
    amp_Xp6[:,i] = (amp_Xds[:,i]) - phi3[:,i]
    foundlows = np.where(amp_Xds[:,i] < (phi3[:,i] + (beta*amp_Xna)))
    amp_Xp6[foundlows,i] = beta*amp_Xds[foundlows,i]

    beta = 0.3
    amp_Xp7[:,i] = (amp_Xds[:,i]) - phi3[:,i]
    foundlows = np.where(amp_Xds[:,i] < (phi3[:,i] + (beta*amp_Xna)))
    amp_Xp7[foundlows,i] = beta*amp_Xds[foundlows,i]

    beta = 0.5
    amp_Xp8[:,i] = (amp_Xds[:,i]) - phi3[:,i]
    foundlows = np.where(amp_Xds[:,i] < (phi3[:,i] + (beta*amp_Xna)))
    amp_Xp8[foundlows,i] = beta*amp_Xds[foundlows,i]

    beta = 0.7
    amp_Xp9[:,i] = (amp_Xds[:,i]) - phi3[:,i]
    foundlows = np.where(amp_Xds[:,i] < (phi3[:,i] + (beta*amp_Xna)))
    amp_Xp9[foundlows,i] = beta*amp_Xds[foundlows,i]

    beta = 1.0
    amp_Xp10[:,i] = (amp_Xds[:,i]) - phi3[:,i]
    foundlows = np.where(amp_Xds[:,i] < (phi3[:,i] + (beta*amp_Xna)))
    amp_Xp10[foundlows,i] = beta*amp_Xds[foundlows,i]



    ax3.plot(freqs_d,phi3[:,i],       ':',lw=.75,label='phi, gamma=' + str(gamma))
    ax3.plot(freqs_d,amp_Xds[:,i],       'k-',lw=.75,label='Xds')
    #ax3.plot(freqs_d,amp_Xna[:,i],       'k--',lw=.75,label='Xna')
    ax3.plot(freqs_d,amp_Xp6[:,i],       '-',lw=.75,label='beta = 0.1',color='#900C3F')
    ax3.plot(freqs_d,amp_Xp7[:,i],       '-',lw=.75,label='beta = 0.3',color='#C70039')
    ax3.plot(freqs_d,amp_Xp8[:,i],       '-',lw=.75,label='beta = 0.5',color='#FF5733')
    ax3.plot(freqs_d,amp_Xp9[:,i],       '-',lw=.75,label='beta = 0.7',color='#AC063C')
    ax3.plot(freqs_d,amp_Xp10[:,i],       '-',lw=.75,label='beta = 1.0',color='#FFC305')
    #ax3.plot(freqs_d,beta*amp_Xna,      'k:',lw=1.75,label='beta=' + str(beta) + '*Xna')
    #ax3.plot(freqs_d,beta*amp_Xds[:,i],      'g:',lw=1.75,label='beta=' + str(beta) + '*Xds')
    ax3.legend()

    fig.savefig(os.path.join(outpath, event_list + '_'+ station_list + '_' + figname + '.png'))

def simple_nonlin_param_one_tf(amp_Xd, amp_Xn, freqs_d, gamma, beta, outpath, fig, event_list, station_list, timeframe="100", figname="one_timeframe_nonlin"):
    m, n = amp_Xd.shape 
    #muy = 0.3 #should be between 0.1-0.5
    #p = 2
    phi = np.zeros((m,n))
    phi0 = np.zeros((m,n))
    phi1 = np.zeros((m,n))
    phi2 = np.zeros((m,n))
    phi3 = np.zeros((m,n))
    phi4 = np.zeros((m,n))
    phi5 = np.zeros((m,n))
    rho = np.zeros((m,n))
    rho0 = np.zeros((m,n))
    rho1 = np.zeros((m,n))
    rho2 = np.zeros((m,n))
    #amp_Xds = np.zeros((m,n))
    amp_Xp = np.zeros((m,n))
    amp_Xp0 = np.zeros((m,n))
    amp_Xp1 = np.zeros((m,n))
    amp_Xp2 = np.zeros((m,n))
    amp_Xp3 = np.zeros((m,n))
    amp_Xp4 = np.zeros((m,n))
    amp_Xp5nobeta5 = np.zeros((m,n))
    alpha = np.zeros((n))
    alpha_mean = np.zeros((n))
    amp_Xna = np.mean(amp_Xn,axis=1)

    
    # EXPLORE SIMPLE NON LINEAR SPECTRAL SUBTRACTION PARAMETERS
    
    #######################################################################################################################################
    # PLOT ROLE OF PHI
    #######################################################################################################################################    
    # SELECT A SINGLE TIME FRAME TO TEST  
    i = timeframe   #   snr=9.1  alpha=2.6        alpha0-3/20*snr[7800]     
    alpha = np.max(amp_Xn,axis=1) # I think this makes more sense
    alpha_mean = np.mean(amp_Xn,axis=1) # I think this makes more sense
    #amp_Xds[:,i] = (muy)*amp_Xd[:,(i-1)]+(1-muy)*amp_Xd[:,i] 

    gamma = 0.5
    i0 = 300
    rho0[:,i0] = amp_Xd[:,i0]/amp_Xna
    phi0[:,i0] = (2*alpha_mean) / (1+(gamma*rho0[:,i0]))
#    phi0[:,i0] = alpha / rho0[:,i0]
    i1 = 500
    rho1[:,i1] = amp_Xd[:,i1]/amp_Xna
#    phi1[:,i1] = alpha / rho1[:,i1]
    phi1[:,i1] = (2*alpha_mean) / (1+(gamma*rho1[:,i1]))


    #COMPARE PHI CALCULATION 
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(7.5,10))
    ax1.plot(freqs_d,phi0[:,i0],       'r-',lw=.95,label='low SNR, timeframe ' + str(i0))
#    ax1.plot(freqs_d,phi0[:,i0],       'r:',lw=.95,label='low SNR, timeframe ' + str(i0) + ', gamma ' + str(gamma0))
    ax1.plot(freqs_d,phi1[:,i1],       'b-',lw=.95,label='high SNR, timeframe ' + str(i1))
   # ax1.plot(freqs_d,phi4[:,i],       'g:',lw=.75,label='phi, alpha/1+(gamma(0.5)*rho)')
    #ax1.plot(freqs_d,alpha,       'r-',lw=.75,label='alpha(Xn_max)')
    #ax1.plot(freqs_d,alpha_mean,       'b-',lw=.75,label='alpha(Xn_mean)')
    ax1.legend()
#    ax1.set_title('timeframe ' + str(i))
    ax1.set_title('alpha = 2/(1+gamma*SNR)')
 #   ax1.set(xlim=[0, 20])
 #   ax1.set(ylim=[0.1, 1000])
    ax1.set_xlabel('frequency')
    ax1.set_ylabel('alpha*Xn_mean')


    
    beta = 0
    amp_Xp0[:,i0] = (amp_Xd[:,i0]) - phi0[:,i0]
    foundlows = np.where(amp_Xd[:,i0] < (phi0[:,i0] + (beta*amp_Xna)))
    amp_Xp0[foundlows,i0] = beta*amp_Xd[foundlows,i0]


    amp_Xp1[:,i1] = (amp_Xd[:,i1]) - phi1[:,i1]
    foundlows = np.where(amp_Xd[:,i1] < (phi1[:,i1] + (beta*amp_Xna)))
    amp_Xp1[foundlows,i1] = beta*amp_Xd[foundlows,i1]


    ax2.plot(freqs_d,amp_Xd[:,i0],       'r:',lw=.95,label='Xd, low SNR ',color='red')
    ax2.plot(freqs_d,amp_Xd[:,i1],       'b:',lw=.95,label='Xd, high SNR ',color='blue')
    #ax2.plot(freqs_d,0.5*amp_Xd[:,i],       'k:',lw=1.75,label='Xd* beta = 0.5')
    ax2.plot(freqs_d,amp_Xna,       'k--',lw=1.75,label='Xn')
    ax2.plot(freqs_d,amp_Xp0[:,i0],       '-',lw=.95,label='Xp, low SNR, beta = 0',color='red')
    ax2.plot(freqs_d,amp_Xp1[:,i1],       '-',lw=.95,label='Xp, high SNR, beta = 0',color='blue')
#    ax2.plot(freqs_d,amp_Xp2[:,i],       '-',lw=.75,label='beta = 0.5',color='cyan')
#    ax2.plot(freqs_d,amp_Xp3[:,i],       '-',lw=.75,label='beta = 0.7',color='lightblue')
    ax2.legend()
    #ax2.set_title('phi = alpha(Xn_max)/rho')

    fig.savefig(os.path.join(outpath, event_list + '_'+ station_list + '_' + figname + '.png'))

def simple_nonlin_phi(amp_Xd, amp_Xn, freqs_d, gamma, beta, outpath, fig, event_list, station_list, timeframe="100", figname="phi_simple_nonlin"):
    m, n = amp_Xd.shape 
    #muy = 0.3 #should be between 0.1-0.5
    #p = 2
    phi = np.zeros((m,n))
    phi0 = np.zeros((m,n))
    phi1 = np.zeros((m,n))
    phi2 = np.zeros((m,n))
    phi3 = np.zeros((m,n))
    phi4 = np.zeros((m,n))
    phi5 = np.zeros((m,n))
    phi6 = np.zeros((m,n))
    phi7 = np.zeros((m,n))
    rho = np.zeros((m,n))
    rho0 = np.zeros((m,n))
    rho1 = np.zeros((m,n))
    rho2 = np.zeros((m,n))
    rho3 = np.zeros((m,n))
    rho4 = np.zeros((m,n))
    rho5 = np.zeros((m,n))
    rho6 = np.zeros((m,n))
    rho7 = np.zeros((m,n))
    alpha = np.zeros((n))
    alpha_mean = np.zeros((n))
    amp_Xna = np.mean(amp_Xn,axis=1)

    
    # EXPLORE SIMPLE NON LINEAR SPECTRAL SUBTRACTION PARAMETERS
    
    #######################################################################################################################################
    # PLOT ROLE OF PHI
    #######################################################################################################################################    
    # SELECT A SINGLE TIME FRAME TO TEST  
    i = timeframe   #   snr=9.1  alpha=2.6        alpha0-3/20*snr[7800]  
    alpha = np.max(amp_Xn,axis=1) # I think this makes more sense
    alpha_mean = np.mean(amp_Xn,axis=1) 
    #amp_Xds[:,i] = (muy)*amp_Xd[:,(i-1)]+(1-muy)*amp_Xd[:,i] 
    gamma0 = 0.5

    i0 = 300
    rho0[:,i0] = amp_Xd[:,i0]/amp_Xna
#    phi0[:,i0] = alpha  / rho0[:,i0]
    phi0[:,i0] = (2 * alpha_mean) / (1 + (gamma0*rho0[:,i0]))
 
    i1 = 500
    rho1[:,i1] = amp_Xd[:,i1]/amp_Xna
#    phi1[:,i1] = alpha / rho1[:,i1]
    phi1[:,i1] = (2 * alpha_mean) / (1 + (gamma0*rho1[:,i1]))

    gamma1 = 0.1

    i0 = 300
    rho2[:,i0] = amp_Xd[:,i0]/amp_Xna
#   phi2[:,i0] = alpha / rho2[:,i0]
    phi2[:,i0] = (2 * alpha_mean)  / (1 + (gamma1*rho2[:,i0]))

    i1 = 500
    rho3[:,i1] = amp_Xd[:,i1]/amp_Xna
#    phi3[:,i1] = alpha / rho3[:,i1]
    phi3[:,i1] = (2 * alpha_mean) / (1 + (gamma1*rho3[:,i1]))


    #gamma = 0.1
#
    #phi2[:,i] = alpha / (1 + gamma*rho[:,i])
#
    #phi3[:,i] = alpha_mean / (1 + gamma*rho[:,i])
#
    #gamma = 0.5
#
    #phi4[:,i] = alpha / (1 + (gamma*rho[:,i]))
#
    #phi5[:,i] = alpha_mean / (1 + (gamma*rho[:,i]))
#
    #gamma = 1.0
#
    #phi6[:,i] = alpha / (1 + (gamma*rho[:,i]))
#
    #phi7[:,i] = alpha_mean / (1 + (gamma*rho[:,i]))
#

    #COMPARE PHI CALCULATION 
    fig, (ax1) = plt.subplots(1,1,figsize=(7.5,10))
    #ax1.plot(rho[:,i],phi0[:,i],       'r-',lw=.75,label='phi, = 1/rho')
    #ax1.plot(rho[:,i],phi1[:,i],       'r:',lw=.75,label='phi = alpha_min/rho')
    ax1.semilogy(rho0[20:66,i0],phi0[20:66,i0],       'r-',lw=.95,label='low SNR, timeframe ' + str(i0) + ', gamma ' + str(gamma0))
    ax1.semilogy(rho1[20:66,i1],phi1[20:66,i1],       'b-',lw=.95,label='high SNR, timeframe ' + str(i1) + ', gamma ' + str(gamma0))
    ax1.semilogy(rho2[20:66,i0],phi2[20:66,i0],       'r:',lw=.95,label='low SNR, timeframe ' + str(i0) + ', gamma ' + str(gamma1))
    ax1.semilogy(rho3[20:66,i1],phi3[20:66,i1],       'b:',lw=.95,label='high SNR, timeframe ' + str(i1) + ', gamma ' + str(gamma1))
     #ax1.plot(rho[:,i],phi3[:,i],       'b:',lw=.75,label='phi = alpha_mean & gamma = 0.1')
    #ax1.semilogy(rho[20:66,i],phi5[20:66,i],       'g-',lw=.95,label='phi =  gamma = 0.5')
    #ax1.plot(rho[:,i],phi5[:,i],       'g:',lw=.75,label='phi = alpha_mean & gamma = 0.5')
    #ax1.semilogy(rho[20:66,i],phi7[20:66,i],       'y-',lw=.95,label='phi = gamma = 1.0')
    #ax1.semilogx(rho[:,i],phi7[:,i],       'y:',lw=.95,label='phi = gamma = 0.5')
    #ax1.plot(rho[:,i],alpha,       '-',lw=.75,label='alpha_max')
    #ax1.plot(rho[:,i],alpha_mean,       ':',lw=.75,label='alpha_mean')
    ax1.legend()
#    ax1.set_title('timeframe ' + str(i) + '  phi = alpha(Xn_mean)/rho')
    ax1.set_title('alpha = 2/(1+gamma*SNR)')
    ax1.set(xlim=[0, 20])
    ax1.set(ylim=[0.1, 1000])
    ax1.set_xlabel('SNR')
    ax1.set_ylabel('alpha*Xn_mean') 

    fig.savefig(os.path.join(outpath, event_list + '_'+ station_list + '_' + figname + '.png'))

def oversub_param_one_tf(amp_Xd, amp_Xn, freqs_d, ss_type, outpath, fig, event_list, station_list, timeframe="100", figname="one_timeframe_oversub"):
    m, n = amp_Xd.shape 
    p = 2
    muy = 0.3 # you might need to put this into function definition if you end up using smooth over sub a lot
    amp_Xna = np.mean(amp_Xn,axis=1)
    amp_XdP = amp_Xd**p                     # appended P is shorthand for **p
    amp_XnaP = amp_Xna**p
    amp_Xds = np.zeros((m,n))
    amp_XdsP = amp_Xds ** p
    SNR = np.zeros((1,n))
    alpha = np.zeros((1,n))
    # EXPLORE SPECTRAL SUBTRACTION PARAMETERS
    #alpha0=4 #vary between 3-6 (Beruiti et al'79), normally taken as 4 (Kamath & Loizou'02)
    #beta=0.2 #should be between 0-1.
    amp_XpP = np.zeros((m,n))
    amp_XpP1 = np.zeros((m,n))
    amp_XpP2 = np.zeros((m,n))
    amp_XpP3 = np.zeros((m,n))
    amp_XpP4 = np.zeros((m,n))
    amp_XpP5 = np.zeros((m,n))
    amp_XpP6 = np.zeros((m,n))
    amp_XpP7 = np.zeros((m,n))
    amp_XpP8 = np.zeros((m,n))
    amp_XpP9 = np.zeros((m,n))
    amp_XpPnobeta1 = np.zeros((m,n))
    amp_XpPnobeta2 = np.zeros((m,n))
    amp_XpPnobeta3 = np.zeros((m,n))
    amp_XpPnobeta4 = np.zeros((m,n))
    amp_XpPnobeta5 = np.zeros((m,n))
    amp_XpPnobeta6 = np.zeros((m,n))
    amp_XpPnobeta7 = np.zeros((m,n))
    amp_XpPnobeta8 = np.zeros((m,n))
    amp_XpPnobeta9 = np.zeros((m,n))
    
    #######################################################################################################################################
    # PLOT ROLE OF ALPHA AND BETA
    #######################################################################################################################################    
    # SELECT A SINGLE TIME FRAME TO TEST  
    #j = 7000   #   snr=0.6  alpha=3.9        alpha0-3/20*snr[7000]
    #j = 7500   #   snr=5.2  alpha=3.2        alpha0-3/20*snr[7500]
    j = timeframe   #   snr=9.1  alpha=2.6        alpha0-3/20*snr[7800]     
    #j = 7800   #   snr=9.1  alpha=2.6        alpha0-3/20*snr[7800]  
    beta = .5
    alpha0 = 4.0
    if ss_type == "over":
        SNR[:,j] = np.sum(amp_XdP[:,j]) / np.sum(amp_XnaP)
    elif ss_type == "smooth_over":
        amp_Xds[:,j] = (muy)*amp_Xd[:,(j-1)]+(1-muy)*amp_Xd[:,j] 
        amp_XdsP[:,j] = amp_Xds[:,j] ** p
        SNR[:,j] = np.sum(amp_XdsP[:,j]) / np.sum(amp_XnaP)
        amp_XdP[:,j] = amp_XdsP[:,j] # to avoid confusion for the rest of the plots. not sure if this is right thing to do. 
    SNR[:,j] = 10*np.log10(SNR[:,j]) #convert snr to decibels
    if SNR[:,j] < -5:
        alpha[:,j] = alpha0+(3/4)
    elif SNR[:,j] >= -5 and SNR[:,j] < 20:
        alpha[:,j] = alpha0-(3/20*SNR[:,j])
    elif SNR[:,j] >= 20:
        alpha[:,j] = alpha0-3 
    amp_XpP2[:,j] = amp_XdP[:,j] - alpha[:,j]*amp_XnaP  
    foundlows = np.where(amp_XpP2[:,j] < beta*amp_XnaP)
    amp_XpP2[foundlows,j] = beta*amp_XnaP[foundlows]
    #COMPARE 
    # 900C3F C70039 FF5733 FFC305
    fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize=(7.5,10))
    ax1.plot(freqs_d,amp_XdP[:,j],       'k-',lw=.75,label='Xd, or Xds if smooth over ss')
    ax1.plot(freqs_d,amp_XnaP,           'k--',lw=.75,label='Xna')
    #ax1.plot(freqs_d,amp_XpPnobeta2[:,j],':',lw=.75,label='alpha=2.3, no beta',color='#C70039')
    ax1.plot(freqs_d,amp_XpP2[:,j],      '-',lw=.75,label='Xp, alpha0=' + str(alpha0) + ', beta=' + str(beta),color='#C70039')
    ax1.plot(freqs_d,beta*amp_XnaP,      'k:',lw=.75,label='beta*Xna')
    ax1.legend()
    ax1.set(xlim=[.5, 10])
    ax1.set_title('timeframe ' + str(j))

    #######################################################################################################################################
    # MAKE A PLOT COMPARING DIFFERENT ALPHAS
    #######################################################################################################################################    
    # SELECT A SINGLE TIME FRAME TO TEST  
    #j = 7000   #   snr=0.6  alpha=3.9        alpha0-3/20*snr[7000]
    #j = 7500   #   snr=5.2  alpha=3.2        alpha0-3/20*snr[7500]
    j = timeframe   #   snr=9.1  alpha=2.6        alpha0-3/20*snr[7800]     
    #j = 7800   #   snr=9.1  alpha=2.6        alpha0-3/20*snr[7800]     
    alpha0 = 1
    if ss_type == "over":
        SNR[:,j] = np.sum(amp_XdP[:,j]) / np.sum(amp_XnaP)
    elif ss_type == "smooth_over":
        amp_Xds[:,j] = (muy)*amp_Xd[:,(j-1)]+(1-muy)*amp_Xd[:,j] 
        amp_XdsP[:,j] = amp_Xds[:,j] ** p
        SNR[:,j] = np.sum(amp_XdsP[:,j]) / np.sum(amp_XnaP)
        amp_XdP[:,j] = amp_XdsP[:,j] # to avoid confusion for the rest of the plots. not sure if this is right thing to do.         
    SNR[:,j] = 10*np.log10(SNR[:,j]) #convert snr to decibels
    if SNR[:,j] < -5:
        alpha[:,j] = alpha0+(3/4)
    elif SNR[:,j] >= -5 and SNR[:,j] < 20:
        alpha[:,j] = alpha0-(3/20*SNR[:,j])
    elif SNR[:,j] >= 20:
        alpha[:,j] = alpha0-3 
    amp_XpP1[:,j] = amp_XdP[:,j] - (alpha[:,j]*amp_XnaP)  
    amp_XpPnobeta1[:,j] = amp_XpP1[:,j] 
    foundlows = np.where(amp_XpP1[:,j] < beta*amp_XnaP)
    amp_XpP1[foundlows,j] = beta*amp_XnaP[foundlows]
    alpha0 = 2
    if ss_type == "over":
        SNR[:,j] = np.sum(amp_XdP[:,j]) / np.sum(amp_XnaP)
    elif ss_type == "smooth_over":
        amp_Xds[:,j] = (muy)*amp_Xd[:,(j-1)]+(1-muy)*amp_Xd[:,j] 
        amp_XdsP[:,j] = amp_Xds[:,j] ** p
        SNR[:,j] = np.sum(amp_XdsP[:,j]) / np.sum(amp_XnaP)
        amp_XdP[:,j] = amp_XdsP[:,j] # to avoid confusion for the rest of the plots. not sure if this is right thing to do. 
    SNR[:,j] = 10*np.log10(SNR[:,j]) #convert snr to decibels
    if SNR[:,j] < -5:
        alpha[:,j] = alpha0+(3/4)
    elif SNR[:,j] >= -5 and SNR[:,j] < 20:
        alpha[:,j] = alpha0-(3/20*SNR[:,j])
    elif SNR[:,j] >= 20:
        alpha[:,j] = alpha0-3 
    amp_XpP2[:,j] = amp_XdP[:,j] - (alpha[:,j]*amp_XnaP)  
    amp_XpPnobeta2[:,j] = amp_XpP2[:,j] 
    foundlows = np.where(amp_XpP2[:,j] < beta*amp_XnaP)
    amp_XpP2[foundlows,j] = beta*amp_XnaP[foundlows]
    alpha0 = 3
    if ss_type == "over":
        SNR[:,j] = np.sum(amp_XdP[:,j]) / np.sum(amp_XnaP)
    elif ss_type == "smooth_over":
        amp_Xds[:,j] = (muy)*amp_Xd[:,(j-1)]+(1-muy)*amp_Xd[:,j] 
        amp_XdsP[:,j] = amp_Xds[:,j] ** p
        SNR[:,j] = np.sum(amp_XdsP[:,j]) / np.sum(amp_XnaP)
        amp_XdP[:,j] = amp_XdsP[:,j] # to avoid confusion for the rest of the plots. not sure if this is right thing to do. 
    SNR[:,j] = 10*np.log10(SNR[:,j]) #convert snr to decibels
    if SNR[:,j] < -5:
        alpha[:,j] = alpha0+(3/4)
    elif SNR[:,j] >= -5 and SNR[:,j] < 20:
        alpha[:,j] = alpha0-(3/20*SNR[:,j])
    elif SNR[:,j] >= 20:
        alpha[:,j] = alpha0-3 
    amp_XpP3[:,j] = amp_XdP[:,j] - (alpha[:,j]*amp_XnaP)  
    amp_XpPnobeta3[:,j] = amp_XpP3[:,j] 
    foundlows = np.where(amp_XpP3[:,j] < beta*amp_XnaP)
    amp_XpP3[foundlows,j] = beta*amp_XnaP[foundlows]
    alpha0 = 4
    if ss_type == "over":
        SNR[:,j] = np.sum(amp_XdP[:,j]) / np.sum(amp_XnaP)
    elif ss_type == "smooth_over":
        amp_Xds[:,j] = (muy)*amp_Xd[:,(j-1)]+(1-muy)*amp_Xd[:,j] 
        amp_XdsP[:,j] = amp_Xds[:,j] ** p
        SNR[:,j] = np.sum(amp_XdsP[:,j]) / np.sum(amp_XnaP)
        amp_XdP[:,j] = amp_XdsP[:,j] # to avoid confusion for the rest of the plots. not sure if this is right thing to do. 
    SNR[:,j] = 10*np.log10(SNR[:,j]) #convert snr to decibels
    if SNR[:,j] < -5:
        alpha[:,j] = alpha0+(3/4)
    elif SNR[:,j] >= -5 and SNR[:,j] < 20:
        alpha[:,j] = alpha0-(3/20*SNR[:,j])
    elif SNR[:,j] >= 20:
        alpha[:,j] = alpha0-3 
    amp_XpP4[:,j] = amp_XdP[:,j] - (alpha[:,j]*amp_XnaP)  
    amp_XpPnobeta4[:,j] = amp_XpP4[:,j] 
    foundlows = np.where(amp_XpP4[:,j] < beta*amp_XnaP)
    amp_XpP4[foundlows,j] = beta*amp_XnaP[foundlows]
    alpha0 = 5
    if ss_type == "over":
        SNR[:,j] = np.sum(amp_XdP[:,j]) / np.sum(amp_XnaP)
    elif ss_type == "smooth_over":
        amp_Xds[:,j] = (muy)*amp_Xd[:,(j-1)]+(1-muy)*amp_Xd[:,j] 
        amp_XdsP[:,j] = amp_Xds[:,j] ** p
        SNR[:,j] = np.sum(amp_XdsP[:,j]) / np.sum(amp_XnaP)
        amp_XdP[:,j] = amp_XdsP[:,j] # to avoid confusion for the rest of the plots. not sure if this is right thing to do. 
    SNR[:,j] = 10*np.log10(SNR[:,j]) #convert snr to decibels
    if SNR[:,j] < -5:
        alpha[:,j] = alpha0+(3/4)
    elif SNR[:,j] >= -5 and SNR[:,j] < 20:
        alpha[:,j] = alpha0-(3/20*SNR[:,j])
    elif SNR[:,j] >= 20:
        alpha[:,j] = alpha0-3 
    amp_XpP5[:,j] = amp_XdP[:,j] - (alpha[:,j]*amp_XnaP)  
    amp_XpPnobeta5[:,j] = amp_XpP5[:,j] 
    foundlows = np.where(amp_XpP5[:,j] < beta*amp_XnaP)
    amp_XpP5[foundlows,j] = beta*amp_XnaP[foundlows]    
    alpha0 = 6
    if ss_type == "over":
        SNR[:,j] = np.sum(amp_XdP[:,j]) / np.sum(amp_XnaP)
    elif ss_type == "smooth_over":
        amp_Xds[:,j] = (muy)*amp_Xd[:,(j-1)]+(1-muy)*amp_Xd[:,j] 
        amp_XdsP[:,j] = amp_Xds[:,j] ** p
        SNR[:,j] = np.sum(amp_XdsP[:,j]) / np.sum(amp_XnaP)
        amp_XdP[:,j] = amp_XdsP[:,j] # to avoid confusion for the rest of the plots. not sure if this is right thing to do. 
    SNR[:,j] = 10*np.log10(SNR[:,j]) #convert snr to decibels
    if SNR[:,j] < -5:
        alpha[:,j] = alpha0+(3/4)
    elif SNR[:,j] >= -5 and SNR[:,j] < 20:
        alpha[:,j] = alpha0-(3/20*SNR[:,j])
    elif SNR[:,j] >= 20:
        alpha[:,j] = alpha0-3 
    amp_XpP6[:,j] = amp_XdP[:,j] - (alpha[:,j]*amp_XnaP)  
    amp_XpPnobeta6[:,j] = amp_XpP6[:,j] 
    foundlows = np.where(amp_XpP6[:,j] < beta*amp_XnaP)
    amp_XpP6[foundlows,j] = beta*amp_XnaP[foundlows]    
    alpha0 = 7
    if ss_type == "over":
        SNR[:,j] = np.sum(amp_XdP[:,j]) / np.sum(amp_XnaP)
    elif ss_type == "smooth_over":
        amp_Xds[:,j] = (muy)*amp_Xd[:,(j-1)]+(1-muy)*amp_Xd[:,j] 
        amp_XdsP[:,j] = amp_Xds[:,j] ** p
        SNR[:,j] = np.sum(amp_XdsP[:,j]) / np.sum(amp_XnaP)
        amp_XdP[:,j] = amp_XdsP[:,j] # to avoid confusion for the rest of the plots. not sure if this is right thing to do. 
    SNR[:,j] = 10*np.log10(SNR[:,j]) #convert snr to decibels
    if SNR[:,j] < -5:
        alpha[:,j] = alpha0+(3/4)
    elif SNR[:,j] >= -5 and SNR[:,j] < 20:
        alpha[:,j] = alpha0-(3/20*SNR[:,j])
    elif SNR[:,j] >= 20:
        alpha[:,j] = alpha0-3 
    amp_XpP7[:,j] = amp_XdP[:,j] - (alpha[:,j]*amp_XnaP)  
    amp_XpPnobeta7[:,j] = amp_XpP7[:,j] 
    foundlows = np.where(amp_XpP7[:,j] < beta*amp_XnaP)
    amp_XpP7[foundlows,j] = beta*amp_XnaP[foundlows]
    alpha0 = 8
    if ss_type == "over":
        SNR[:,j] = np.sum(amp_XdP[:,j]) / np.sum(amp_XnaP)
    elif ss_type == "smooth_over":
        amp_Xds[:,j] = (muy)*amp_Xd[:,(j-1)]+(1-muy)*amp_Xd[:,j] 
        amp_XdsP[:,j] = amp_Xds[:,j] ** p
        SNR[:,j] = np.sum(amp_XdsP[:,j]) / np.sum(amp_XnaP)
        amp_XdP[:,j] = amp_XdsP[:,j] # to avoid confusion for the rest of the plots. not sure if this is right thing to do. 
    SNR[:,j] = 10*np.log10(SNR[:,j]) #convert snr to decibels
    if SNR[:,j] < -5:
        alpha[:,j] = alpha0+(3/4)
    elif SNR[:,j] >= -5 and SNR[:,j] < 20:
        alpha[:,j] = alpha0-(3/20*SNR[:,j])
    elif SNR[:,j] >= 20:
        alpha[:,j] = alpha0-3 
    amp_XpP8[:,j] = amp_XdP[:,j] - (alpha[:,j]*amp_XnaP)  
    amp_XpPnobeta8[:,j] = amp_XpP8[:,j] 
    foundlows = np.where(amp_XpP8[:,j] < beta*amp_XnaP)
    amp_XpP8[foundlows,j] = beta*amp_XnaP[foundlows]
    alpha0 = 9
    if ss_type == "over":
        SNR[:,j] = np.sum(amp_XdP[:,j]) / np.sum(amp_XnaP)
    elif ss_type == "smooth_over":
        amp_Xds[:,j] = (muy)*amp_Xd[:,(j-1)]+(1-muy)*amp_Xd[:,j] 
        amp_XdsP[:,j] = amp_Xds[:,j] ** p
        SNR[:,j] = np.sum(amp_XdsP[:,j]) / np.sum(amp_XnaP)
        amp_XdP[:,j] = amp_XdsP[:,j] # to avoid confusion for the rest of the plots. not sure if this is right thing to do. 
    SNR[:,j] = 10*np.log10(SNR[:,j]) #convert snr to decibels
    if SNR[:,j] < -5:
        alpha[:,j] = alpha0+(3/4)
    elif SNR[:,j] >= -5 and SNR[:,j] < 20:
        alpha[:,j] = alpha0-(3/20*SNR[:,j])
    elif SNR[:,j] >= 20:
        alpha[:,j] = alpha0-3 
    amp_XpP9[:,j] = amp_XdP[:,j] - (alpha[:,j]*amp_XnaP)  
    amp_XpPnobeta9[:,j] = amp_XpP9[:,j] 
    foundlows = np.where(amp_XpP9[:,j] < beta*amp_XnaP)
    amp_XpP9[foundlows,j] = beta*amp_XnaP[foundlows]
    #COMPARE 
    # 900C3F C70039 FF5733 FFC305
    ax2.plot(freqs_d,amp_XdP[:,j],       'k-',lw=1.25,label='Xd, or Xds if smooth')
#    ax2.plot(freqs_d,amp_XnaP,           'k--',lw=.75,label='Xna')
    ax2.plot(freqs_d,amp_XpP1[:,j],      '-',lw=.75,label='alpha0=1',color='navy')
#    ax2.plot(freqs_d,amp_XpP1[:,j],      '-',lw=.75,label='alpha0=1, beta=0.2',color='navy')
#    ax2.plot(freqs_d,amp_XpPnobeta2[:,j],':',lw=.75,label='alpha0=2, no beta',color='#C70039')
    ax2.plot(freqs_d,amp_XpP2[:,j],      '-',lw=.75,label='alpha0=2',color='darkblue')
#    ax2.plot(freqs_d,amp_XpPnobeta3[:,j],':',lw=.75,label='alpha0=3, no beta',color='#FF5733')
    ax2.plot(freqs_d,amp_XpP3[:,j],      '-',lw=.75,label='alpha0=3',color='indigo')
    ax2.plot(freqs_d,amp_XpP4[:,j],      '-',lw=.75,label='alpha0=4',color='blue')
    ax2.plot(freqs_d,amp_XpP5[:,j],      '-',lw=.75,label='alpha0=5',color='cyan')
    ax2.plot(freqs_d,amp_XpP6[:,j],      '-',lw=.75,label='alpha0=6',color='aqua')
    ax2.plot(freqs_d,amp_XpP7[:,j],      '-',lw=.75,label='alpha0=7',color='lightblue')
    ax2.plot(freqs_d,amp_XpP8[:,j],      '-',lw=.75,label='alpha0=8',color='turquoise')
    ax2.plot(freqs_d,amp_XpP9[:,j],      '-',lw=.75,label='alpha0=9',color='red')
    ax2.plot(freqs_d,amp_XpPnobeta9[:,j],':',lw=1.75,label='alpha0=9, no beta',color='red')
    ax2.plot(freqs_d,beta*amp_XnaP,      'k:',lw=1.75,label='(beta=' + str(beta) + ')*Xna')
    ax2.legend()
    ax2.set(xlim=[.5, 10])
    #ax2.set_title('timeframe' + str(j))
 
    #######################################################################################################################################
    # MAKE A PLOT COMPARING DIFFERENT BETAS
    #######################################################################################################################################
    # SELECT A SINGLE TIME FRAME TO TEST  
    j = timeframe   #   snr=9.1  alpha=2.6        alpha0-3/20*snr[7800]     
    alpha0 = 4
    if ss_type == "over":
        SNR[:,j] = np.sum(amp_XdP[:,j]) / np.sum(amp_XnaP)
    elif ss_type == "smooth_over":
        amp_Xds[:,j] = (muy)*amp_Xd[:,(j-1)]+(1-muy)*amp_Xd[:,j] 
        amp_XdsP[:,j] = amp_Xds[:,j] ** p
        SNR[:,j] = np.sum(amp_XdsP[:,j]) / np.sum(amp_XnaP)
        amp_XdP[:,j] = amp_XdsP[:,j] # to avoid confusion for the rest of the plots. not sure if this is right thing to do. 
    SNR[:,j] = 10*np.log10(SNR[:,j]) #convert snr to decibels
    if SNR[:,j] < -5:
        alpha[:,j] = alpha0+(3/4)
    elif SNR[:,j] >= -5 and SNR[:,j] < 20:
        alpha[:,j] = alpha0-(3/20*SNR[:,j])
    elif SNR[:,j] >= 20:
        alpha[:,j] = alpha0-3 
    beta = 0.3
    amp_XpP1[:,j] = amp_XdP[:,j] - (alpha[:,j]*amp_XnaP)  
    amp_XpPnobeta1[:,j] = amp_XpP1[:,j] 
    foundlows = np.where(amp_XpP1[:,j] < beta*amp_XnaP)
    amp_XpP1[foundlows,j] = beta*amp_XnaP[foundlows]
    beta = 0.9
    amp_XpP3[:,j] = amp_XdP[:,j] - (alpha[:,j]*amp_XnaP)  
    amp_XpPnobeta3[:,j] = amp_XpP3[:,j] 
    foundlows = np.where(amp_XpP3[:,j] < beta*amp_XnaP)
    amp_XpP3[foundlows,j] = beta*amp_XnaP[foundlows]
    beta = 0.6
    amp_XpP2[:,j] = amp_XdP[:,j] - (alpha[:,j]*amp_XnaP)  
    amp_XpPnobeta2[:,j] = amp_XpP2[:,j] 
    foundlows = np.where(amp_XpP2[:,j] < beta*amp_XnaP)
    amp_XpP2[foundlows,j] = beta*amp_XnaP[foundlows]
  #  beta = 2.0
  #  amp_XpP4[:,j] = amp_XdP[:,j] - (alpha[:,j]*amp_XnaP)  
  #  amp_XpPnobeta4[:,j] = amp_XpP4[:,j] 
  #  foundlows = np.where(amp_XpP4[:,j] < beta*amp_XnaP)
  #  amp_XpP4[foundlows,j] = beta*amp_XnaP[foundlows]
  #  alpha0 = 4
  #  beta = 3.0
  #  amp_XpP5[:,j] = amp_XdP[:,j] - (alpha[:,j]*amp_XnaP)  
  #  amp_XpPnobeta5[:,j] = amp_XpP5[:,j] 
  #  foundlows = np.where(amp_XpP5[:,j] < beta*amp_XnaP)
  #  amp_XpP5[foundlows,j] = beta*amp_XnaP[foundlows]
    #COMPARE 
    # 900C3F C70039 FF5733 FFC305
    ax3.plot(freqs_d,amp_XdP[:,j],       'k-',lw=.75,label='Xd, or Xds if smooth')
    ax3.plot(freqs_d,amp_XnaP,           'k--',lw=.75,label='Xna')
    ax3.plot(freqs_d,amp_XpP1[:,j],      '-',lw=.75,label='alpha0=4, beta=0.3',color='#900C3F')
    ax3.plot(freqs_d,amp_XpP2[:,j],      '-',lw=.75,label='alpha0=4, beta=0.6',color='#C70039')
    ax3.plot(freqs_d,amp_XpP3[:,j],      '-',lw=.75,label='alpha0=4, beta=0.9',color='#FF5733')
#    ax3.plot(freqs_d,amp_XpP4[:,j],      '-',lw=.75,label='alpha0=4, beta=2.0',color='#AC063C')
#    ax3.plot(freqs_d,amp_XpP5[:,j],      '-',lw=.75,label='alpha0=4, beta=3.0',color='#FFC305')
    ax3.plot(freqs_d,beta*amp_XnaP,      'k:',lw=.75,label='(beta=0.6)*Xna')
    ax3.legend()
    ax3.set(xlim=[.5, 10])
    ax3.set_xlabel('frequency (Hz)')
    fig.savefig(os.path.join(outpath, event_list + '_'+ station_list + '_' + figname + '.png'))
#
def sub_param_one_tf(amp_Xd, amp_Xn, freqs_d, outpath, fig, event_list, station_list, timeframe="100", figname="one_timeframe_alpha_beta"):
    m, n = amp_Xd.shape 
    p = 2
    amp_Xna = np.mean(amp_Xn,axis=1)
    amp_XdP = amp_Xd**p                     # appended P is shorthand for **p
    amp_XnaP = amp_Xna**p
    
    # EXPLORE SPECTRAL SUBTRACTION PARAMETERS
    #alpha0=4 #vary between 3-6 (Beruiti et al'79), normally taken as 4 (Kamath & Loizou'02)
    #beta=0.2 #should be between 0-1.
    amp_XpP = np.zeros((m,n))
    amp_XpP1 = np.zeros((m,n))
    amp_XpP2 = np.zeros((m,n))
    amp_XpP3 = np.zeros((m,n))
    amp_XpP4 = np.zeros((m,n))
    amp_XpP5 = np.zeros((m,n))
    amp_XpPnobeta1 = np.zeros((m,n))
    amp_XpPnobeta2 = np.zeros((m,n))
    amp_XpPnobeta3 = np.zeros((m,n))
    amp_XpPnobeta4 = np.zeros((m,n))
    amp_XpPnobeta5 = np.zeros((m,n))
    
    #######################################################################################################################################
    # PLOT ROLE OF ALPHA AND BETA
    #######################################################################################################################################    
    # SELECT A SINGLE TIME FRAME TO TEST  
    #j = 7000   #   snr=0.6  alpha=3.9        alpha0-3/20*snr[7000]
    #j = 7500   #   snr=5.2  alpha=3.2        alpha0-3/20*snr[7500]
    j = timeframe   #   snr=9.1  alpha=2.6        alpha0-3/20*snr[7800]     
    #j = 7800   #   snr=9.1  alpha=2.6        alpha0-3/20*snr[7800]  
    beta = .3
    alpha = 2.3
    amp_XpP2[:,j] = amp_XdP[:,j] - alpha*amp_XnaP  
    amp_XpPnobeta2[:,j] = amp_XpP2[:,j] 
    foundlows = np.where(amp_XpP2[:,j] < beta*amp_XnaP)
    amp_XpP2[foundlows,j] = beta*amp_XnaP[foundlows]
    #COMPARE 
    # 900C3F C70039 FF5733 FFC305
    fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize=(7.5,10))
    ax1.plot(freqs_d,amp_XdP[:,j],       'k-',lw=.75,label='Xd')
    ax1.plot(freqs_d,amp_XnaP,           'k--',lw=.75,label='Xna')
    ax1.plot(freqs_d,amp_XpPnobeta2[:,j],':',lw=.75,label='alpha=2.3, no beta',color='#C70039')
    ax1.plot(freqs_d,amp_XpP2[:,j],      '-',lw=.75,label='alpha=2.3, beta=0.3',color='#C70039')
    ax1.plot(freqs_d,beta*amp_XnaP,      'k:',lw=.75,label='beta*Xna')
    ax1.legend()
    ax1.set(xlim=[.5, 10])
    ax1.set_title('timeframe ' + str(j))

    #######################################################################################################################################
    # MAKE A PLOT COMPARING DIFFERENT ALPHAS
    #######################################################################################################################################    
    # SELECT A SINGLE TIME FRAME TO TEST  
    #j = 7000   #   snr=0.6  alpha=3.9        alpha0-3/20*snr[7000]
    #j = 7500   #   snr=5.2  alpha=3.2        alpha0-3/20*snr[7500]
    j = timeframe   #   snr=9.1  alpha=2.6        alpha0-3/20*snr[7800]     
    #j = 7800   #   snr=9.1  alpha=2.6        alpha0-3/20*snr[7800]     
    alpha = 1
    amp_XpP1[:,j] = amp_XdP[:,j] - (alpha*amp_XnaP)  
    amp_XpPnobeta1[:,j] = amp_XpP1[:,j] 
    foundlows = np.where(amp_XpP1[:,j] < beta*amp_XnaP)
    amp_XpP1[foundlows,j] = beta*amp_XnaP[foundlows]
    alpha = 2
    amp_XpP2[:,j] = amp_XdP[:,j] - (alpha*amp_XnaP)  
    amp_XpPnobeta2[:,j] = amp_XpP2[:,j] 
    foundlows = np.where(amp_XpP2[:,j] < beta*amp_XnaP)
    amp_XpP2[foundlows,j] = beta*amp_XnaP[foundlows]
    alpha = 3
    amp_XpP3[:,j] = amp_XdP[:,j] - (alpha*amp_XnaP)  
    amp_XpPnobeta3[:,j] = amp_XpP3[:,j] 
    foundlows = np.where(amp_XpP3[:,j] < beta*amp_XnaP)
    amp_XpP3[foundlows,j] = beta*amp_XnaP[foundlows]
    #COMPARE 
    # 900C3F C70039 FF5733 FFC305
    ax2.plot(freqs_d,amp_XdP[:,j],       'k-',lw=.75,label='Xd')
    ax2.plot(freqs_d,amp_XnaP,           'k--',lw=.75,label='Xna')
    ax2.plot(freqs_d,amp_XpPnobeta1[:,j],':',lw=.75,label='alpha=1, no beta',color='#900C3F')
    ax2.plot(freqs_d,amp_XpP1[:,j],      '-',lw=.75,label='alpha=1, beta=0.2',color='#900C3F')
    ax2.plot(freqs_d,amp_XpPnobeta2[:,j],':',lw=.75,label='alpha=2, no beta',color='#C70039')
    ax2.plot(freqs_d,amp_XpP2[:,j],      '-',lw=.75,label='alpha=2, beta=0.2',color='#C70039')
    ax2.plot(freqs_d,amp_XpPnobeta3[:,j],':',lw=.75,label='alpha=3, no beta',color='#FF5733')
    ax2.plot(freqs_d,amp_XpP3[:,j],      '-',lw=.75,label='alpha=3, beta=0.2',color='#FF5733')
    ax2.plot(freqs_d,beta*amp_XnaP,      'k:',lw=.75,label='beta*Xna')
    ax2.legend()
    ax2.set(xlim=[.5, 10])
    #ax2.set_title('timeframe' + str(j))
 
    #######################################################################################################################################
    # MAKE A PLOT COMPARING DIFFERENT BETAS
    #######################################################################################################################################
    # SELECT A SINGLE TIME FRAME TO TEST  
    j = timeframe   #   snr=9.1  alpha=2.6        alpha0-3/20*snr[7800]     
    alpha = 3
    beta = 0.3
    amp_XpP1[:,j] = amp_XdP[:,j] - (alpha*amp_XnaP)  
    amp_XpPnobeta1[:,j] = amp_XpP1[:,j] 
    foundlows = np.where(amp_XpP1[:,j] < beta*amp_XnaP)
    amp_XpP1[foundlows,j] = beta*amp_XnaP[foundlows]
    alpha = 3
    beta = 0.6
    amp_XpP2[:,j] = amp_XdP[:,j] - (alpha*amp_XnaP)  
    amp_XpPnobeta2[:,j] = amp_XpP2[:,j] 
    foundlows = np.where(amp_XpP2[:,j] < beta*amp_XnaP)
    amp_XpP2[foundlows,j] = beta*amp_XnaP[foundlows]
    alpha = 3
    beta = 0.9
    amp_XpP3[:,j] = amp_XdP[:,j] - (alpha*amp_XnaP)  
    amp_XpPnobeta3[:,j] = amp_XpP3[:,j] 
    foundlows = np.where(amp_XpP3[:,j] < beta*amp_XnaP)
    amp_XpP3[foundlows,j] = beta*amp_XnaP[foundlows]
    alpha = 3
    beta = 2.0
    amp_XpP4[:,j] = amp_XdP[:,j] - (alpha*amp_XnaP)  
    amp_XpPnobeta4[:,j] = amp_XpP4[:,j] 
    foundlows = np.where(amp_XpP4[:,j] < beta*amp_XnaP)
    amp_XpP4[foundlows,j] = beta*amp_XnaP[foundlows]
    alpha = 3
    beta = 3.0
    amp_XpP5[:,j] = amp_XdP[:,j] - (alpha*amp_XnaP)  
    amp_XpPnobeta5[:,j] = amp_XpP5[:,j] 
    foundlows = np.where(amp_XpP5[:,j] < beta*amp_XnaP)
    amp_XpP5[foundlows,j] = beta*amp_XnaP[foundlows]
    #COMPARE 
    # 900C3F C70039 FF5733 FFC305
    ax3.plot(freqs_d,amp_XdP[:,j],       'k-',lw=.75,label='Xd')
    ax3.plot(freqs_d,amp_XnaP,           'k--',lw=.75,label='Xna')
    ax3.plot(freqs_d,amp_XpP1[:,j],      '-',lw=.75,label='alpha=3, beta=0.3',color='#900C3F')
    ax3.plot(freqs_d,amp_XpP2[:,j],      '-',lw=.75,label='alpha=3, beta=0.6',color='#C70039')
    ax3.plot(freqs_d,amp_XpP3[:,j],      '-',lw=.75,label='alpha=3, beta=0.9',color='#FF5733')
    ax3.plot(freqs_d,amp_XpP4[:,j],      '-',lw=.75,label='alpha=3, beta=2.0',color='#AC063C')
    ax3.plot(freqs_d,amp_XpP5[:,j],      '-',lw=.75,label='alpha=3, beta=3.0',color='#FFC305')
    ax3.plot(freqs_d,beta*amp_XnaP,      'k:',lw=.75,label='beta*Xna')
    ax3.legend()
    ax3.set(xlim=[.5, 10])
    ax3.set_xlabel('frequency (Hz)')
    fig.savefig(os.path.join(outpath, event_list + '_'+ station_list + '_' + figname + '.png'))

def processed_signal_tf(amp_Xd, amp_Xn, freqs_d, outpath, fig, event_list, station_list, x1=1485, now1=11, step1=1, x3=1390, now2=11, step2=20, figname="cons_timeframe_processed_signal"):
    #x1 start of interval for first fig.
    #now1 number of windows  for the first fig.
    #step1 step size for the first fig.
    #x2 start of interval for second fig.
    #now2 number of windows  for the second fig.
    #step2 step size for the second fig.

    m, n = amp_Xd.shape 
    p = 2
    amp_Xna = np.mean(amp_Xn,axis=1)
    amp_XdP = amp_Xd**p                     # appended P is shorthand for **p
    amp_XnaP = amp_Xna**p
    
    # EXPLORE SPECTRAL SUBTRACTION PARAMETERS
    amp_XpP = np.zeros((m,n))
    amp_XpP1 = np.zeros((m,n))
    amp_XpP2 = np.zeros((m,n))
    amp_XpP3 = np.zeros((m,n))
    amp_XpPnobeta1 = np.zeros((m,n))
    amp_XpPnobeta2 = np.zeros((m,n))
    amp_XpPnobeta3 = np.zeros((m,n))
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(7.5,10))
 
    #############################################
    # COMPARE TIMEFRAME VARIABILITY
    #############################################
    x2=x1+(step1*now1) #end of interval
    tfw1 = np.arange(x1,x2,step1)  #timeframe window 
    for i in range(now1):
        ax1.plot(freqs_d, amp_XdP[:,tfw1[i]], '-', lw=.75, color=np.random.rand(3,))
    ax1.set(xlim=[.5, 10])
    ax1.set_title(str(now1) + ' consecutive timeframes [' + str(x1) + ':' + str(step1) + ':' +  str(x2) + ']')
    
    #######################################################################################################################################
    # COMPARE VARIABILITY EACH SECOND
    #######################################################################################################################################
    x4=x3+(step2*now2) #end of interval
    tfw2 = np.arange(x3,x4,step2)  #timeframe window 
    for j in range(now2):
        ax2.plot(freqs_d, amp_XdP[:,tfw2[j]], '-', lw=.75, color=np.random.rand(3,))
    ax2.set(xlim=[.5, 10])
    ax2.set_title(str(now2) + ' consecutive timeframes [' + str(x3) + ':' + str(step2) + ':' +  str(x4) + ']')
    ax2.set_xlabel('frequency (Hz)')
    fig.savefig(os.path.join(outpath, event_list + '_'+ station_list + '_' + figname + '.png'))

#Compare the effects of different alpha values on waveforms

def alpha_comp_wfs(t, tro, trd, amp_Xd, amp_Xn, phase_Xd, scales_d, omega0, dj, outpath, fig, event_list, station_list, figname="alpha_comparison_wfs"):
    st_all = Stream(traces=[tro, trd]) 
    alpha0list = [1, 2, 4, 6, 10]
    dt = tro.stats.delta
    mother = wavelet.Morlet(omega0)              # required for pycwt
    for i in range(len(alpha0list)):
        # DO SUBTRACTION
        beta = 0.0
        alpha0 = alpha0list[i]
        amp_Xp, SNR, alpha = ss.constant_subtraction(amp_Xd,amp_Xn,2,alpha0,beta)
        Xp = amp_Xp * np.exp(1.j*phase_Xd)
        trp = trd.copy()
        trp.data = wavelet.icwt(Xp, scales_d, dt, dj, mother)
        #trp.data = mlwt.icwt(Xp, trd) 
        # PLOT WAVEFORMS
        trp.stats.location = 'alpha'+str(alpha0)
        st_all += trp
    st_all.plot(automerge=False,equal_scale=True,linewidth=1,fig=fig)
    fig.savefig(os.path.join(outpath, event_list + '_'+ station_list + '_' + figname + '.png'))
    #fig.autofmt_xdate()    

#Compare the effects of different alpha values on scaleograms
def alpha_comp_scals(t, tro, trd, amp_Xo, amp_Xd, amp_Xn, phase_Xd, scales_d, freqs_d, omega0, dj, outpath, fig, event_list, station_list, figname="alpha_comparison_scals"):
    alpha0list = [1, 2, 4, 6, 10]
    nop= len(alpha0list)+2 #number of subplots
    dt = tro.stats.delta
    mother = wavelet.Morlet(omega0)              # required for pycwt  
    fig = plt.figure(figsize=(7.5,10))
    ax1 = fig.add_subplot((nop),1,1)
    maxamp = abs(amp_Xd).max()/2           # these two lines just adjust the color scale
    minamp = 0
    t, f = np.meshgrid(trd.times(), freqs_d)
    im1 = ax1.pcolormesh(t, f, np.abs(amp_Xo), cmap=cm.hot, vmin=minamp, vmax=maxamp)
    ax1.text(1.5, 8.5, 'original CWT amplitude', fontsize=8, bbox=dict(facecolor='white'))
    ax1.axes.xaxis.set_visible(False)
    ax2 = fig.add_subplot((nop),1,2)
    im2 = ax2.pcolormesh(t, f, np.abs(amp_Xd), cmap=cm.hot, vmin=minamp, vmax=maxamp)
    ax2.text(1.5, 8.5, 'degraded CWT amplitude', fontsize=8, bbox=dict(facecolor='white'))
    ax2.axes.xaxis.set_visible(False)
    ax2.set_ylabel('frequency (Hz)')
    for i in range(len(alpha0list)):
        ax3 = fig.add_subplot(nop, 1, (i+3))
        # DO SUBTRACTION
        beta = 0.0
        alpha0 = alpha0list[i]
        amp_Xp, SNR, alpha = ss.constant_subtraction(amp_Xd,amp_Xn,2,alpha0,beta)
        im = ax3.pcolormesh(t, f, np.abs(amp_Xp), cmap=cm.hot, vmin=minamp, vmax=maxamp)
        #ax1.set_ylabel('frequency (Hz)')
        text = 'alpha'+str(alpha0)
        ax3.text(1.5, 8.5, text, fontsize=8, bbox=dict(facecolor='white'))
        ax3.axes.xaxis.set_visible(False)
    ax3.axes.xaxis.set_visible(True)
    fig.savefig(os.path.join(outpath, event_list + '_'+ station_list + '_' + figname + '.png'))

##################################################################################
##################################################################################
##################################################################################
######################THESE PLOTS ARE NOT USED ANYMORE############################
##################################################################################
##################################################################################
##################################################################################
#very first plot to plot all scaleograms, not used anymore
def all(t, tr, X, freq, IX, outpath, fig, event_list, station_list, figname="original"): 
    maxamp = abs(X).max()/2           # these two lines just adjust the color scale
    minamp = 0
    tX, f = np.meshgrid(tr.times(), freq)
    ax11 = fig.add_axes([0.1, 0.75, 0.7, 0.2])
    ax12 = fig.add_axes([0.1, 0.1, 0.7, 0.60], sharex=ax11)
    ax13 = fig.add_axes([0.83, 0.1, 0.03, 0.6])
    ax14 = fig.add_axes([0.1, -0.15, 0.7, 0.2], sharex=ax11)
    ax11.plot(t, tr.data, 'k', linewidth=0.3)
    img = ax12.pcolormesh(tX, f, np.abs(X), cmap=obspy_sequential, vmin=minamp, vmax=maxamp)
    ax12.set_ylabel("Frequency [Hz]")
    ax12.set_xlim(t[0], t[-1])
    ax12.set_ylim(freq[-1], freq[0])
    ax14.plot(t, IX, 'k', linewidth=0.3)
    ax14.set_xlabel("Time after %s [s]" % tr.stats.starttime)
    fig.colorbar(img, cax=ax13)
    fig.autofmt_xdate()    
    fig.savefig(os.path.join(outpath, event_list + '_'+ station_list + '_' + figname + '.png'))

#very first plot to plot all scaleograms, not used anymore, in log plot

def all_log(t, tr, X, freq, IX, outpath, fig, event_list, station_list, figname="original"): 
    ax11 = fig.add_axes([0.1, 0.75, 0.7, 0.2])
    ax12 = fig.add_axes([0.1, 0.1, 0.7, 0.60], sharex=ax11)
    ax13 = fig.add_axes([0.83, 0.1, 0.03, 0.6])
    ax14 = fig.add_axes([0.1, -0.15, 0.7, 0.2], sharex=ax11)
    ax11.plot(t, tr.data, 'k', linewidth=0.3)
    img = ax12.imshow(np.abs(X), extent=[t[0], t[-1], freq[-1], freq[0]],
                     aspect='auto', interpolation='nearest', cmap=obspy_sequential)
    # Hackish way to overlay a logarithmic scale over a linearly scaled image.
    ax12.set_ylabel("Frequency [Hz] (log)")
    twin_ax = ax12.twinx()
    twin_ax.set_yscale('log')
    twin_ax.set_xlim(t[0], t[-1])
    twin_ax.set_ylim(freq[-1], freq[0])
    ax12.tick_params(which='both', labelleft=False, left=False)
    twin_ax.tick_params(which='both', labelleft=True, left=True, labelright=False)
    ax14.plot(t, IX, 'k', linewidth=0.3)
    ax14.set_xlabel("Time after %s [s]" % tr.stats.starttime)
    fig.colorbar(img, cax=ax13)
    fig.autofmt_xdate()    
    fig.savefig(os.path.join(outpath, event_list + '_'+ station_list + '_' + figname + '.pdf'))
    #fig.savefig('xx.pdf', bbox_inches='tight')

#def 
#    # PLOT THE FORIER TRANSFORM OF THE FULL TRD AND TRN TRACES
#    fig, ax = plt.subplots()
#    line1=ax.plot(fftfreqs_d,np.abs(fft_d), 'r-', label='degraded', lw=.5)
#    line2=ax.plot(fftfreqs_n,np.abs(fft_n), 'k-', label='noise', lw=.5)
#    ax.set_xscale('log')
#    ax.set_yscale('log')
#    ax.set_xlim([0.1, 10])
#    ax.legend()
#    plt.savefig('TMP_ffts.png')


#SPECTRA COMPARISON PLOT, COMPARES THE SPECTRA OF ORIGINAL, PROCESSED, NOISE AND DEGRADED SIGNALS
def spectra(amp_Xo, amp_Xd, amp_Xn, amp_Xp, freqs_d, outpath, fig, event_list, station_list, figname="spectra_comparison"): 
    # PLOT SAMPLE COLUMNS FROM THE DEGRADED AND PROCESSED CWT
    mean_amp_Xo = np.mean(amp_Xo,axis=1)
    mean_amp_Xn = np.mean(amp_Xn,axis=1)
    mean_amp_Xd = np.mean(amp_Xd,axis=1)  
    mean_amp_Xp = np.mean(amp_Xp,axis=1) 
    #mean_amp_r = np.mean(amp_r,axis=1) 
    fig, ax = plt.subplots()
    #for i in range(1600, 1600):
    #    ax.plot(freqs_d,amp_d[:,i], 'k-', lw=.25)
    #    ax.plot(freqs_d,amp_p[:,i], 'y-', lw=.25)
    #    #ax.plot(freqs_d,amp_r[:,i], 'b-', lw=.25)
    ax.plot(freqs_d,mean_amp_Xo, 'b-', lw=2,label='original')
    ax.plot(freqs_d,mean_amp_Xd, 'k-', lw=2,label='degraded')
    ax.plot(freqs_d,mean_amp_Xp, 'g-', lw=2,label='processed')
    #ax.plot(freqs_d,mean_amp_r, 'b-', lw=2,label='residual')
    ax.plot(freqs_d,mean_amp_Xn, 'r-', lw=2,label='noise')
    fig.savefig(os.path.join(outpath, event_list + '_'+ station_list + '_' + figname + '.pdf'))
    #plt.savefig('TMP_spectra_comparison.png')

#SCALES VS FREQUENCY FOR CWT
def scales_freq(freqs_d,scales_d, outpath, fig, event_list, station_list, figname="frequency_scale"): 
    # MAKE A PLOT OF SCALES VS FREQUENCY
    fig, ([ax1, ax2]) = plt.subplots(2,1)
    h1 = ax1.loglog(freqs_d,scales_d) 
    ax1.set_xlabel('frequency')
    ax1.set_ylabel('scale')
    plt.setp(h1, color='k', marker='o', markerfacecolor='r', markeredgecolor='k' )
    h2 = ax2.plot(freqs_d,scales_d) 
    ax2.set_xlabel('frequency')
    ax2.set_ylabel('scale')
    plt.setp(h2, color='k', marker='o', markerfacecolor='r', markeredgecolor='k' )
    fig.savefig(os.path.join(outpath, event_list + '_'+ station_list + '_' + figname + '.pdf'))
    #plt.savefig('TMP_frequency_vs_scale.png')    
    
