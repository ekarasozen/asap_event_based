import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from obspy import Stream
from obspy.imaging.cm import obspy_sequential
import sys
sys.path.append("/Users/ezgikarasozen/Documents/Research/Array_processing/asap/")
import spectral_subtraction as ss
import pycwt as wavelet  ###pycwt
import mlwt ###mlpy

    
def wfs(t, tro, trd, trp, fig, event_list, figname="wfs"): 
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
    fig.savefig(event_list + '_'+ figname + '.png', bbox_inches='tight')

def scals(t, tr, Xo, Xd, Xp, freq, fig, event_list, figname="scals"): 
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
    fig.savefig(event_list + '_'+ figname + '.png', bbox_inches='tight')



def sub_param_one_tf(amp_Xd, amp_Xn, freqs_d, fig, event_list, timeframe="100", figname="one_timeframe_alpha_beta"):
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
    fig.savefig(event_list + '_'+ figname + '.png', bbox_inches='tight')

def processed_signal_tf(amp_Xd, amp_Xn, freqs_d, fig, event_list, x1=1485, now1=11, step1=1, x3=1390, now2=11, step2=20, figname="cons_timeframe_processed_signal"):
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
    fig.savefig(event_list + '_'+ figname + '.png', bbox_inches='tight')
    
 
def spectra(amp_o, amp_d, amp_n, amp_p, freqs_d, fig, event_list, figname="spectra_comparison"): 
    # PLOT SAMPLE COLUMNS FROM THE DEGRADED AND PROCESSED CWT
    mean_amp_o = np.mean(amp_o,axis=1)
    mean_amp_n = np.mean(amp_n,axis=1)
    mean_amp_d = np.mean(amp_d,axis=1)  
    mean_amp_p = np.mean(amp_p,axis=1) 
    #mean_amp_r = np.mean(amp_r,axis=1) 
    fig, ax = plt.subplots()
    #for i in range(1600, 1600):
    #    ax.plot(freqs_d,amp_d[:,i], 'k-', lw=.25)
    #    ax.plot(freqs_d,amp_p[:,i], 'y-', lw=.25)
    #    #ax.plot(freqs_d,amp_r[:,i], 'b-', lw=.25)
    ax.plot(freqs_d,mean_amp_o, 'b-', lw=2,label='original')
    ax.plot(freqs_d,mean_amp_d, 'k-', lw=2,label='degraded')
    ax.plot(freqs_d,mean_amp_p, 'g-', lw=2,label='processed')
    #ax.plot(freqs_d,mean_amp_r, 'b-', lw=2,label='residual')
    ax.plot(freqs_d,mean_amp_n, 'r-', lw=2,label='noise')
    fig.savefig(event_list + '_'+ figname + '.pdf', bbox_inches='tight')
    #plt.savefig('TMP_spectra_comparison.png')


def scales_freq(freqs_d,scales_d, fig, event_list, figname="frequency_scale"): 
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
    fig.savefig(event_list + '_'+ figname + '.pdf', bbox_inches='tight')
    #plt.savefig('TMP_frequency_vs_scale.png')

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



def subtraction_performance(amp_Xd,amp_Xp,freqs_d,picktime,tro,trd,trp,tr_SNR,tr_alpha,metrics,alpha0,beta,fig,event_list,figname="subtraction_parameters"):
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
    
    alpha_beta_text = ' alpha0: ' + str(alpha0) + '    beta: ' + str(round(beta)) 
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
    ax1.set_title('degraded CWT amplitude')
    ax1.set_xlim(windowstart, windowend)

    im2 = ax2.plot(t[1,:], tr_alpha.data, 'r--', linewidth=1,label='alpha')
    im2 = ax2.plot(t[1,:], tr_SNR.data/10, 'r-', linewidth=1,label='SNR')
    ax2.set_ylabel('0.1*SNR(db) & alpha')
    ax2.set_yticks([-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7])
    ax2.grid()
    ax2.set_ylim(0.1* tr_SNR.data.min(), 1.05*tr_alpha.data.max())
    ax2.legend(loc='upper left')
    ax2.text(windowstart, -1,alpha_beta_text, style='italic', fontsize=8)
    ax2.axes.xaxis.set_visible(False)
    ax2.set_title('subtraction parameters')
    ax2.set_xlim(windowstart, windowend)

    im3 = ax3.pcolormesh(t, f, np.abs(amp_Xp), cmap=cm.hot, vmin=minamp, vmax=maxamp)
    ax3.set_ylabel('frequency (Hz)')
    ax3.axes.xaxis.set_visible(False)
    ax3.set_title('processed CWT amplitude')
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
    fig.savefig(event_list + '_'+ figname + '.png', bbox_inches='tight')
    return 
    
def alpha_comp_wfs(t, tro, trd, amp_Xd, amp_Xn, phase_Xd, scales_d, omega0, dj, fig, event_list, figname="alpha_comparison_wfs"):
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
    plt.savefig(event_list + '_'+ figname + '.png', bbox_inches='tight')
    #fig.autofmt_xdate()    

def alpha_comp_scals(t, tro, trd, amp_Xo, amp_Xd, amp_Xn, phase_Xd, scales_d, freqs_d, omega0, dj, fig, event_list, figname="alpha_comparison_scals"):
    alpha0list = [1, 2, 4, 6, 10]
    nop= len(alpha0list)+2 #number of subplots
    dt = tro.stats.delta
    mother = wavelet.Morlet(omega0)              # required for pycwt  
    fig1 = plt.figure(figsize=(7.5,10))
    ax1 = fig1.add_subplot((nop),1,1)
    maxamp = abs(amp_Xd).max()/2           # these two lines just adjust the color scale
    minamp = 0
    t, f = np.meshgrid(trd.times(), freqs_d)
    im1 = ax1.pcolormesh(t, f, np.abs(amp_Xo), cmap=cm.hot, vmin=minamp, vmax=maxamp)
    ax1.text(1.5, 8.5, 'original CWT amplitude', fontsize=8, bbox=dict(facecolor='white'))
    ax1.axes.xaxis.set_visible(False)
    ax2 = fig1.add_subplot((nop),1,2)
    im2 = ax2.pcolormesh(t, f, np.abs(amp_Xd), cmap=cm.hot, vmin=minamp, vmax=maxamp)
    ax2.text(1.5, 8.5, 'degraded CWT amplitude', fontsize=8, bbox=dict(facecolor='white'))
    ax2.axes.xaxis.set_visible(False)
    ax2.set_ylabel('frequency (Hz)')
    for i in range(len(alpha0list)):
        ax3 = fig1.add_subplot(nop, 1, (i+3))
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
    fig1.savefig(event_list + '_'+ figname + '.png', bbox_inches='tight')


def all(t, tr, X, freq, IX, fig, event_list, figname="original"): 
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
    fig.savefig(event_list + '_'+ figname + '.png', bbox_inches='tight')

def all_log(t, tr, X, freq, IX, fig, event_list, figname="original"): 
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
    fig.savefig(event_list + '_'+ figname + '.pdf', bbox_inches='tight')
    #fig.savefig('xx.pdf', bbox_inches='tight')
