import numpy as np
import matplotlib.pyplot as plt
from obspy import Stream
from obspy.imaging.cm import obspy_sequential
    
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
