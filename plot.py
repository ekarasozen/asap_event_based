import numpy as np
import matplotlib.pyplot as plt
from obspy import Stream
from obspy.imaging.cm import obspy_sequential
    
def all(t, tr, X, freq, IX, fig, event_list, figname="original"): 
    ax11 = fig.add_axes([0.1, 0.75, 0.7, 0.2])
    ax12 = fig.add_axes([0.1, 0.1, 0.7, 0.60], sharex=ax11)
    ax13 = fig.add_axes([0.83, 0.1, 0.03, 0.6])
    ax14 = fig.add_axes([0.1, -0.15, 0.7, 0.2], sharex=ax11)
    ax11.plot(t, tr.data, 'k')
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
    ax14.plot(t, IX, 'k')
    ax14.set_xlabel("Time after %s [s]" % tr.stats.starttime)
    fig.colorbar(img, cax=ax13)
    fig.autofmt_xdate()    
    fig.savefig(event_list + '_'+ figname+ '.pdf', bbox_inches='tight')
    #fig.savefig('xx.pdf', bbox_inches='tight')
