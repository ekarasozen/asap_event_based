import sys
sys.path.append("/Users/ezgikarasozen/Documents/Research/Array_processing/asap_w_obspy_git/")
from getdata import event
from getdata import taup
from prepdata import prep
from addnoise import whitenoise
import wavelet
import matplotlib.pyplot as plt
from obspy import Trace
from obspy import Stream
from obspy.imaging.cm import obspy_sequential
import numpy as np
import mlpy.wavelet as wave
import json

file = open('parameters.json')
inputs = json.loads(file.read())
event_list = inputs['event_list']


#event_list=['ak0184nb2fzw','ak014cou8d9l']
#event_list=['ak0184nb2fzw']
event_id = ['https://earthquake.usgs.gov/fdsnws/event/1/query?eventid=' + s for s in event_list]
for e, lab in enumerate(event_id):
    st = event(ev_id=event_id[e], network_list=['IM','XM'], station_code="BC", pick="P", channel="SHZ", start_time=20, end_time=60)
    st = prep(st,filter_type="bandpass", freqmin=1, freqmax=3)
    #DATA WITHOUT NOISE
    scales = wavelet.scales(st[0])
    t, freq = wavelet.param(st[0],scales)
    X = wavelet.cwt(st[0],scales)
    fig1 = plt.figure()
    ax11 = fig1.add_axes([0.1, 0.75, 0.7, 0.2])
    ax12 = fig1.add_axes([0.1, 0.1, 0.7, 0.60], sharex=ax11)
    ax13 = fig1.add_axes([0.83, 0.1, 0.03, 0.6])
    ax14 = fig1.add_axes([0.1, -0.15, 0.7, 0.2], sharex=ax11)
    ax11.plot(t, st[0].data, 'k')
    img = ax12.imshow(np.abs(X), extent=[t[0], t[-1], freq[-1], freq[0]],
                     aspect='auto', interpolation='nearest', cmap=obspy_sequential)
    # Hackish way to overlay a logarithmic scale over a linearly scaled image.
    ax12.set_ylabel("Frequency [Hz] (log)",labelpad=30)
    twin_ax = ax12.twinx()
    twin_ax.set_yscale('log')
    twin_ax.set_xlim(t[0], t[-1])
    twin_ax.set_ylim(1, 50)
    ax12.tick_params(which='both', labelleft=False, left=False)
    twin_ax.tick_params(which='both', labelleft=True, left=True, labelright=False)
    IX = wavelet.icwt(X, st[0])
    ax14.plot(t, IX, 'k')
    ax14.set_xlabel("Time after %s [s]" % st[0].stats.starttime)
    fig1.colorbar(img, cax=ax13)
    fig1.autofmt_xdate()    
    fig1.savefig(event_list[e] + '.pdf', bbox_inches='tight')
    #DATA WITH NOISE
    nos = len(st)
    st_wn = st.copy()
    for s in range(nos):
        st_wn[s].data = whitenoise(st_wn[s],type=2,amplitude=200,min_freq=0.1,max_freq=0.4)
        #st_wn[s].data = whitenoise(st_wn[s],type=1,amplitude=20)
    print(st[0].std())
    scales = wavelet.scales(st_wn[0])
    t, freq = wavelet.param(st_wn[0],scales)
    X = wavelet.cwt(st_wn[0],scales)
    fig2 = plt.figure()
    ax21 = fig2.add_axes([0.1, 0.75, 0.7, 0.2])
    ax22 = fig2.add_axes([0.1, 0.1, 0.7, 0.60], sharex=ax21)
    ax23 = fig2.add_axes([0.83, 0.1, 0.03, 0.6])
    ax24 = fig2.add_axes([0.1, -0.15, 0.7, 0.2], sharex=ax21)
    ax21.plot(t, st_wn[0].data, 'k')
    img = ax22.imshow(np.abs(X), extent=[t[0], t[-1], freq[-1], freq[0]],
                     aspect='auto', interpolation='nearest', cmap=obspy_sequential)
    # Hackish way to overlay a logarithmic scale over a linearly scaled image.
    ax22.set_ylabel("Frequency [Hz] (log)",labelpad=30)
    twin_ax = ax22.twinx()
    twin_ax.set_yscale('log')
    twin_ax.set_xlim(t[0], t[-1])
    twin_ax.set_ylim(1, 50)
    ax22.tick_params(which='both', labelleft=False, left=False)
    twin_ax.tick_params(which='both', labelleft=True, left=True, labelright=False)
    IX = wavelet.icwt(X, st_wn[0])
    ax24.plot(t, IX, 'k')
    ax24.set_xlabel("Time after %s [s]" % st_wn[0].stats.starttime)
    fig2.colorbar(img, cax=ax23)
    fig2.autofmt_xdate()    
    fig2.savefig(event_list[e] + '_wn.pdf', bbox_inches='tight')
    #NOISE REMOVED - THIS PART WILL HAVE ITS OWN MODULE AS "SPECTRAL SUBTRACTION METHODS"
    nos = len(st)
    st_won = st_wn.copy()
    scales = wavelet.scales(st_won[0])
    t, freq = wavelet.param(st_won[0],scales)
    idx = np.where(np.logical_and(freq>=0.1, freq<=0.4))
    X[idx] = 0
    X = (X).real
#    X = wavelet.cwt(st_won[0],scales)
    fig3 = plt.figure()
    ax31 = fig3.add_axes([0.1, 0.75, 0.7, 0.2])
    ax32 = fig3.add_axes([0.1, 0.1, 0.7, 0.60], sharex=ax31)
    ax33 = fig3.add_axes([0.83, 0.1, 0.03, 0.6])
    ax34 = fig3.add_axes([0.1, -0.15, 0.7, 0.2], sharex=ax31)
    ax31.plot(t, st_won[0].data, 'k')
    img = ax32.imshow(np.abs(X), extent=[t[0], t[-1], freq[-1], freq[0]],
                     aspect='auto', interpolation='nearest', cmap=obspy_sequential)
    # Hackish way to overlay a logarithmic scale over a linearly scaled image.
    ax32.set_ylabel("Frequency [Hz] (log)",labelpad=30)
    twin_ax = ax32.twinx()
    twin_ax.set_yscale('log')
    twin_ax.set_xlim(t[0], t[-1])
    twin_ax.set_ylim(1, 50)
    ax32.tick_params(which='both', labelleft=False, left=False)
    twin_ax.tick_params(which='both', labelleft=True, left=True, labelright=False)
    IX = wavelet.icwt(X, st_won[0])
    ax34.plot(t, IX, 'k')
    ax34.set_xlabel("Time after %s [s]" % st_won[0].stats.starttime)
    fig3.colorbar(img, cax=ax33)
    fig3.autofmt_xdate()    
    fig3.savefig(event_list[e] + '_won.pdf', bbox_inches='tight')
