import sys
sys.path.append("/Users/ezgikarasozen/Documents/Research/Array_processing/asap_w_obspy_git/")
from getdata import event
from getdata import taup
from prepdata import prep
from addnoise import whitenoise
import matplotlib.pyplot as plt
from obspy import Trace
from obspy import Stream
from obspy.signal.tf_misfit import cwt
from obspy.imaging.cm import obspy_sequential
import numpy as np


# in your code
import json
file = open('deneme.json')
inputs = json.loads(file.read())
event_list = inputs['event_list']


#event_list=['ak0184nb2fzw','ak014cou8d9l']
#event_list=['ak0184nb2fzw']
event_id = ['https://earthquake.usgs.gov/fdsnws/event/1/query?eventid=' + s for s in event_list]
for e, lab in enumerate(event_id):
    st = event(ev_id=event_id[e], network_list=['IM','XM'], station_code="BC", pick="P", channel="SHZ", start_time=20, end_time=60)
    st = prep(st,filter_type="bandpass", freqmin=1, freqmax=3)
    fig1 = plt.figure()
    dx, dy = 0.05, 0.05
    ax1 = fig1.add_subplot(311)
    ax1.plot(st[0].times("matplotlib"), st[0].data, "k-", linewidth=0.3)
    ax1.xaxis_date()
    fig3 = plt.figure()
    npts = st[0].stats.npts
    dt = st[0].stats.delta
    t = np.linspace(0, dt * npts, npts)
    st_sc = cwt(st[0].data, dt, 8, 1, 50)
    ax21 = fig3.add_subplot(311)
    x, y = np.meshgrid(t, np.logspace(np.log10(1), np.log10(50), st_sc.shape[0]))
    ax21.set_xlabel("Time after %s [s]" % st[0].stats.starttime)
    ax21.set_ylabel("Frequency [Hz]")
    ax21.set_yscale('log')
    ax21.set_ylim(1, 50)                        
    st_n = st.copy()
    noise = st.copy()
    noise.clear()
    trace1 = Trace()
    trace1.stats.sampling_rate = 20.0
    noise = Stream(traces=[trace1])
    st_wo =noise.copy()
    nos = len(st)
    scal = ax21.pcolormesh(x, y, np.abs(st_sc), cmap=obspy_sequential)
    fig3.colorbar(scal)
    for s in range(nos):
        #st_n[s].data = whitenoise(st_n[s],type=2,amplitude=20,min_freq=0.1,max_freq=0.4)
        st_n[s].data = whitenoise(st_n[s],type=1,amplitude=20)
    ax2 = fig1.add_subplot(312)
    ax2.plot(st_n[0].times("matplotlib"), st_n[0].data, "k-", linewidth=0.3)
    ax2.xaxis_date()
    dt_n = st_n[0].stats.delta
    st_n_sc = cwt(st_n[0].data, dt, 8, 1, 50)
    ax22 = fig3.add_subplot(312)
    x, y = np.meshgrid(t, np.logspace(np.log10(1), np.log10(50), st_n_sc.shape[0]))
    ax22.set_xlabel("Time after %s [s]" % st_n[0].stats.starttime)
    ax22.set_ylabel("Frequency [Hz]")
    ax22.set_yscale('log')
    ax22.set_ylim(1, 50)                        
    n_scal = ax22.pcolormesh(x, y, np.abs(st_n_sc), cmap=obspy_sequential)
    fig3.colorbar(n_scal)
#    for s in range(nos):
#        noise[s].data = st_n[s].data - st[s].data
 #      st_wo[s].data = st_n[s].data - noise[s].data
    noise[0].data = st_n[0].data - st[0].data
    st_wo[0].data = st_n[0].data - noise[0].data
    ax3 = fig1.add_subplot(313)
    ax3.plot(st_wo[0].times("matplotlib"), st_wo[0].data, "k-", linewidth=0.3)
    ax3.xaxis_date()
    noise_sc = st_n_sc-st_sc 
    st_wo_sc = st_n_sc-noise_sc       
    ax23 = fig3.add_subplot(313)
    x, y = np.meshgrid(t, np.logspace(np.log10(1), np.log10(50), st_wo_sc.shape[0]))
    ax23.set_xlabel("Time after %s [s]" % st[0].stats.starttime)
    ax23.set_ylabel("Frequency [Hz]")
    ax23.set_yscale('log')
    ax23.set_ylim(1, 50) 
    wo_scal = ax23.pcolormesh(x, y, np.abs(st_wo_sc), cmap=obspy_sequential)
    fig3.colorbar(wo_scal)
fig1.autofmt_xdate()    
fig1.savefig('add_noise.pdf', bbox_inches='tight')
    

fig3.autofmt_xdate()    
fig3.savefig('add_noise_cwt.pdf', bbox_inches='tight')


