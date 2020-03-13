import sys
sys.path.append("/Users/ezgikarasozen/Documents/Research/Array_processing/asap_w_obspy_git/")
from getdata import event
from getdata import taup
from prepdata import prep
from addnoise import whitenoise
import matplotlib.pyplot as plt
from obspy import Trace
from obspy import Stream

#event_list=['ak0184nb2fzw','ak014cou8d9l']
event_list=['ak0184nb2fzw']
event_id = ['https://earthquake.usgs.gov/fdsnws/event/1/query?eventid=' + s for s in event_list]
for e, lab in enumerate(event_id):
    st = event(ev_id=event_id[e], network_list=['IM','XM'], station_code="BC", pick="P", channel="SHZ", start_time=20, end_time=60)
    st = prep(st,filter_type="bandpass", freqmin=1, freqmax=3)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(311)
    ax1.plot(st[0].times("matplotlib"), st[0].data, "k-", linewidth=0.3)
    ax1.xaxis_date()
    st_n = st.copy()
    st_wo = st.copy()
    nos = len(st_n)
    for s in range(nos):
        st_n[s].data = whitenoise(st_n[s],type=2,amplitude=20,min_freq=0.1,max_freq=0.4)
        #st_n[s].data = whitenoise(st_n[s],type=1,amplitude=20)
        ax2 = fig1.add_subplot(312)
        ax2.plot(st_n[0].times("matplotlib"), st_n[0].data, "k-", linewidth=0.3)
        ax2.xaxis_date()
    for s in range(nos):
        st_wo[s].data = st_n[s].data - st[s].data
        ax3 = fig1.add_subplot(313)
        ax3.plot(st_wo[0].times("matplotlib"), st_wo[0].data, "k-", linewidth=0.3)
        ax3.xaxis_date()
fig1.autofmt_xdate()    
fig1.savefig('add_noise.pdf', bbox_inches='tight')
    



