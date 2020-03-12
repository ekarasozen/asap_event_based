import sys
sys.path.append("/Users/ezgikarasozen/Documents/Research/Array_processing/asap_w_obspy_git/")
from getdata import event
from getdata import taup
from prepdata import prep
from addnoise import whitenoise

event_list=['ak0184nb2fzw','ak014cou8d9l']
event_id = ['https://earthquake.usgs.gov/fdsnws/event/1/query?eventid=' + s for s in event_list]
for e, lab in enumerate(event_id):
    st = event(ev_id=event_id[e], network_list=['IM','XM'], station_code="BC", pick="P", channel="SHZ", start_time=20*60, end_time=60*60)
    st = prep(st,filter_type="bandpass", freqmin=1, freqmax=3)
    st_n= st.copy()
    nos = len(st_n)
    for s in range(nos):
        st_n[s].data = whitenoise(st_n[s],type="2",amplitude=20,min_freq=0.1,max_freq=0.4)
    print(st_n)



