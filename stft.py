import sys
sys.path.append("/Users/ezgikarasozen/Documents/Research/Array_processing/asap_w_obspy_git/")
from getdata import event
from getdata import taup
from getdata import prep
import parameters
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


event_list = parameters.event_list



event_id = ['https://earthquake.usgs.gov/fdsnws/event/1/query?eventid=' + s for s in event_list]
st = event(ev_id=event_id[0], network_list=['IM','XM'], station_code="BC", pick="P", channel="SHZ", start_time=20, end_time=60)
st = prep(st,filter_type="bandpass", freqmin=1, freqmax=3)
fs = st[0].stats.sampling_rate
print(fs)
amp = 2 * np.sqrt(2)
f, t, Zxx = signal.stft(st[0], fs)
plt.pcolormesh(t, f, np.abs(Zxx))
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()