import numpy as np
import matplotlib.pyplot as plt
import mlpy.wavelet as wave
import sys
import math
sys.path.append("/Users/ezgikarasozen/Documents/Research/Array_processing/asap_w_obspy_git/")
from getdata import event
from getdata import taup
from prepdata import prep
from addnoise import whitenoise

from obspy.imaging.cm import obspy_sequential

event_id = ['https://earthquake.usgs.gov/fdsnws/event/1/query?eventid=ak0184nb2fzw']
st = event(ev_id=event_id[0], network_list=['IM','XM'], station_code="BC", pick="P", channel="SHZ", start_time=20, end_time=60)
st[0].data = whitenoise(st[0],type=2,amplitude=200,min_freq=0.1,max_freq=0.4)
N = len(st[0].data)
dt = st[0].stats.delta
dj = 0.05
t0=st[0].stats.starttime
time = np.linspace(0, dt * N, N)
signal = st[0].data
omega0 = 6

t = np.arange(st[0].stats.npts) / st[0].stats.sampling_rate


# approximate scales through frequencies
scales = wave.autoscales(N, dt=dt, dj=dj, wf='morlet', p=omega0)
freq = (omega0 + np.sqrt(2.0 + omega0 ** 2)) / (4 * np.pi * scales[1:])
X = wave.cwt(x=signal, dt=dt, scales=scales, wf='morlet', p=omega0)
idx = np.where(np.logical_and(freq>=0.1, freq<=0.4))[0]
#nl=200*st[0].std()

X[idx] = 1
X = (X).real

#From Torrence & Compo'97 Table 2
Cr=0.776 #reconstruction factor 
Wo=(math.pi)**(-1/4) #Psi0(0)
#missing part of the eqn 11 is defined here as Sc=Scale
Sc=(dj*math.sqrt(dt))/(Cr*Wo)
print(Sc)
IX = wave.icwt(X=(X*Sc), dt=dt, scales=scales, wf='morlet', p=omega0)
#print(Y.shape)
#np.savetxt('wavelet_8.out', Y, delimiter=',', newline='n')
#np.savetxt('inv_wavelet_8.out', IX, delimiter=' ', newline='n')

fig = plt.figure()
ax1 = fig.add_axes([0.1, 0.75, 0.7, 0.2])
ax2 = fig.add_axes([0.1, 0.1, 0.7, 0.60], sharex=ax1)
ax3 = fig.add_axes([0.83, 0.1, 0.03, 0.6])
ax4 = fig.add_axes([0.1, -0.15, 0.7, 0.2], sharex=ax1)

ax1.plot(t, st[0].data, 'k')

img = ax2.imshow(np.abs(X), extent=[t[0], t[-1], freq[-1], freq[0]],
                 aspect='auto', interpolation='nearest', cmap=obspy_sequential)
# Hackish way to overlay a logarithmic scale over a linearly scaled image.
ax2.set_ylabel("Frequency [Hz] (log)",labelpad=30)
twin_ax = ax2.twinx()
twin_ax.set_yscale('log')
twin_ax.set_xlim(t[0], t[-1])
twin_ax.set_ylim(1, 50)
ax2.tick_params(which='both', labelleft=False, left=False)
#twin_ax.set_yscale('log')
twin_ax.tick_params(which='both', labelleft=True, left=True, labelright=False)
ax4.plot(t, IX, 'k')
ax4.set_xlabel("Time after %s [s]" % st[0].stats.starttime)
fig.colorbar(img, cax=ax3)

#plt.show()



fig.autofmt_xdate()    
fig.savefig('cwt_noise.pdf', bbox_inches='tight')
