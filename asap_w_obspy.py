from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.dates import date2num
import numpy as np
from obspy.clients.fdsn import Client
client_wm = Client("IRIS")
from obspy.clients.iris import Client  #this is needed for gc_distaz calculation, for some reason it doesn't accept fdsn
import obspy
from obspy.core.util import AttribDict
from obspy.imaging.cm import obspy_sequential
from obspy.signal.invsim import corn_freq_2_paz
from obspy.signal.array_analysis import array_processing
from obspy.core.event import read_events
from obspy import UTCDateTime
from obspy import read_inventory, read_events
from obspy import read
from matplotlib.ticker import FormatStrFormatter
# -*- coding: utf-8 -*-

from obspy.taup import TauPyModel
from obspy.taup import plot_travel_times
#CHOOSE FROM LOCAL V MODELS FOR TAUP CALCULATION
#model_t = TauPyModel(model="ak135")
model_l = TauPyModel(model="northak")
#model_l = TauPyModel(model="scak")
#model_l = TauPyModel(model="kaktovik")


#event_list = ['ak018aap2cqu'] #kaktovik mag6 mainshock
#event_list = ['ak018vt8d37', 'ak0198jpaeek']
#event_list = ['us2000cr6x', 'us2000cr8c', 'us2000cr37', 'us2000crj9', 'us2000cyn8', 'us2000cynn', 'us2000czdu', 'us2000czh4', 'us2000d0qp', 'us2000d1c0', 'us2000d097', 'us2000cnv4', 'us2000cp5m', 'us2000cp6u', 'us2000cp8n', 'us2000cpc1', 'us2000cplq', 'us2000cpmf', 'us2000cpuq', 'us2000cq7b', 'us2000cq9c', 'us2000cq9u', 'us2000cq92', 'us2000cqaw', 'us2000cqbt', 'us2000cqbw', 'us2000cqe7', 'us2000cqs2', 'us2000cqsv', 'us2000cquh', 'us2000cqus', 'us2000cqv1', 'us2000cqv9', 'us2000cqwq', 'us2000cn2c', 'us2000cn06', 'us2000cn9g', 'us2000cn17', 'us2000cn20', 'us2000cnbp', 'us2000cncp', 'us2000cnfc', 'us2000cnhe', 'us2000cnih', 'us2000cniv', 'us2000cnlq', 'us2000cnlx', 'us2000cnnb', 'us2000cnp3', 'us2000cnsa', 'ak018njcg8s', 'ak018vt8d37', 'ak0181u649m5', 'ak0181749ev4', 'us1000cd95', 'us1000cdnj', 'us1000cfjg', 'us1000cg4d', 'us1000cg7h', 'us1000che9', 'us1000chh4', 'us1000chqm', 'us1000chx1', 'us1000ci6q', 'us2000cmy9', 'us2000cmyb', 'us2000cmyc', 'us2000cmz0', 'us2000cn0n', 'us2000cn0y', 'us1000d6nh', 'us1000d5r5', 'ak0183b4m226', 'us1000d3mt', 'us1000d3m0', 'us1000d324', 'us1000d24j', 'ak01834kwkl7', 'us1000d1xl', 'us1000d1sj', 'us1000d1a0', 'us2000dduj', 'us2000dc5x', 'us2000dc1e', 'us2000dbwl', 'us2000dbw6', 'us2000dbvy', 'us2000dajg', 'us2000d93z', 'ak0182plgsrf', 'us2000d8bf', 'us2000d85j', 'ak0182ksu6cf', 'us2000d6lb', 'us2000d4wt', 'ak0182azrhvn', 'us2000d417', 'us2000d415', 'us2000d277', 'us2000d1pc', 'us1000e3ks', 'us1000e2re', 'us1000e1bp', 'us1000e1bk', 'us1000e0fk', 'us1000dxwb', 'us1000dvgc', 'us1000du7s', 'us1000du7f', 'us1000dtv0', 'us1000dq27', 'us1000dpy4', 'us2000e1sw', 'us2000e0z5', 'us2000e0qc', 'us2000e0kw', 'us2000e0jm', 'us2000e02w', 'ak0184nb2fzw', 'us2000dxyb', 'us2000dvrd', 'us2000dvr1', 'us1000ddgh', 'us1000dcsb', 'us1000dcd4', 'us1000dc6e', 'us1000dc0e', 'ak01843dldfm', 'us1000daza', 'us1000dayk', 'us1000d9vq', 'us1000d9ry', 'us1000d9et', 'ak0183tcw7lk', 'us1000d7kc', 'us1000e9md', 'us1000e96z', 'us1000e6b7', 'us1000e5dl', 'ak018aaq826y', 'ak018aapimtg', 'ak018aapf6fw', 'ak018aapea91', 'ak0189ni53xo', 'ak0189f3mjil', 'us2000g8u6', 'us2000g4lk', 'ak0188x5as43', 'ak0188rwahak', 'ak0188n4v5yj', 'ak0188laqakg', 'us2000ftxh', 'ak0188dbg38e', 'ak0188d4oyaj', 'us2000fsrv', 'us2000fqup', 'us2000fq8q', 'ak01881pylyd', 'us1000ez1v', 'ak0187o88z87', 'us1000erwb', 'us1000ermm', 'us1000eb8x', 'us1000eb7g', 'ak018b4lg4sy', 'ak018b13lw5m', 'us1000ghkn', 'us1000h3hd', 'ak018aswvwj1', 'ak018aspusyi', 'ak018anu8lce', 'ak018amcmumm', 'ak018am92po3', 'ak018akht57l', 'ak018akgatno', 'ak018akg7zql', 'ak018aixywwk', 'ak018ah6tyku', 'ak018ah6ang1', 'ak018ae1ar10', 'ak018ae034ky', 'ak018acfg224', 'ak018acbvbfa', 'ak018acahybq', 'ak018ac8zxt5', 'ak018ac6wakp', 'ak018aaufgk0', 'ak018aathb7x', 'ak018aat5b5o', 'ak018aat3zpq', 'ak018dtutpev', 'ak018dd9hnq1', 'ak018da83wwq', 'ak018d8nlpcd', 'ak018d04tq82', 'ak018cyhsaze', 'ak018cqboglj', 'us1000h62f', 'ak018cl5lkrj', 'us1000h4kd', 'us2000hfk6', 'us2000hfc8', 'ak20254307', 'ak20238992', 'ak018ep9y178', 'ak018fcoysc3', 'us1000hxdt', 'us1000huua']
event_list = ['ak0181u649m5', 'ak0181749ev4', 'us1000cd95', 'us1000cdnj', 'us1000cfjg', 'us1000cg4d', 'us1000cg7h', 'us1000che9', 'us1000chh4', 'us1000chqm', 'us1000chx1', 'us1000ci6q', 'us2000cmy9', 'us2000cmyb', 'us2000cmyc', 'us2000cmz0', 'us2000cn0n', 'us2000cn0y', 'us1000d6nh', 'us1000d5r5', 'ak0183b4m226', 'us1000d3mt', 'us1000d3m0', 'us1000d324', 'us1000d24j', 'ak01834kwkl7', 'us1000d1xl', 'us1000d1sj', 'us1000d1a0', 'us2000dduj', 'us2000dc5x', 'us2000dc1e', 'us2000dbwl', 'us2000dbw6', 'us2000dbvy', 'us2000dajg', 'us2000d93z', 'ak0182plgsrf', 'us2000d8bf', 'us2000d85j', 'ak0182ksu6cf', 'us2000d6lb', 'us2000d4wt', 'ak0182azrhvn', 'us2000d417', 'us2000d415', 'us2000d277', 'us2000d1pc', 'us1000e3ks', 'us1000e2re', 'us1000e1bp', 'us1000e1bk', 'us1000e0fk', 'us1000dxwb', 'us1000dvgc', 'us1000du7s', 'us1000du7f', 'us1000dtv0', 'us1000dq27', 'us1000dpy4', 'us2000e1sw', 'us2000e0z5', 'us2000e0qc', 'us2000e0kw', 'us2000e0jm', 'us2000e02w', 'ak0184nb2fzw', 'us2000dxyb', 'us2000dvrd', 'us2000dvr1', 'us1000ddgh', 'us1000dcsb', 'us1000dcd4', 'us1000dc6e', 'us1000dc0e', 'ak01843dldfm', 'us1000daza', 'us1000dayk', 'us1000d9vq', 'us1000d9ry', 'us1000d9et', 'ak0183tcw7lk', 'us1000d7kc', 'us1000e9md', 'us1000e96z', 'us1000e6b7', 'us1000e5dl', 'ak018aaq826y', 'ak018aapimtg', 'ak018aapf6fw', 'ak018aapea91', 'ak0189ni53xo', 'ak0189f3mjil', 'us2000g8u6', 'us2000g4lk', 'ak0188x5as43', 'ak0188rwahak', 'ak0188n4v5yj', 'ak0188laqakg', 'us2000ftxh', 'ak0188dbg38e', 'ak0188d4oyaj', 'us2000fsrv', 'us2000fqup', 'us2000fq8q', 'ak01881pylyd', 'us1000ez1v', 'ak0187o88z87', 'us1000erwb', 'us1000ermm', 'us1000eb8x', 'us1000eb7g', 'ak018b4lg4sy', 'ak018b13lw5m', 'us1000ghkn', 'us1000h3hd', 'ak018aswvwj1', 'ak018aspusyi', 'ak018anu8lce', 'ak018amcmumm', 'ak018am92po3', 'ak018akht57l', 'ak018akgatno', 'ak018akg7zql', 'ak018aixywwk', 'ak018ah6tyku', 'ak018ah6ang1', 'ak018ae1ar10', 'ak018ae034ky', 'ak018acfg224', 'ak018acbvbfa', 'ak018acahybq', 'ak018ac8zxt5', 'ak018ac6wakp', 'ak018aaufgk0', 'ak018aathb7x', 'ak018aat5b5o', 'ak018aat3zpq', 'ak018dtutpev', 'ak018dd9hnq1', 'ak018da83wwq', 'ak018d8nlpcd', 'ak018d04tq82', 'ak018cyhsaze', 'ak018cqboglj', 'us1000h62f', 'ak018cl5lkrj', 'us1000h4kd', 'us2000hfk6', 'us2000hfc8', 'ak20254307', 'ak20238992', 'ak018ep9y178', 'ak018fcoysc3', 'us1000hxdt', 'us1000huua']
#event_list = ['ak0198gdhuwa','ak0198jpaeek'] #test events
#run_name = input("Run Name (for output file names):") #YOU CAN CHANGE THIS TO FOLDER NAME MAYBE. 
event_id = ['https://earthquake.usgs.gov/fdsnws/event/1/query?eventid=' + s for s in event_list]
array_code = input("Array Code (BC, BM, IL, IM):")

filter_type = "bandpass" # waveform filtering
freqmin = 1 # waveform & array processing filtering
freqmax = 3 # waveform & array processing filtering
ts_full = 20 #for full waveform P - ts_full (in seconds)
te_full = 80 #for full waveform P + te_full (in seconds)
ts_win = 1 # for trimmed waveform P pick window +/- (in seconds)
te_win = 3 # for trimmed waveform P pick window +/- (in seconds)
win_len = te_win+ts_win #should be x2 of t_win in seconds

file1 = open(array_code + "_obspy_processing.out","w")


#plot waveforms wrt distance not alphabetically. 
#maybe work with better bmap options, if you have time


for e, lab in enumerate(event_id):
    fig1 = plt.figure(figsize=(20,8))
    ax1 = fig1.add_subplot(151)
    m = Basemap(resolution='l', projection='aea',
                llcrnrlon=190, llcrnrlat=50,
                urcrnrlon=235, urcrnrlat=72,
                lon_0=212.5, lat_0=61, ax=ax1)
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    m.fillcontinents(color='antiquewhite', lake_color='tab:blue')
    m.drawparallels(np.arange(-80., 81., 10.)) # draw parallels and meridians.
    m.drawmeridians(np.arange(-180., 181., 10.))
    m.drawmapboundary(fill_color='lightblue')
    fig1.bmap = m
    time_array = np.empty((0, 100))
    st_lat = np.empty((0, 100))
    st_lon = np.empty((0, 100))
    st_nam = np.empty((0, 100))
    cat = read_events(event_id[e])
    evot = (cat[0].origins[0].time)
    evlat = cat[0].origins[0].latitude
    evlon= cat[0].origins[0].longitude
    evdep= (cat[0].origins[0].depth) / 1000 #convert to km
    print(event_id[e])
    if cat[0].origins[0].evaluation_mode == "automatic":
       print("This event is not yet revised")
       continue
    event_date = cat[0].origins[0].time.date
    event_time = cat[0].origins[0].time.time.replace(microsecond=0)#TRICK TO CUT MILLISECONDS = ADD .replace(microsecond=0)
    cat.plot(fig=fig1, show=False, title="", color="date", colorbar=False)
    ax1.annotate(("{}"*5).format("Event time: ", event_date," ",event_time,"\n"), xy=(0, (1.1)), xycoords='axes fraction', fontsize=11)
    ax1.annotate(("{0:8}{1:6.3f}{2:2}{3:7.3f}{4:2}{5:3.0f}{6:2}").format("Origin: ", evlat,"\u00b0"" ", evlon,"\u00b0"" ", evdep,"km"), xy=(0, (1.05)), xycoords='axes fraction', fontsize=11)
    #ARRAY PICKS FROM QUAKEML
    picks = cat[0].picks
    for i in picks:
       station_code = i.waveform_id.station_code
       network_code = i.waveform_id.network_code
       if array_code == "BC":
          if (network_code=="IM" or network_code=="XM") and station_code[0:3]=="BC0":
             time_array = np.append(time_array, [i.time])
             t=i.time
             stn_pick = station_code # p pick is from this station
          #for cases when there are no picks from BC array but from the nearby TA
          elif network_code=="TA" and station_code=="L27K":
             time_array = np.append(time_array, [i.time])
             t=i.time
             stn_pick = station_code # p pick is from this station
       if array_code == "BM":
          if (network_code=="IM" or network_code=="XM") and station_code[0:3]=="BM0":
             time_array = np.append(time_array, [i.time])
             t=i.time
             stn_pick = station_code # p pick is from this station
       if array_code == "IL":
          if (network_code=="IM" or network_code=="XM") and station_code[0:2]=="IL": #this also covers IL31
             time_array = np.append(time_array, [i.time])
             t=i.time
             stn_pick = station_code # p pick is from this station
       if array_code == "IM":
          if (network_code=="IM" or network_code=="XM") and station_code[0:3]=="IM0":
             time_array = np.append(time_array, [i.time])
             t=i.time
             stn_pick = station_code # p pick is from this station
       #IF YOU WANT TO TAKE EARLIEST PICK FOR START TIME:
       #t = (np.amin(time_array))
    #OBSPY TAUP FOR THE CASES WHEN THERE ARE NO AVAILABLE PICKS FROM QUAKEML
    if time_array.size == 0:
       print("There are no available picks from the quakeml file, pick time will be calculated by the Taup")
       st_lat_tp = np.empty((0, 100))
       st_lon_tp = np.empty((0, 100))
       if array_code == "BC":
          inventory_tp = client_wm.get_stations(network="IM", station="BC*")
       if array_code == "BM":
          inventory_tp = client_wm.get_stations(network="IM", station="BM*")
       if array_code == "IL":
          inventory_tp = client_wm.get_stations(network="IM", station="IL*")
       if array_code == "IM":
          inventory_tp = client_wm.get_stations(network="IM", station="IM*")
       nos_tp = len(inventory_tp[0])
       for tp in range (nos_tp):
          st_lat_tp = np.append(st_lat_tp, [inventory_tp[0][tp].latitude])
          st_lon_tp = np.append(st_lon_tp, [inventory_tp[0][tp].longitude])
       stalat_tp = np.mean(st_lat_tp)
       stalon_tp = np.mean(st_lon_tp)
       client = Client()
       gc_distaz_tp = client.distaz(stalat=stalat_tp, stalon=stalon_tp, evtlat=evlat, evtlon=evlon)
       gc_dist_tp = gc_distaz_tp['distance']
       arrivals = model_l.get_travel_times(float(evdep),float(gc_dist_tp),phase_list=["P", "Pn", "Pg"])
       if len(arrivals) == 0:
          print('There are no available P, Pn or Pg picks from Taup')
          continue
#       arrivals_2 = model_l.get_travel_times(float(evdep),float(gc_dist_tp))
       else: 
          t = evot + arrivals[0].time #make sure this is the earliest one 
       stn_pick = "Calculated with Taup"
       #print(arrivals_2)
       #print(arrivals[0])
       #continue #this was being used before taup integration
    #WAVEFORMS FROM IRIS
    if array_code == "BC":
       array_name = "Beaver Creek"
       inventory = client_wm.get_stations(network="IM", station="BC*")
       try: #FOR THE CASES WHEN THERE ARE NO WAVEFORMS
          st = client_wm.get_waveforms("IM", "BC*", "*", "SHZ", t - (ts_full+0.1), t + (te_full+0.1), attach_response=True)
          if len(st) > 0:
             print('Waveform data found!')
       except Exception:
             print('No waveform data found!')
             continue
    if array_code == "BM":
       array_name = "Burnt Mountain"
       inventory = client_wm.get_stations(network="IM", station="BM*")
       try:
          st = client_wm.get_waveforms("IM", "BM*", "*", "SHZ", t - (ts_full+0.1), t + (te_full+0.1), attach_response=True)
          if len(st) > 0:
             print('Waveform data found!')
       except Exception:
             print('No waveform data found!')
             continue
    if array_code == "IL":
       array_name = "Eielson"
       inventory = client_wm.get_stations(network="IM", station="IL*")
       try:
          st = client_wm.get_waveforms("IM", "IL*", "*", "SHZ", t - (ts_full+0.1), t + (te_full+0.1), attach_response=True)
          if len(st) > 0:
             print('Waveform data found!')
       except Exception:
             print('No waveform data found!')
             continue
    if array_code == "IM":
       array_name = "Indian Mountain"
       inventory = client_wm.get_stations(network="IM", station="IM*")
       try:
          st = client_wm.get_waveforms("IM", "IM*", "*", "SHZ", t - (ts_full+0.1), t + (te_full+0.1), attach_response=True)
          if len(st) > 0:
             print('Waveform data found!')
       except Exception:
             print('No waveform data found!')
             continue
    st.filter(filter_type, freqmin=freqmin, freqmax=freqmax)
    st.taper(max_percentage=0.05, type='cosine')
    #PLOT STATIONS
    inventory.plot(fig=fig1, show=False)
    try:
       y_min = np.nanmin(st[:]) #for y axis to share same min and max values within all wfs
       y_max = np.nanmax(st[:])
    except NotImplementedError: #not sure why there is an error like this but this is the way around:
        print('To ambiguous, therefore not implemented, skipping this event')
        continue
    #PLOT WAVEFORMS
    nos = len(st)
    for s in range(nos):
        st[s].stats.coordinates = AttribDict({
           'latitude': inventory[0][s].latitude,
           'elevation': inventory[0][s].elevation,
           'longitude': inventory[0][s].longitude})
        st_lat = np.append(st_lat, [inventory[0][s].latitude])
        st_lon = np.append(st_lon, [inventory[0][s].longitude])
        st_nam = np.append(st_nam, [st[s].stats.station])
        ax2 = fig1.add_subplot(nos, 4, ((4*s)+2))
        wf_all = st[s]
        ax2.plot(wf_all.times("matplotlib"), wf_all.data, "k-", linewidth=0.3)
        text = (st[s].stats.station)
        ax2.set_ylim(ymin=y_min, ymax=y_max)
        ax2.xaxis_date()
        ax2.axvline(date2num((t-ts_win).datetime), lw=0.8, c='darkblue', ls='--', label='time win.')
        ax2.axvline(date2num((t+te_win).datetime), lw=0.8, c='darkblue', ls='--')
        ax2.axvline(date2num(t.datetime), lw=0.8, c='darkred', label='P pick')
        ax2.text(0.10, 0.95, text, transform=ax2.transAxes, fontsize=8, fontweight='bold', verticalalignment='top')
        st_trim = st[s].trim(t - (ts_win+0.1), t + (te_win+0.1)) #trim is done inside the loop, because otherwise st changes and whole wf cannot be plotted
        wf_trim = st_trim.taper(max_percentage=0.1, type='cosine')
        #wf_y_min = np.nanmin(wf_trim[:]) #for y axis to share same min and max values within all wfs
        #wf_y_max = np.nanmax(wf_trim[:])
        ax3 = fig1.add_subplot(nos, 4, ((4*s)+3))
        ax3.set_ylim(ymin=y_min, ymax=y_max)
        ax3.plot(wf_trim.times("matplotlib"), wf_trim.data, "k-", linewidth=0.3)
        ax3.xaxis_date()
        ax3.axvline(date2num((t-ts_win).datetime), lw=0.8, c='darkblue', ls='--', label='time win.')
        ax3.axvline(date2num((t+te_win).datetime), lw=0.8, c='darkblue', ls='--')
        ax3.axvline(date2num(t.datetime), lw=0.8, c='darkred')
        ax3.text(0.10, 0.95, text, transform=ax3.transAxes, fontsize=8, fontweight='bold', verticalalignment='top')
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax2.legend(loc='lower left', fontsize=6)
    ax2.annotate(("{}"*5).format("Full waveform: P - ",ts_full, " sec. to P + ",te_full, " sec."), xy=(0, (nos+0.1)), xycoords='axes fraction', fontsize=11)
    fig1.autofmt_xdate()
    if array_code == "IL":
       plt.text(date2num(t.datetime)-0.000002,0,'   P',rotation=90, color='darkred')
    else:
       plt.text(date2num(t.datetime)-0.000002,0,'P pick   ',rotation=90, color='darkred')
    ax3.annotate(("{}"*5).format("Trimmed waveform: P - ",ts_win, " sec. to P + ",te_win, " sec."), xy=(0, (nos+0.1)), xycoords='axes fraction', fontsize=11)
    ax3.legend(loc='lower left', fontsize=6)
    #GREAT CIRCLE BACKAZIMUTH & DISTANCE CALCULATION
    stalat = np.mean(st_lat)
    stalon = np.mean(st_lon)
    client = Client()
    gc_distaz = client.distaz(stalat=stalat, stalon=stalon, evtlat=evlat, evtlon=evlon)
    gc_dist = gc_distaz['distance']
    gc_baz = gc_distaz['backazimuth']
    #ax3.axvline(date2num(cal_pt.datetime), lw=0.8, c='cyan') #was used to plot taup pick. 
    stime = t - ts_win
    etime = t + te_win
    kwargs = dict(
       # slowness grid: X min, X max, Y min, Y max, Slow Step
       sll_x=-0.2, slm_x=0.2, sll_y=-0.2, slm_y=0.2, sl_s=0.002,
       # sliding window properties
       win_len=win_len, win_frac=0.05,
       # frequency properties
       frqlow=freqmin, frqhigh=freqmax, prewhiten=0,
       # restrict output
       semb_thres=-1e9, vel_thres=-1e9, timestamp='mlabday',
       stime=stime, etime=etime
    )
    out = array_processing(st, **kwargs)
    cl_rlp = out[:, 1] #calculated relative power
    cl_abp = out[:, 2] #calculated absolute power
    cl_baz = out[:, 3] #calculated backazimuth
    cl_slw = out[:, 4] #calculated slowness
#OBSPY ARRAY PROCESSING PLOT #2 POLAR PLOT
    cmap = obspy_sequential #colormaps
    # make output human readable, adjust backazimuth to values between 0 and 360
    t, rel_power, abs_power, baz, slow = out.T
    baz[baz < 0.0] += 360
    # choose number of fractions in plot (desirably 360 degree/N is an integer!)
    N = 36
    N2 = 30
    abins = np.arange(N + 1) * 360. / N
    sbins = np.linspace(0, 0.3, N2 + 1)
    # sum rel power in bins given by abins and sbins
    hist, baz_edges, sl_edges = np.histogram2d(baz, slow, bins=[abins, sbins], weights=rel_power)
    # transform to radian
    baz_edges = np.radians(baz_edges)
    # add polar and colorbar axes
    cax4 = fig1.add_axes([0.9, 0.20, 0.035, 0.15])
    ax4 = fig1.add_axes([0.65, 0.35, 0.40, 0.4], polar=True)
    ax4.set_theta_direction(-1)
    ax4.set_theta_zero_location("N")
    ax4.set_thetagrids(np.arange(0, 360, 20), labels=np.arange(0, 360, 20))
    dh = abs(sl_edges[1] - sl_edges[0])
    dw = abs(baz_edges[1] - baz_edges[0])
    # circle through backazimuth
    for i, row in enumerate(hist):
        bars = ax4.bar(x=(i * dw) * np.ones(N2),
                      height=dh * np.ones(N2),
                      width=dw, bottom=dh * np.arange(N2),
                     color=cmap(row / hist.max()))
    # set slowness limits
    ax4.set_ylim(0, 0.3)
#    ax4.set_title(("{}"*5).format("Beaver Creek "))
    [i.set_color('grey') for i in ax4.get_yticklabels()]
    ColorbarBase(cax4, cmap=cmap,
                norm=Normalize(vmin=hist.min(), vmax=hist.max()))
#OBSPY ARRAY PROCESSING PLOT #1
#    labels = ['rel_power', 'abs_power', 'baz', 'slowness']
#     xlocator = mdates.AutoDateLocator()
#     for i, lab in enumerate(labels):
#         ax4 = fig1.add_subplot(4, 4, (i + 1)*4)
#         ax4.scatter(out[:, 0], out[:, i + 1], c=out[:, 1], alpha=0.6,
#                    edgecolors='none', cmap=obspy_sequential)
#         ax4.set_ylabel(lab)
#         ax4.set_ylim(out[:, i + 1].min(), out[:, i + 1].max())
# #        ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2f')) #no need for now.
#         ax4.yaxis.set_label_position("right")
#         ax4.xaxis.set_major_locator(xlocator)
#         ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
#     ax4.annotate(("{}"*1).format("Array Processing with Obspy"), xy=(0, 4.08), xycoords='axes fraction', fontsize=11)
    fig1.subplots_adjust(left=0.15, top=0.95, right=0.95, bottom=0.2, hspace=0)
    file1.write("{0:14} {1:4.2f} {2:6.2f} {3:4.2f}".format(event_list[e],float(cl_rlp),float(cl_baz),float(cl_slw)))
    file1.write("\n")
    #TEXT TO APPEND
    ax1.annotate(("{}"*7).format("Filter type: ",filter_type, " ", freqmin, " - ", freqmax, " Hz" ), xy=(0, (-0.1)), xycoords='axes fraction', fontsize=11)
    ax1.annotate(("{}"*2).format("P pick is from: ",stn_pick), xy=(0, (-0.2)), xycoords='axes fraction', fontsize=11)
    ax1.annotate(("{}"*3).format("Time window length: ",win_len, " sec."), xy=(0, (-0.3)), xycoords='axes fraction', fontsize=11)
    ax1.annotate(("{0:10}{1:6.2f}{2:1}").format("True BAZ: ",float(gc_baz),"\u00b0"), xy=(0, (-0.4)), xycoords='axes fraction', fontsize=11)
    ax1.annotate(("{0:14}{1:4.2f}{2:1}").format("Apparent BAZ: ",float(np.mean(cl_baz)),"\u00b0"), xy=(0, (-0.5)), xycoords='axes fraction', fontsize=11)
    ax1.annotate(("{0:18}{1:4.2f}{2:1}").format("Apparent slowness: ",np.mean(cl_slw)," s/km"), xy=(0, (-0.6)), xycoords='axes fraction', fontsize=11)
    ax1.annotate(("{0:18}{1:4.2f}").format("Relative power: ",np.mean(cl_rlp)), xy=(0, (-0.7)), xycoords='axes fraction', fontsize=11)
    fig1.suptitle(("{}"*4).format("Event Name: ", event_list[e][2:], ", Array: ", array_name), y=1.04,fontweight='bold')
    fig1.savefig(event_list[e] + "_" + array_code + '_base.pdf', bbox_inches='tight')
#    fig1.savefig(run_name + "_" + event_list[e] + "_" + array_code + '_base.pdf', bbox_inches='tight')
file1.close()
    