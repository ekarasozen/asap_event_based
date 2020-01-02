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
from obspy.core.util import AttribDict
from obspy.imaging.cm import obspy_sequential
from obspy.signal.invsim import corn_freq_2_paz
from obspy.signal.array_analysis import array_processing
from obspy.core.event import read_events
from obspy.core.event import ResourceIdentifier
# could also do:
# from obspy.core.event import *
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

#event_list = ['ak01743krbh2', 'ak0173ylxr8d', 'ak0171dm7fc5', 'ak017a95ih8', 'ak0167hy1d59', 'ak0166y3vho0', 'ak0166fx3f0m', 'ak0165h6po0t', 'ak0164u18ywd', 'ak0164k3z03c', 'ak0164gsjups', 'ak0164a41tzu', 'ak0163ylihyn', 'ak0163wxwn70', 'ak0163oocsvm', 'ak01637s98ri', 'ak0162zhxu4q', 'ak0162w6v3nk', 'ak0162nx85ru', 'ak0162kmqt2b', 'ak0161shz9cp', 'ak0161lvujcv', 'ak01615cgrp2', 'ak01610duz03', 'ak016evvbbc', 'ak015gbfpd44', 'ak015ft8vcrv', 'ak015f93ukeq', 'ak015ek9z6j7', 'ak015e8oqtm0', 'ak015dxhl4kt', 'ak015d3pxpm0', 'ak015c1b588a', 'ak015bwq413k', 'ak015bv2m0tt', 'ak015bl54jdp', 'ak015bct4cs7', 'ak015b4jejcs', 'ak015appz2xo', 'ak015ac5diey', 'ak0159h21fd0', 'ak0158x7ylke', 'ak0158s79jft', 'ak0158ew0dxq', 'ak0157jlwjqp', 'ak0157813cgc', 'ak0156o48o5o', 'ak0156kvkkbh', 'ak0153wwpwby', 'ak0152fnkuvs', 'ak015291c43f']

event_list = ['ak017a95ih8', 'ak0164a41tzu', 'ak0153wwpwby']




#event_list = ['ak018aap2cqu'] #kaktovik mag6 mainshock
#event_list = ['ak014azdbz0f'] #test events
#run_name = input("Run Name (for output file names):") #YOU CAN CHANGE THIS TO FOLDER NAME MAYBE. 
path = input("Enter the path of your file: ")
event_id = ['https://earthquake.usgs.gov/fdsnws/event/1/query?eventid=' + s for s in event_list]
array_code = input("Array Code (bc, bm, il, im):")

filter_type = "bandpass" # waveform filtering
freqmin = 2 # waveform & array processing filtering
freqmax = 4 # waveform & array processing filtering
ts_full = 10 #for full waveform P - ts_full (in seconds)
te_full = 10 #for full waveform P + te_full (in seconds)
ts_win = 1 # for trimmed waveform P pick window +/- (in seconds)
te_win = 4 # for trimmed waveform P pick window +/- (in seconds)
win_len = te_win+ts_win #should be x2 of t_win in seconds
ttIM = 45.5
ttBM = 44.1
ttBC = 48.7
ttIL =  5.5



file1 = open(path + array_code + "_obspy_processing.out","w")
file1.write("EVENT_ID"+"\t"+" AN"+"\t"+"DATE"+"\t"+"   TIME"+"\t"+"    MAG"+"\t"+"LAT"+"\t"+"   LON"+"\t"+"     DP"+"\t"+" DIST"+"\t"+" TBAZ"+"\t"+"MBAZ"+"\t"+"EBAZ"+" "+"RPW"+"  "+"SLW"+"\t"+"PICK"+"  "+"NOS"+"\t"+"TYPE"+"\n")


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
    #m.drawcountries()
    #m.drawstates()
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
    evmag = (cat[0].magnitudes[0].mag)
    evot = (cat[0].origins[0].time)
    evlat = cat[0].origins[0].latitude
    evlon= cat[0].origins[0].longitude
    evdep= (cat[0].origins[0].depth) / 1000 #convert to km
    evtype= cat[0].event_type
    print('##########################',event_list[e],'##########################')
    if cat[0].origins[0].evaluation_mode == "automatic":
       print("This event is not yet revised")
       continue
    event_date = cat[0].origins[0].time.date
    event_time = cat[0].origins[0].time.time.replace(microsecond=0)#TRICK TO CUT MILLISECONDS = ADD .replace(microsecond=0)
    cat.plot(fig=fig1, show=False, title="", color="date", colorbar=False)
    ax1.annotate(("{}"*5).format("Event time: ", event_date," ",event_time,"\n"), xy=(0, (1.1)), xycoords='axes fraction', fontsize=11)
    ax1.annotate(("{0:8}{1:6.3f}{2:2}{3:7.3f}{4:2}{5:3.0f}{6:3}").format("Origin: ", evlat,"\u00b0"" ", evlon,"\u00b0"" ", evdep," km"), xy=(0, (1.05)), xycoords='axes fraction', fontsize=11)
    #ARRAY PICKS FROM QUAKEML
    picks = cat[0].picks
    arrivals = cat[0].origins[0].arrivals
    # REMOVED LARGE SECTION SCANNING OVER PHASE PICKS .....
  
    # REMOVED LARGE TAUP SECTION ...

    
    #WAVEFORMS FROM IRIS
    print('========= array code: ',array_code)     ################
    stn_pick = "man"
    if array_code == "bc":
       array_name = "Beaver Creek"
       t = evot + ttBC
       inventory = client_wm.get_stations(network="IM", station="BC*")
       try: #FOR THE CASES WHEN THERE ARE NO WAVEFORMS
          st = client_wm.get_waveforms("IM", "BC*", "*", "SHZ", t - (ts_full+0.1), t + (te_full+0.1), attach_response=True)
          if len(st) > 0:
             print('Waveform data found!')
       except Exception:
             print('No waveform data found!')
             continue
    if array_code == "bm":
       array_name = "Burnt Mountain"
       t = evot + ttBM
       inventory = client_wm.get_stations(network="IM", station="BM*")
       try:
          st = client_wm.get_waveforms("IM", "BM*", "*", "SHZ", t - (ts_full+0.1), t + (te_full+0.1), attach_response=True)
          if len(st) > 0:
             print('Waveform data found!')
       except Exception:
             print('No waveform data found!')
             continue
    if array_code == "il":
       array_name = "Eielson"
       t = evot + ttIL
       inventory = client_wm.get_stations(network="IM", station="IL*")
       try:
          st = client_wm.get_waveforms("IM", "IL*", "*", "SHZ", t - (ts_full+0.1), t + (te_full+0.1), attach_response=True)
          if len(st) > 0:
             print('Waveform data found!')
       except Exception:
             print('No waveform data found!')
             continue
    if array_code == "im":
       array_name = "Indian Mountain"
       t = evot + ttIM
       inventory = client_wm.get_stations(network="IM", station="IM*")
       try:
          st = client_wm.get_waveforms("IM", "IM*", "*", "SHZ", t - (ts_full+0.1), t + (te_full+0.1), attach_response=True)
          if len(st) > 0:
             print('Waveform data found!')
       except Exception:
             print('No waveform data found!')
             continue
    st.detrend(type='simple')
    st.taper(max_percentage=0.05, type='cosine')
    st.filter(filter_type, freqmin=freqmin, freqmax=freqmax)
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
        ax3 = fig1.add_subplot(nos, 4, ((4*s)+3))
        #ax3.set_ylim(ymin=y_min/10, ymax=y_max/10) #until we figure out a better way to do this, it seems to work. or the following.
#         if y_max >= 1000: 
#            ax3.set_ylim(ymin=y_min/10, ymax=y_max/10)
#         else:
#            ax3.set_ylim(ymin=y_min, ymax=y_max)        
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
    err_baz = gc_baz - cl_baz
    if err_baz > 180:
       err_baz = err_baz - 360
    else:
       err_baz = err_baz
    #print('------>  ',gc_baz,float(cl_baz),err_baz) ######
    fig1.subplots_adjust(left=0.15, top=0.95, right=0.95, bottom=0.2, hspace=0)
    file1.write("{0:12} {1:2} {2:} {3:} {4:3.1f} {5:6.3f} {6:7.3f} {7:3.0f} {8:6.3f} {9:6.2f} {10:6.2f} {11:7.2f} {12:4.2f} {13:5.3f} {14:4} {15:2}     {16:10}".format(event_list[e],array_code,event_date,event_time,evmag,evlat,evlon,evdep,float(gc_dist),float(gc_baz),float(cl_baz),float(err_baz),float(cl_rlp),float(cl_slw),stn_pick,nos,evtype))
    file1.write("\n")
    #TEXT TO APPEND
    ax1.annotate(("{}"*7).format("Filter type: ",filter_type, " ", freqmin, " - ", freqmax, " Hz" ), xy=(0, (-0.1)), xycoords='axes fraction', fontsize=11)
    ax1.annotate(("{}"*2).format("P pick is from: ",stn_pick), xy=(0, (-0.2)), xycoords='axes fraction', fontsize=11)
    ax1.annotate(("{}"*3).format("Time window length: ",win_len, " sec."), xy=(0, (-0.3)), xycoords='axes fraction', fontsize=11)
    ax1.annotate(("{0:10}{1:6.2f}{2:1}").format("True BAZ: ",float(gc_baz),"\u00b0"), xy=(0, (-0.4)), xycoords='axes fraction', fontsize=11)
    ax1.annotate(("{0:14}{1:4.2f}{2:1}").format("Apparent BAZ: ",float(np.mean(cl_baz)),"\u00b0"), xy=(0, (-0.5)), xycoords='axes fraction', fontsize=11)
    ax1.annotate(("{0:18}{1:5.3f}{2:1}").format("Apparent slowness: ",np.mean(cl_slw)," s/km"), xy=(0, (-0.6)), xycoords='axes fraction', fontsize=11)
    ax1.annotate(("{0:18}{1:4.2f}").format("Relative power: ",np.mean(cl_rlp)), xy=(0, (-0.7)), xycoords='axes fraction', fontsize=11)
    fig1.suptitle(("{}"*4).format("Event Name: ", event_list[e][2:], ", Array: ", array_name), y=1.04,fontweight='bold')
    fig1.savefig(path + event_list[e] + "_" + array_code + '.pdf', bbox_inches='tight')
#    fig1.savefig(run_name + "_" + event_list[e] + "_" + array_code + '_base.pdf', bbox_inches='tight')
file1.close()
    