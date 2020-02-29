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
from array_processing.algorithms.helpers import getrij, wlsqva_proc
from array_processing.tools.plotting import array_plot
from array_processing.tools import arraySig
from array_processing.tools.plotting import arraySigPlt, arraySigContourPlt

#CHOOSE FROM LOCAL V MODELS FOR TAUP CALCULATION
#model_t = TauPyModel(model="ak135")
model_l = TauPyModel(model="northak")
#model_l = TauPyModel(model="scak")
#model_l = TauPyModel(model="kaktovik")

#ALL 191 FINAL SELECTION BENCHMARK DATASET AS OF FEB 12 2020
event_list = ['ak014dlss56k', 'ak014dltci5h', 'ak014e8xt2cn', 'ak014frher8l', 'ak014ft3m0mj', 'ak014fy4p2ie', 'ak015dyswg6d', 'ak015e8owu57', 'ak015f2i7ddl', 'ak016aw4yt72', 'ak016bjapymo', 'ak016c6edk4t', 'ak016d02wlg5', 'ak016exvhy2o', 'ak016frfe9e2', 'ak016ggavb4a', 'ak016nguz3k ', 'ak017b65zkfy', 'ak017bodzh0t', 'ak017bv19dof', 'ak017crsm45v', 'ak017df0437h', 'ak017e7b798p', 'ak017ec3jenv', 'ak018a0vq7zd', 'ak018cbizgzx', 'ak018coiadru', 'ak018cyhsaze', 'ak018d8dv7u8', 'ak018d718yqc', 'ak018da4zt40', 'ak018dlecu0 ', 'ak018dtutpev', 'ak018ekig6gu', 'ak018exsi6u4', 'ak018fcoysc3', 'ak018fe8g0c9', 'ak018ffr9ir9', 'ak018ggaykit', 'ak018lmayyr ', 'ak018njcg8s ', 'ak018vt8d37 ', 'ak0142zpz7gu', 'ak0143elpe3h', 'ak0144f1g1sr', 'ak0146arywc2', 'ak0146uon19t', 'ak0153elsnkx', 'ak0161ae7d1u', 'ak0161ui3nog', 'ak0161w3grph', 'ak0162cceroh', 'ak0162kq0d9f', 'ak0166arin7w', 'ak0172cmv3bz', 'ak0175vrtiuu', 'ak0176ukq2m ', 'ak0178ry7omj', 'ak0181u649m5', 'ak0182plgsrf', 'ak0184nb2fzw', 'ak0188d4oyaj', 'ak0188dbg38e', 'ak0188laqakg', 'ak0188n4v5yj', 'ak0188rwahak', 'ak0188yre6y6', 'ak0189f3mjil', 'ak0193exoam ', 'ak01418xv3ou', 'ak01439k6eg1', 'ak01548d9sgp', 'ak01613w9jul', 'ak01613z8owf', 'ak01615h8iq3', 'ak01660vze8u', 'ak01731b4jgx', 'ak01768zlboj', 'ak01834kwkl7', 'ak01834pp956', 'ak01843dldfm', 'ak01881pylyd', 'ak01959muqd ', 'ak20238992  ', 'ak014azd2qdx', 'ak017ekiw14d', 'ak0144yzrbk3', 'ak0152w8vj95', 'ak01529eirts', 'ak11283063  ', 'ak11283102  ', 'ak0162r8cyoc', 'ak019f60bldg', 'ak019cjifiox', 'ak0195aujzk ', 'ak018fcly2gn', 'ak018exuv372', 'ak018dvjh5lp', 'ak018bt9hjyj', 'ak018brqwtxf', 'ak018bhnjq88', 'ak018a0o8sox', 'ak018385is5m', 'ak0181pj43t1', 'ak018bslaoq ', 'ak017gq6fcgb', 'ak017acht7h1', 'ak01798mrffb', 'ak0175dkwkh5', 'ak017364arsu', 'ak0172wgoepm', 'ak016gmqlubi', 'ak016g81va03', 'ak016esnderb', 'ak016cmvhua7', 'ak0167ecnhxm', 'ak0164vknuin', 'ak0163b3daun', 'ak0162zo7agy', 'ak12582327  ', 'ak015gl77xng', 'ak015g6a50za', 'ak015exsjmzz', 'ak015dvqf7ni', 'ak015c1bohz3', 'ak0155c2fepa', 'ak0154nab8or', 'ak015334kpsf', 'ak01512dcxjw', 'ak015s9n15d ', 'ak014g82h2ix', 'ak014ezbxm08', 'ak014ec80d8x', 'ak014cesdo83', 'ak0147q41giz', 'ak0147q2ucs5', 'ak0147mqlwe6', 'ak0146wg24xs', 'ak0146km1s2x', 'ak01450fkr51', 'ak01450db0ka', 'ak0144yzrbk3', 'ak0143tdy6iq', 'ak0143cqwuiz', 'ak11170007  ', 'ak0142cdxuwg', 'ak0141ki158f', 'ak014liqjwc ', 'ak0149xaih2 ', 'ak014573iey ', 'ak0198oydwoy', 'ak018gq3vdva', 'ak018g66jrl0', 'ak018fo7boo4', 'ak018ep9y178', 'ak018dd9hnq1', 'ak018da83wwq', 'ak018cyhsaze', 'ak018cqboglj', 'ak018cl5lkrj', 'ak018c1nz4l9', 'ak018bzunhdz', 'ak018bt43tkf', 'ak018bfv88tt', 'ak018be7ojx4', 'ak018b7my403', 'ak018b4lg4sy', 'ak0162fousa0', 'ak01549xx0v9', 'ak0141iq8aeb', 'ak018efj6da9', 'ak018d718yqc', 'ak01634ktrdx', 'ak0147odw83x', 'ak01479cs1ql', 'ak01460tnurs', 'ak0145qxe8b6', 'ak0145qx2tq1', 'ak0145nmf4mh', 'ak0145nlz3oh', 'ak0145nlrjqg', 'ak0145nlaph8', 'ak0145nl41pp', 'ak0145nkrjaf', 'ak016gg2nkth', 'ak0153hov4fk', 'ak0145aam3cq']

#event_list = ['ak018aap2cqu', 'ak014azdbz0f'] #kaktovik mag6 mainshock
#event_list = ['ak014azdbz0f'] #test events
#run_name = input("Run Name (for output file names):") #YOU CAN CHANGE THIS TO FOLDER NAME MAYBE. 
path = input("Enter the path of your file: ")
event_id = ['https://earthquake.usgs.gov/fdsnws/event/1/query?eventid=' + s for s in event_list]
array_code = input("Array Code (bc, bm, il, im):")

filter_type = "bandpass" # waveform filtering
freqmin = 1 # waveform & array processing filtering
freqmax = 3 # waveform & array processing filtering
ts_full = 20 #for full waveform P - ts_full (in seconds)
te_full = 80 #for full waveform P + te_full (in seconds)
ts_win = 1 # for trimmed waveform P pick window +/- (in seconds)
te_win = 3 # for trimmed waveform P pick window +/- (in seconds)
win_len = te_win+ts_win #should be x2 of t_win in seconds
WINOVER = 0.04

file1 = open(path + array_code + "_uaf_processing.out","w")
file1.write("EVENT_ID"+"\t"+" AN"+"\t"+"DATE"+"\t"+"   TIME"+"\t"+"    MAG"+"\t"+"LAT"+"\t"+"   LON"+"\t"+"     DP"+"\t"+" DIST"+"\t"+" TBAZ"+"\t"+"MBAZ"+"\t"+"EBAZ"+" "+"RPW"+"  "+"SLW"+"  "+"ERR"+"\t"+"PICK"+"  "+"NOS"+"\t"+"TYPE"+"\n")


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
    evmag = (cat[0].magnitudes[0].mag)
    evot = (cat[0].origins[0].time)
    evlat = cat[0].origins[0].latitude
    evlon= cat[0].origins[0].longitude
    evdep= (cat[0].origins[0].depth) / 1000 #convert to km
    evtype= cat[0].event_type
    print(event_id[e])
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
    for i in picks:
       station_code = i.waveform_id.station_code
       network_code = i.waveform_id.network_code
       if array_code == "bc":
          if (network_code=="IM" or network_code=="XM") and station_code[0:3]=="BC0":
             for j in arrivals:
                if j.pick_id.id==i.resource_id and j.phase=="P": #done to make sure this is a P pick. for loops are most probably redundant but couldn't figure out a better way to solve this *yet*
                   time_array = np.append(time_array, [i.time])
                   t=i.time
             stn_pick = station_code # p pick is from this station
          #for cases when there are no picks from BC array but from the nearby TA
          elif network_code=="TA" and station_code=="L27K":
             time_array = np.append(time_array, [i.time])
             for j in arrivals:
                if j.pick_id.id==i.resource_id and j.phase=="P":
                   time_array = np.append(time_array, [i.time])
                   t=i.time
             stn_pick = station_code # p pick is from this station
       if array_code == "bm":
          if (network_code=="IM" or network_code=="XM") and station_code[0:3]=="BM0":
             time_array = np.append(time_array, [i.time])
             for j in arrivals:
                if j.pick_id.id==i.resource_id and j.phase=="P":
                   time_array = np.append(time_array, [i.time])
                   t=i.time
             stn_pick = station_code # p pick is from this station
       if array_code == "il":
          if (network_code=="IM" or network_code=="XM") and station_code[0:2]=="IL": #this also covers IL31
             time_array = np.append(time_array, [i.time])
             for j in arrivals:
                if j.pick_id.id==i.resource_id and j.phase=="P":
                   time_array = np.append(time_array, [i.time])
                   t=i.time
             stn_pick = station_code # p pick is from this station
       if array_code == "im":
          if (network_code=="IM" or network_code=="XM") and station_code[0:3]=="IM0":
             time_array = np.append(time_array, [i.time])
             for j in arrivals:
                if j.pick_id.id==i.resource_id and j.phase=="P":
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
       if array_code == "bc":
          inventory_tp = client_wm.get_stations(network="IM", station="BC*")
       if array_code == "bm":
          inventory_tp = client_wm.get_stations(network="IM", station="BM*")
       if array_code == "il":
          inventory_tp = client_wm.get_stations(network="IM", station="IL*")
       if array_code == "im":
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
       arrivals = model_l.get_travel_times(float(evdep),float(gc_dist_tp),phase_list=["P", "Pn", "Pg", "p"])
       if len(arrivals) == 0:
          print('There are no available Pg, Pn, P or p picks from Taup')
          continue
#       arrivals_2 = model_l.get_travel_times(float(evdep),float(gc_dist_tp))
       else: 
          t = evot + arrivals[0].time #make sure this is the earliest one 
       stn_pick = "Taup"
       #print(arrivals_2)
       #print(arrivals[0])
       #continue #this was being used before taup integration
    #WAVEFORMS FROM IRIS
    if array_code == "bc":
       array_name = "Beaver Creek"
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
       inventory = client_wm.get_stations(network="IM", station="IM*")
       try:
          st = client_wm.get_waveforms("IM", "IM*", "*", "SHZ", t - (ts_full+0.1), t + (te_full+0.1), attach_response=True)
          if len(st) > 0:
             print('Waveform data found!')
       except Exception:
             print('No waveform data found!')
             continue
    st.detrend("linear")
    st.detrend("demean")
    st.taper(max_percentage=0.05, type='cosine')
    st.filter(filter_type, freqmin=freqmin, freqmax=freqmax,corners=2, zerophase=True)
    tvec = st[0].times('matplotlib')
    #PLOT STATIONS
    inventory.plot(fig=fig1, show=False)
    try:
       y_min = np.nanmin(st[:]) #for y axis to share same min and max values within all wfs
       y_max = np.nanmax(st[:])
    except NotImplementedError: #not sure why there is an error like this but this is the way around:
        print('To ambiguous, therefore not implemented, skipping this event')
        continue
    tr= st.copy()
    st_trim = st.trim(t - (ts_win+0.1), t + (te_win+0.1)) #trim is done inside the loop, because otherwise st changes and whole wf cannot be plotted
#    st_trim_t = st_trim+t.taper(max_percentage=0.1, type='cosine')
    try:
        y_min_trim = np.nanmin(st_trim[:]) #for y axis to share same min and max values within all wfs
        y_max_trim = np.nanmax(st_trim[:])
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
        wf_all = tr[s]
        ax2.plot(wf_all.times("matplotlib"), wf_all.data, "k-", linewidth=0.3)
        text = (st[s].stats.station)
        wf_trim= st_trim[s]
        ax2.set_ylim(ymin=y_min, ymax=y_max)
        ax2.xaxis_date()
        ax2.axvline(date2num((t-ts_win).datetime), lw=0.8, c='darkblue', ls='--', label='time win.')
        ax2.axvline(date2num((t+te_win).datetime), lw=0.8, c='darkblue', ls='--')
        ax2.axvline(date2num(t.datetime), lw=0.8, c='darkred', label='P pick')
        ax2.text(0.10, 0.95, text, transform=ax2.transAxes, fontsize=8, fontweight='bold', verticalalignment='top')
        #st_trim = st[s].trim(t - (ts_win+0.1), t + (te_win+0.1)) #trim is done inside the loop, because otherwise st changes and whole wf cannot be plotted
        #wf_trim = st_trim.taper(max_percentage=0.1, type='cosine')
        ax3 = fig1.add_subplot(nos, 4, ((4*s)+3))
        ax3.plot(wf_trim.times("matplotlib"), wf_trim.data, "k-", linewidth=0.3)
        ax3.set_ylim(ymin=y_min_trim, ymax=y_max_trim)
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
##################################ARRAY_PROCESSING_STARTS########################################################################################################################################
    rij = getrij(st_lat, st_lon)
    vel, baz, sig_tau, mdccm, t, data = wlsqva_proc(st, rij, tvec, win_len, WINOVER)
    #print(vel,baz,sig_tau,mdccm)
    cl_baz = baz
    cl_slw = 1/vel
    cl_rlp = mdccm
#   fig11, axs1 = array_plot(st, t, mdccm, vel, cl_baz, ccmplot=True, sigma_tau=sig_tau)
#   fig11.savefig(path + event_list[e] + "_1_" + array_code + '.pdf', bbox_inches='tight')
#   SIGLEVEL = 1/st[0].stats.sampling_rate
#   KMAX = 400
#   TRACE_V = 0.33
#   sigV, sigTh, impResp, vel, th, kvec = arraySig(rij, kmax=KMAX, sigLevel=SIGLEVEL)
#    fig12 = arraySigPlt(rij, SIGLEVEL, sigV, sigTh, impResp, vel, th, kvec)
#   fig12.savefig(path + event_list[e] + "_2_" + array_code + '.pdf', bbox_inches='tight')
#   fig13 = arraySigContourPlt(sigV, sigTh, vel, th, trace_v=TRACE_V)
#   fig13.savefig(path + event_list[e] + "_3_" + array_code + '.pdf', bbox_inches='tight')
    #%% Delay and sum beam
    from array_processing.tools import beamForm
    beam = beamForm(data, rij, st[0].stats.sampling_rate, 50)
    #%% Pure state filter
    from array_processing.tools import psf
    x_psf, P = psf(data, p=2, w=3, n=3, window=None)
##################################ARRAY_PROCESSING_OVER########################################################################################################################################
    err_baz = gc_baz - cl_baz
    if err_baz > 180:
       err_baz = err_baz - 360
    elif err_baz < -180:
       err_baz = err_baz + 360
    else:
       err_baz = err_baz
    #print(gc_baz,float(cl_baz),err_baz)
    fig1.subplots_adjust(left=0.15, top=0.95, right=0.95, bottom=0.2, hspace=0)

    file1.write("{0:12} {1:2} {2:} {3:} {4:3.1f} {5:6.3f} {6:7.3f} {7:3.0f} {8:6.3f} {9:6.2f} {10:6.2f} {11:7.2f} {12:4.2f} {13:5.3f}  {14:5.3f} {15:4} {16:2}     {17:10}".format(event_list[e],array_code,event_date,event_time,evmag,evlat,evlon,evdep,float(gc_dist),float(gc_baz),float(cl_baz),float(err_baz),float(cl_rlp),float(cl_slw),float(sig_tau),stn_pick,nos,evtype))
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
##    fig1.savefig(run_name + "_" + event_list[e] + "_" + array_code + '_base.pdf', bbox_inches='tight')
#file1.close()