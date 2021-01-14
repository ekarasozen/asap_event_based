# code takes quakeml info, calculates time from pick
# then gets the waveform info.
import numpy as np
from obspy import read_inventory, read_events
from obspy import read
from obspy.clients.fdsn import Client
client_wm = Client("IRIS")
from obspy.clients.iris import Client  #this is needed for gc_distaz calculation
from obspy.taup import TauPyModel
from obspy.taup import plot_travel_times
from obspy.core.util import AttribDict
from obspy import Stream, Trace
#TA option is not added. 

def taup(inventory,evlat,evlon,evdep,model_l,phase_list):
    st_lat_tp = np.empty((0, 100))
    st_lon_tp = np.empty((0, 100))
    nos_tp = len(inventory[0])
    for tp in range (nos_tp):
        st_lat_tp = np.append(st_lat_tp, [inventory[0][tp].latitude])
        st_lon_tp = np.append(st_lon_tp, [inventory[0][tp].longitude])
    stalat_tp = np.mean(st_lat_tp)
    stalon_tp = np.mean(st_lon_tp)
    client = Client()
    gc_distaz_tp = client.distaz(stalat=stalat_tp, stalon=stalon_tp, evtlat=evlat, evtlon=evlon)
    gc_dist_tp = gc_distaz_tp['distance']
    arrivals = model_l.get_travel_times(float(evdep),float(gc_dist_tp),phase_list)
    if len(arrivals) > 0:
       t = arrivals[0].time 
    else: 
       print('There are no available Pg, Pn, P or p picks from Taup')
#       continue this casse is not tested yet.
    return t
    

def keskin_event(db_name,ev_id,inventory, start_time, end_time):
    time_array = np.empty((0, 100))
    inventory = read_inventory(inventory, format= 'STATIONXML')
    cat = read_events(db_name, format= 'QUAKEML')
    evot = cat[ev_id].origins[0].time
    print(evot)
    evlat = cat[ev_id].origins[0].latitude
    evlon= cat[ev_id].origins[0].longitude
    evdep= (cat[ev_id].origins[0].depth) / 1000 #convert to km
    picks = cat[ev_id].picks
    arrivals = cat[ev_id].origins[0].arrivals
    for i in picks:
       i_station_code = i.waveform_id.station_code
       i_network_code = i.waveform_id.network_code
    if time_array.size == 0:
       print("There are no available picks from the quakeml file, pick time will be calculated by the Taup")
       picktime = evot + taup(inventory,evlat,evlon,evdep,model_l=TauPyModel(model="ak135"),phase_list=["P", "Pn", "Pg", "p"])
    st1 = read("../keskin_data/BR101.SHZ.2018") 
    st2 = read("../keskin_data/BR102.SHZ.2018") 
    st3 = read("../keskin_data/BR103.SHZ.2018") 
    st4 = read("../keskin_data/BR104.SHZ.2018") 
    st5 = read("../keskin_data/BR105.SHZ.2018") 
    st6 = read("../keskin_data/BR106.SHZ.2018") 
    st = Stream(traces=[st1[0],st2[0],st3[0],st4[0],st5[0], st6[0]])
    st[0].stats.coordinates = AttribDict({
        'latitude': 39.725346,
        'elevation': 1463.7,
        'longitude': 33.639096})    
    st[1].stats.coordinates = AttribDict({
        'latitude': 39.735643,
        'elevation': 1552.4,
        'longitude': 33.648541})    
    st[2].stats.coordinates = AttribDict({
        'latitude': 39.719633,
        'elevation': 1515.0,
        'longitude': 33.657081})    
    st[3].stats.coordinates = AttribDict({
        'latitude': 39.707446,
        'elevation': 1374.9,
        'longitude': 33.641485})    
    st[4].stats.coordinates = AttribDict({
        'latitude': 39.718039,
        'elevation': 1458.5,
        'longitude': 33.617315})    
    st[5].stats.coordinates = AttribDict({
        'latitude': 39.733703,
        'elevation': 1400.2,
        'longitude': 33.617991})    
    st.trim((picktime - start_time), (picktime + end_time))
    return st, picktime



def event(ev_id, network_list, station_code, pick, channel, start_time, end_time):
    time_array = np.empty((0, 100))
    cat = read_events(ev_id)
    evot = cat[0].origins[0].time
    evlat = cat[0].origins[0].latitude
    evlon= cat[0].origins[0].longitude
    evdep= (cat[0].origins[0].depth) / 1000 #convert to km
    print(ev_id)
    if cat[0].origins[0].evaluation_mode == "automatic":
       print("This event is not yet revised")
#       continue
    picks = cat[0].picks
    arrivals = cat[0].origins[0].arrivals
    inventory = client_wm.get_stations(network=network_list[0], station=station_code+"*")
    for i in picks:
        i_station_code = i.waveform_id.station_code
        i_network_code = i.waveform_id.network_code
        if (i_network_code==network_list[0] or i_network_code==network_list[1]) and i_station_code[0:2]==station_code:
            for j in arrivals:
                if j.pick_id.id==i.resource_id and j.phase==pick: 
                    time_array = np.append(time_array, [i.time])
                    picktime=i.time
    if time_array.size == 0:
       print("There are no available picks from the quakeml file, pick time will be calculated by the Taup")
       picktime = evot + taup(inventory,evlat,evlon,evdep,model_l=TauPyModel(model="northak"),phase_list=["P", "Pn", "Pg", "p"])
    try:
        st = client_wm.get_waveforms(network_list[0], station_code+"*", "*", channel, (picktime - start_time), (picktime + end_time), attach_response=True)
        if len(st) > 0:
            print('Waveform data found!')
    except Exception:
            print('No waveform data found!')
#            continue
    return st, picktime
    
def prep(st,filter_type,freqmin,freqmax):
    st.detrend("linear")
    st.detrend("demean")
    st.taper(max_percentage=0.05, type='cosine')
    #st.filter(filter_type, freqmin=freqmin, freqmax=freqmax)
    return st
