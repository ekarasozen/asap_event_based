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
                    t=i.time
    if time_array.size == 0:
       print("There are no available picks from the quakeml file, pick time will be calculated by the Taup")
       t = evot + taup(inventory,evlat,evlon,evdep,model_l=TauPyModel(model="northak"),phase_list=["P", "Pn", "Pg", "p"])
    try:
        st = client_wm.get_waveforms(network_list[0], station_code+"*", "*", channel, (t - start_time), (t + end_time), attach_response=True)
        if len(st) > 0:
            print('Waveform data found!')
    except Exception:
            print('No waveform data found!')
#            continue
    return st