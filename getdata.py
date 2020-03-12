# code takes quakeml info, calculates time from pick
# then gets the waveform info.
import numpy as np
from obspy import read_inventory, read_events
from obspy import read
from obspy.clients.fdsn import Client
client_wm = Client("IRIS")
from obspy.clients.iris import Client  #this is needed for gc_distaz calculation, for some            reason it doesn't accept fdsn
#need to implement different network codes (,TA?)
#implement taup too. 
def event(event_list, network_list, station_code, pick, channel, start_time, end_time):
    event_id = ['https://earthquake.usgs.gov/fdsnws/event/1/query?eventid=' + s for s in event_list]
    for e, lab in enumerate(event_id):
        time_array = np.empty((0, 100))
        cat = read_events(event_id[e])
        print(event_id[e])
        if cat[0].origins[0].evaluation_mode == "automatic":
           print("This event is not yet revised")
           continue
        picks = cat[0].picks
        arrivals = cat[0].origins[0].arrivals
        for i in picks:
            i_station_code = i.waveform_id.station_code
            i_network_code = i.waveform_id.network_code
            if (i_network_code==network_list[0] or i_network_code==network_list[1]) and i_station_code[0:2]==station_code:
                print(i_network_code,i_station_code)
                for j in arrivals:
                    if j.pick_id.id==i.resource_id and j.phase==pick: 
                        time_array = np.append(time_array, [i.time])
                        t=i.time
                        print(t)
        inventory = client_wm.get_stations(network=network_list[0], station=station_code+"*")
        print(inventory)
        try:
            st = client_wm.get_waveforms(network_list[0], station_code+"*", "*", channel, (t - start_time), (t + end_time), attach_response=True)
            if len(st) > 0:
                print('Waveform data found!')
        except Exception:
                print('No waveform data found!')
                continue
        return st