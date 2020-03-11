# code takes quakeml info, calculates time from pick
# then gets the waveform info.
import numpy as np
from obspy.clients.fdsn import Client
client_wm = Client("IRIS")
from obspy.clients.iris import Client  #this is needed for gc_distaz calculation, for some            reason it doesn't accept fdsn
#need to implement different network codes (IM,XM,TA?)
#implement taup too. 
def event(evet_id, network_code, station_code, pick, channel, start_time, end_time):
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
        if (i_network_code==network_code) and i_station_code[0:2]==station_code:
            for j in arrivals:
                if j.pick_id.id==i.resource_id and j.phase==pick: 
                    time_array = np.append(time_array, [i.time])
                    t=i.time
       inventory = client_wm.get_stations(network=network_code, station=station_code"*")
       try: #FOR THE CASES WHEN THERE ARE NO WAVEFORMS
          st = client_wm.get_waveforms(network_code, station_code"*", "*", channel, (t - start_time), (t + end_time), attach_response=True)
          if len(st) > 0:
             print('Waveform data found!')
       except Exception:
             print('No waveform data found!')
             continue
    return st