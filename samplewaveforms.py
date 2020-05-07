# This script is intended to return one of a few standard test datasets for use in 
# benchmarking array algorithms
#
# ------- DATASET 1 ---------------
# event:        2018 shallow M4.0 in Prince William Sound 
# stations:     Beaver Creek array
# signal:       +- 300 seconds from BC01 P pick
# noise:        60s sample of degraded signal 
# processing:   detrend, demean
#
# ------- DATASET 2 ---------------
# event:        January 2020 shallow M6.1 in Amchitka Region
# stations:     ANM
# signal:       -300/+1200 seconds from ANM P pick
# noise:        60s sample of degraded signal 
# processing:   detrend, demean
# us60007g8m, ak020173qdzt


from obspy import UTCDateTime 				
from obspy import Trace
from obspy.core.event import read_events 	
from obspy.clients.fdsn import Client 		
client=Client("IRIS") 
import numpy as np


def load(scenario):

    if scenario==1:
        np.random.seed(42)
        amplitude = 5 
        fmin = 1.5
        fmax = 3
        # ORIGINAL WAVFORMS
        cat = read_events('https://earthquake.usgs.gov/fdsnws/event/1/query?eventid=ak0184nb2fzw')
        for i in cat[0].picks:
            if i.waveform_id.station_code=="BC01":
                picktime = i.time
        sto = client.get_waveforms('IM', 'BC0*', "*", 'SHZ', (picktime-300), (picktime+300), attach_response=True)        
        sto.detrend("linear")
        sto.detrend("demean")
        #MAKE NOISE SEGMENT, GARBAGE, AND DEGRADED DATA FOR EACH TRACE
        stg = sto.copy()
        std = sto.copy()
        stn = sto.copy()
        for i in range(len(sto)):
            tro = sto[i].copy()             # ----- original trace -----
            trg = sto[i].copy()             # ---- garbage data -----
            trg.data = np.random.normal(0 ,1 , len(trg.data))
            trg0 = trg.copy()               # copy of unfiltered garbage (for plots)
            trg.filter('bandpass', freqmin=fmin, freqmax=fmax, corners=5, zerophase=True)
            trg.data = trg.data*amplitude*tro.std()
            trd = tro.copy()
            trd.data = trd.data + trg.data  # ---- degraded data -----
            # st_tmp = Stream(traces=[tro, trg0, trg, trd])
            # st_tmp.plot(automerge=False,equal_scale=True,linewidth=.5,outfile='tmp.png')
            # st_tmp[2].spectrogram(log=False, title='TEST', wlen=10, cmap='hot',dbscale=False)
            trn = trd.copy()             # ----- noise sample ------
            trn.trim(picktime-120, picktime-60)
            trn.detrend("linear")
            trn.detrend("demean")
            stg[i] = trg
            std[i] = trd
            stn[i] = trn
#            sto.plot(automerge=False,equal_scale=True,linewidth=.5,outfile='TMP_traces_scenario01_o.png')
#            stg.plot(automerge=False,equal_scale=True,linewidth=.5,outfile='TMP_traces_scenario01_g.png')
#            std.plot(automerge=False,equal_scale=True,linewidth=.5,outfile='TMP_traces_scenario01_d.png')
#            stn.plot(automerge=False,equal_scale=True,linewidth=.5,outfile='TMP_traces_scenario01_n.png')            
        return sto, stg, std, stn, picktime


    elif scenario==2:
        np.random.seed(42)
        amplitude = 3 
        fmin = .5
        fmax = 2.0
        # ORIGINAL WAVFORMS
        cat = read_events('https://earthquake.usgs.gov/fdsnws/event/1/query?eventid=us60007g8m')
        for i in cat[0].picks:
            if i.waveform_id.station_code=="ANM":
                picktime = i.time
        sto = client.get_waveforms('AK', 'ANM', "*", 'BH*', (picktime-300), (picktime+1200), attach_response=True)        
        sto.detrend("linear")
        sto.detrend("demean")
        #MAKE NOISE SEGMENT, GARBAGE, AND DEGRADED DATA FOR EACH TRACE
        stg = sto.copy()
        std = sto.copy()
        stn = sto.copy()
        for i in range(len(sto)):
            tro = sto[i].copy()             # ----- original trace -----
            trg = sto[i].copy()             # ---- garbage data -----
            trg.data = np.random.normal(0 ,1 , len(trg.data))
            trg0 = trg.copy()               # copy of unfiltered garbage (for plots)
            trg.filter('bandpass', freqmin=fmin, freqmax=fmax, corners=5, zerophase=True)
            trg.data = trg.data*amplitude*tro.std()
            trd = tro.copy()
            trd.data = trd.data + trg.data  # ---- degraded data -----
            # st_tmp = Stream(traces=[tro, trg0, trg, trd])
            # st_tmp.plot(automerge=False,equal_scale=True,linewidth=.5,outfile='tmp.png')
            # st_tmp[2].spectrogram(log=False, title='TEST', wlen=10, cmap='hot',dbscale=False)
            trn = trd.copy()             # ----- noise sample ------
            trn.trim(picktime-120, picktime-60)
            trn.detrend("linear")
            trn.detrend("demean")
            stg[i] = trg
            std[i] = trd
            stn[i] = trn
#            sto.plot(automerge=False,equal_scale=True,linewidth=.5,outfile='TMP_traces_scenario01_o.png')
#            stg.plot(automerge=False,equal_scale=True,linewidth=.5,outfile='TMP_traces_scenario01_g.png')
#            std.plot(automerge=False,equal_scale=True,linewidth=.5,outfile='TMP_traces_scenario01_d.png')
#            stn.plot(automerge=False,equal_scale=True,linewidth=.5,outfile='TMP_traces_scenario01_n.png')            
        return sto, stg, std, stn, picktime
        
    else:
        print('Error: load module requires a valid scenario')



