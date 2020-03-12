import sys
sys.path.append("/Users/ezgikarasozen/Documents/Research/Array_processing/asap_w_obspy_git/")
from getdata import event
from getdata import taup

st = event(event_list=['ak0184nb2fzw','ak014cou8d9l'], network_list=['IM','XM'], station_code="BC", pick="P", channel="SHZ", start_time=20*60, end_time=60*60)
print(st)



