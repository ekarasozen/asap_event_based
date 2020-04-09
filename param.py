#waveform parameters #
event_list=['ak0184nb2fzw','ak014cou8d9l']
network_list = ['IM','XM']
station_code = "BC"
pick_type = "P"
channel = "SHZ"
start_time = 20
end_time = 60
filter_type = "bandpass"
filter_freqmin = 1
filter_freqmax = 3
#whitenoise parameters, trg  [g]arbage (aka, noise) used to purposefully degrade a good signals 
noise_type = 2 #refer to addnoise for different types of whitenoise that can be added
noise_amplitude = 200
noise_freqmin = 0.1
noise_freqmax = 0.4
#noise estimation parameters, trn
ibegin = 0
iend = 200