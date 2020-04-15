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
noise_amplitude = 50
noise_freqmin = 1
noise_freqmax = 4
#noise estimation parameters
ibegin = 25
iend = 35
#cwt parameters
#cwt_type = "mlwt"
cwt_type = "pycwt"
dj = 0.05 #scale spacing
omega0 = 6
s0 = 0.096801331 # smallest scale, required for pycwt (maybe not? check this), mlpy automatically calculates this
wf = 'morlet' #type of wavelet for mlpy, is there a way to do this for pycwt too? check
