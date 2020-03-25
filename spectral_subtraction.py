import numpy as np
#Xn: signal with noise
#N: noise estimate - known for now
#
def simple_amplitude(Xn, N):
    Xwon = Xn - N
    return Xwon
    
def simple_magnitude(Xn, N): #haven't tested this yet. 
    Xwon = (Xn ** 2) - (N ** 2)
    return Xwon 