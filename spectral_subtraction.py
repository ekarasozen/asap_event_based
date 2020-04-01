import numpy as np
#Xn: signal with noise
#N: noise estimate - known for now (D in Fukane and Sahare)
#
    
def simple_subtraction(Xn, N, p): 
    #if p = 1; amplitude/magnitude subtraction
    #if p = 2; power subtraction
    Xwon = np.sqrt((Xn ** p) - (N ** p))
    return Xwon 

def over_subtraction(Xn, N, alpha, beta): #haven't tested this yet
    #alpha beta, or SNR values? 
    #alpha over subtraction factor, balue greater than or equal to 1
    #beta spectral floor parameter value between 0 to 1
    if (Xn ** 2) > (alpha + beta)*(N ** 2):
        Xwon = np.sqrt((Xn ** 2) - (N ** 2))
    else:
        Xwon = np.sqrt((beta)*(N ** 2))
    return Xwon 