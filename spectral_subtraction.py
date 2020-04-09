import numpy as np
#Xn: signal with noise
#N: noise estimate - known for now (D in Fukane and Sahare)
    
def simple_subtraction(Xn, N, p): 
    #if p = 1; amplitude/magnitude subtraction
    #if p = 2; power subtraction
    aXn = np.abs(Xn)
    aN = np.abs(N)
    m = (Xn.shape[0])
    n = (Xn.shape[1])
    Xwon = np.zeros((m,n))
    for i in range(0,m):
        Xwon[i,:] = np.sqrt((aXn[i] ** p) - (aN[i] ** p))
    np.savetxt('Xwon.out', Xwon, delimiter=',', newline='\n')   # X is an array
    return Xwon 

def over_subtraction(Xn, N):
    #alpha over subtraction factor, value greater than or equal to 1
    #beta spectral floor parameter value between 0 to 1
    aXn = np.abs(Xn)
    aN = np.abs(N)
    SNR = (np.average(Xn))/(np.std(Xn)) #this is for the entire waveform for now
    alpha0=4 #vary between 3-6 (Beruiti et al'79), normally taken as 4 (Kamath & Loizou'02)
    beta=0.5 #should be between 0-1.
    if SNR < -5:
        alpha = alpha0+(3/4)
    elif SNR >= -5 and SNR < 20:
        alpha = alpha0-(3/20*SNR)
    elif SNR >= 20:
        alpha = alpha0-3 
    m = (aXn.shape[0])
    n = (aXn.shape[1])
    Xwon = np.zeros((m,n))
    for i in range(0,m):
        if ((aXn[i,i]) ** 2) > ((alpha + beta)*((aN[i]) ** 2)):
            Xwon[i,:] = np.sqrt((aXn[i] ** 2) - ((alpha)*(aN[i] ** 2)))
        else:
            Xwon = np.sqrt((beta)*(N[i] ** 2))
    return Xwon     