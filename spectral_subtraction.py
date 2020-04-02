import numpy as np
#Xn: signal with noise
#N: noise estimate - known for now (D in Fukane and Sahare)
#
    
def simple_subtraction(Xn, N, p): 
    #if p = 1; amplitude/magnitude subtraction
    #if p = 2; power subtraction
    Xwon = np.sqrt((Xn ** p) - (N ** p))
    return Xwon 

def over_subtraction(Xn, N): #haven't tested this yet
    #alpha over subtraction factor, value greater than or equal to 1
    #beta spectral floor parameter value between 0 to 1
    SNR = (np.average(Xn.real))/(np.std(Xn.real)) #this is for the entire waveform for now
    alpha0=4 #vary between 3-6 (Beruiti et al'79), normally taken as 4 (Kamath & Loizou'02)
    beta=0.5 #should be between 0-1.
    if SNR < -5:
        alpha = alpha0+(3/4)
    elif SNR >= -5 and SNR < 20:
        alpha = alpha0-(3/20*SNR)
    elif SNR >= 20:
        alpha = alpha0-3 
    for i in range(Xn.shape[0]):  # or range(len(theta))
        for j in range(Xn.shape[1]):  # or range(len(theta))
            if ((Xn[i,j].real) ** 2) > ((alpha + beta)*((N[i,j].real) ** 2)):
                Xwon = np.sqrt((Xn ** 2) - (N ** 2))
            else:
                 Xwon = np.sqrt((beta)*(N ** 2))
            #print(Xwon)
        np.savetxt('xwon.out', Xwon, delimiter=',')   # X is an array
        print(Xwon.shape)
        return Xwon 