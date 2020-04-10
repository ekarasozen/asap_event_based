import numpy as np
#Xn: signal with noise
#N: noise estimate - known for now (D in Fukane and Sahare)
    
def simple_subtraction(Xd, Xn, p): 
    #if p = 1; amplitude/magnitude subtraction
    #if p = 2; power subtraction
    amp_Xd = np.abs(Xd)
    amp_Xn = np.abs(Xn)
    m = (Xd.shape[0])
    n = (Xd.shape[1])
    amp_Xp = np.zeros((m,n))
    for i in range(0,m):
        amp_Xp[i,:] = np.sqrt((amp_Xd[i] ** p) - (amp_Xn[i] ** p))
#    np.savetxt('Xwon.out', Xwon, delimiter=',', newline='\n')   # X is an array
    return amp_Xp 

def over_subtraction(Xd, Xn):
    #alpha over subtraction factor, value greater than or equal to 1
    #beta spectral floor parameter value between 0 to 1
    amp_Xd = np.abs(Xd)
    amp_Xn = np.abs(Xd)
    SNR = (np.average(Xd))/(np.std(Xn)) #this is for the entire waveform for now, amp?
    alpha0=4 #vary between 3-6 (Beruiti et al'79), normally taken as 4 (Kamath & Loizou'02)
    beta=0.5 #should be between 0-1.
    if SNR < -5:
        alpha = alpha0+(3/4)
    elif SNR >= -5 and SNR < 20:
        alpha = alpha0-(3/20*SNR)
    elif SNR >= 20:
        alpha = alpha0-3 
    m = (amp_Xd.shape[0])
    n = (amp_Xd.shape[1])
    amp_Xp = np.zeros((m,n))
    for i in range(0,m):
        if ((amp_Xd[i,i]) ** 2) > ((alpha + beta)*((amp_Xn[i]) ** 2)):
            amp_Xp[i,:] = np.sqrt((amp_Xd[i] ** 2) - ((alpha)*(amp_Xn[i] ** 2)))
        else:
            amp_Xp[i,:] = np.sqrt((beta)*(amp_Xn[i] ** 2))
    return amp_Xp     