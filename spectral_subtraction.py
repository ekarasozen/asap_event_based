import numpy as np
#Xn: signal with noise
#N: noise estimate - known for now (D in Fukane and Sahare)
    
def simple_subtraction(amp_Xd, amp_Xn, p): 
    #if p = 1; amplitude/magnitude subtraction
    #if p = 2; power subtraction
    #amp_Xd = np.abs(Xd)
    minamp = 500                # I picked this value randomly
    m, n = amp_Xd.shape 
    amp_Xp = np.zeros((m,n))
    amp_Xna = np.mean(amp_Xn,axis=1)
    for i in range(0,m):
        amp_Xp[i,:] = (amp_Xd[i,:] ** p) - (amp_Xna[i] ** p)
    result = np.where(amp_Xp<minamp**p)  # find low or negative values
    amp_Xp[result] = minamp**p           # set minimum value
    amp_Xp = amp_Xp ** (1/p)             # square root has to come AFTER the negatives are removed
    return amp_Xp 

def over_subtraction(amp_Xd, amp_Xn): #still working on this
    #alpha over subtraction factor, value greater than or equal to 1
    #beta spectral floor parameter value between 0 to 1
    minamp = 500                # I picked this value randomly
    m, n = amp_Xd.shape 
    amp_Xp = np.zeros((m,n))
    amp_Xna = np.mean(amp_Xn,axis=1)
    SNR = (np.average(amp_Xd))/(np.std(amp_Xd)) #this is for the entire waveform for now, amp?
    print(SNR)
    alpha0=4 #vary between 3-6 (Beruiti et al'79), normally taken as 4 (Kamath & Loizou'02)
    beta=0.5 #should be between 0-1.
    if SNR < -5:
        alpha = alpha0+(3/4)
    elif SNR >= -5 and SNR < 20:
        alpha = alpha0-(3/20*SNR)
    elif SNR >= 20:
        alpha = alpha0-3 
    print(alpha)
    for i in range(0,m):
        if ((amp_Xd[i,i]) ** 2) > ((alpha + beta)*((amp_Xna[i]) ** 2)):
            amp_Xp[i,:] = (amp_Xd[i,:] ** 2) - ((alpha)*(amp_Xna[i] ** 2))
        else:
            amp_Xp[i,:] = (beta)*(amp_Xna[i] ** 2)
    result = np.where(amp_Xp<minamp**2)  # find low or negative values
    amp_Xp[result] = minamp**2           # set minimum value
    amp_Xp = amp_Xp ** (1/2)             # square root has to come AFTER the negatives are removed
    return amp_Xp     