import numpy as np
#Xn: signal with noise
#N: noise estimate - known for now (D in Fukane and Sahare)
    
def simple_subtraction(amp_Xd, amp_Xn, p): 
    #if p = 1; amplitude/magnitude subtraction
    #if p = 2; power subtraction
    #amp_Xd = np.abs(Xd)
    minamp = 0                # eqn 10 in Hassani 2011
    m, n = amp_Xd.shape 
    amp_Xp = np.zeros((m,n))
    amp_Xna = np.mean(amp_Xn,axis=1)
    for i in range(0,m):
        amp_Xp[i,:] = (amp_Xd[i,:] ** p) - (amp_Xna[i] ** p)
    result = np.where(amp_Xp<minamp**p)  # find low or negative values
    amp_Xp[result] = 0         # set minimum value, eqn 10 Hassani 2011
    amp_Xp = amp_Xp ** (1/p)             # square root has to come AFTER the negatives are removed
    #amp_Xp = (np.abs(amp_Xp)) ** (1/p)             # eqn 11 in Hassani 2011
    return amp_Xp 

def over_subtraction(amp_Xd, amp_Xn, p): #still working on this
    #alpha over subtraction factor, value greater than or equal to 1
    #beta spectral floor parameter value between 0 to 1
    m, n = amp_Xd.shape 
    SNR = np.zeros((m))
    alpha = np.zeros((m))
    amp_Xp = np.zeros((m,n))
    amp_Xda = np.mean(amp_Xd,axis=1)
    amp_Xna = np.mean(amp_Xn,axis=1)
    alpha0=4 #vary between 3-6 (Beruiti et al'79), normally taken as 4 (Kamath & Loizou'02)
    beta=0.5 #should be between 0-1.
    for i in range(0,m):
        SNR[i] = (amp_Xda[i])/(amp_Xna[i])
        if SNR[i] < -5:
            alpha[i] = alpha0+(3/4)
        elif SNR[i] >= -5 and SNR[i] < 20:
            alpha[i] = alpha0-(3/20*SNR[i])
        elif SNR[i] >= 20:
            alpha[i] = alpha0-3 
        print(alpha)
        amp_Xp[i,:] = (amp_Xd[i,:] ** p) - ((alpha[i])*(amp_Xna[i] ** p))
        for j in range(0,n):
            if amp_Xp[i,j] > (beta)*(amp_Xna[i] ** p): #Hassani 2011
#            if amp_Xp[i,j] > (beta+alpha[i])*(amp_Xna[i] ** p): Fukane 2011
                amp_Xp[i,j] = amp_Xp[i,j]
            else:
                amp_Xp[i,j] = (beta)*(amp_Xna[i] ** p)
        #np.savetxt('amp_Xp.out', amp_Xp, delimiter=',', newline="\n")   # X is an array
    amp_Xp = amp_Xp ** (1/p)             # square root has to come AFTER the negatives are removed
    return amp_Xp     