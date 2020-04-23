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

def over_subtraction(amp_Xd, amp_Xn, p): 
    #alpha over subtraction factor, value greater than or equal to 1
    #beta spectral floor parameter value between 0 to 1
    m, n = amp_Xd.shape 
    SNR = np.zeros((m))
    alpha = np.zeros((m))
    amp_Xp = np.zeros((m,n))
    amp_Xda = np.mean(amp_Xd,axis=1)
    amp_XdP = amp_Xd ** p
    amp_Xna = np.mean(amp_Xn,axis=1)
    amp_XnaP = amp_Xna ** p
    alpha0=4 #vary between 3-6 (Beruiti et al'79), normally taken as 4 (Kamath & Loizou'02)
    beta=0.2 #should be between 0-1.
    #result = np.transpose(np.where(np.locgical_and(freqs_d>0.5,freqs_d<10)))
    for i in range(0,m):
        #SNR[i] = np.sum(amp_XdP[result,:]) / np.sum(amp_XnaP[result])
        SNR[i] = np.sum(amp_XdP[i,:]) / np.sum(amp_XnaP[i])
        SNR[i] = 10*np.log10(SNR[i]) #convert snr to decibels
        #SNR[i] = (amp_Xda[i])/(amp_Xna[i])
        if SNR[i] < -5:
            alpha[i] = alpha0+(3/4)
        elif SNR[i] >= -5 and SNR[i] < 20:
            alpha[i] = alpha0-(3/20*SNR[i])
        elif SNR[i] >= 20:
            alpha[i] = alpha0-3 
        print(alpha)
        #amp_Xp[i,:] = (amp_Xd[i,:] ** p) - ((alpha[i])*(amp_Xna[i] ** p)) #Hassani 2011
        amp_Xp[i,:] = (amp_Xd[i,:] ** p) - (amp_Xna[i] ** p) # Fukane 2011
        for j in range(0,n):
#           if amp_Xp[i,j] > (beta)*(amp_Xna[i] ** p): #Hassani 2011
            if amp_Xp[i,j] > (beta+alpha[i])*(amp_Xna[i] ** p): # Fukane 2011
                amp_Xp[i,j] = amp_Xp[i,j]
            else:
                amp_Xp[i,j] = (beta)*(amp_Xna[i] ** p)
        #np.savetxt('amp_Xp.out', amp_Xp, delimiter=',', newline="\n")   # X is an array
    amp_Xp = amp_Xp ** (1/p)             # square root has to come AFTER the negatives are removed
    return amp_Xp   

def nonlin_subtraction(amp_Xd, amp_Xn): 
    m, n = amp_Xd.shape 
    SNR = np.zeros((m))
    alpha = np.zeros((m))
    amp_Xp = np.zeros((m,n))
    amp_Xda = np.mean(amp_Xd,axis=1)
    amp_Xna = np.mean(amp_Xn,axis=1)
    beta=0.1 #Fukane 2011
    N = np.max(amp_Xna)
    gamma = 0.5 # scaling factor, r in Fukane, gamma in Lockwood, might be the smoothing factor in Upadhyay taken as 0.5
    SNR = amp_Xda/amp_Xna
    print(N)
    print(SNR)
    alpha = 1 / (1 + (gamma*SNR))
    for i in range(0,m):
        amp_Xp[i,:] = (amp_Xd[i,:]) - ((alpha[i])*(N)) #Fukane 11
        for j in range(0,n):
            if amp_Xd[i,j] > (alpha[i])*(N) + ((beta)*(amp_Xna[i])):
                amp_Xp[i,j] = amp_Xp[i,j]
            else:
                amp_Xp[i,j] = (beta)*(amp_Xd[i,j])
        #np.savetxt('amp_Xp.out', amp_Xp, delimiter=',', newline="\n")   # X is an array
    return amp_Xp   


def mulban_subtraction(tro, amp_Xd, amp_Xn): #mainly from Upadhyay and Karmakar 2013
    p = 2 
    SNR_min = -5 #db
    SNR_max = 20 #db
    alpha_min = 1
    alpha_max = 5
    beta=0.002 #in Kamath and Loizou
    FS = tro.stats.sampling_rate
    #delta :tweaking factor that can be individually set for each frequency band to customize the noise removal properties.
    for i in range(0,m):
    NEED a frequency scale conversion here I guess.
        w [0.09, 0.25, 0.32, 0.25, 0.09]. #what about i? 
        if freqs =< 1: #khz #in Kamath and Loizous
            delta = 1
        elif freqs > 1 and freqs <= ((FS/2) - 2):
            delta = 2.5
        elif freqs > ((FS/2) - 2):
            delta = 1.5
        SNR[i] = np.sum(amp_XdP[i,:]) / np.sum(amp_XnaP[i])
        SNR[i] = 10*np.log10(SNR[i]) #convert snr to decibels
        if SNR[i] < SNR_min: #this is very similar to over subtraction alpha constraints.....
            alpha = alpha_max
        elif SNR[i] >= SNR_min and SNR[i] <= SNR_max:
            alpha = alpha_max + ((SNR[i] - SNR_min)*((alpha_min - alpha_max)/(SNR_max - SNR_min)))
        elif SNR[i] > SNR_max:
            alpha = alpha_min
        (amp_Xp[i,:] ** p) = (amp_Xd[i,:] ** p) - (alpha)*(delta)(amp_Xna[i] ** p) 
        for j in range(0,n):
            if (amp_Xp[i,j] ** p) > (beta)*(amp_Xna[i] ** p): 
                (amp_Xp[i,:] ** p) = (amp_Xp[i,:] ** p)
            else:
                (amp_Xp[i,:] ** p) = (beta)*(amp_Xd[i,:] ** p)
























