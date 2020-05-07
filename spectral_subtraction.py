import numpy as np
from obspy import Stream
   
   
    
##########################################################
# SIMPLE OVER-SUBTRACTION WITH FIXED ALPHA
# SNR is not used, but is included here for consistency
# alpha is set as a constant equal to alpha0


def constant_subtraction(amp_Xd, amp_Xn, p, alpha0, beta): 
    m, n = amp_Xd.shape 
    SNR = np.zeros((1,n))
    alpha = np.zeros((1,n))
    amp_Xp = np.zeros((m,n))
    amp_XpP = np.zeros((m,n))
    amp_Xda = np.mean(amp_Xd,axis=1)
    amp_XdP = amp_Xd ** p
    amp_Xna = np.mean(amp_Xn,axis=1)
    amp_XnaP = amp_Xna ** p
    for i in range(0,n):
        SNR[:,i] = np.sum(amp_XdP[:,i]) / np.sum(amp_XnaP)
        SNR[:,i] = 10*np.log10(SNR[:,i])
        alpha[:,i] = alpha0       #hold alpha constant
        amp_XpP[:,i] = amp_XdP[:,i] - (alpha[:,i] * amp_XnaP)
        belowthreshold = amp_XpP[:,i] < beta*amp_XnaP
        amp_XpP[belowthreshold,i] = beta*amp_XnaP[belowthreshold]
        amp_Xp[:,i] = amp_XpP[:,i] ** (1/p)  
    return amp_Xp, SNR, alpha   
    
    
def simple_subtraction(amp_Xd, amp_Xn, p, alpha0, beta): 
    m, n = amp_Xd.shape 
    SNR = np.zeros((1,n))
    alpha = np.zeros((1,n))
    amp_Xp = np.zeros((m,n))
    amp_XpP = np.zeros((m,n))
    amp_Xda = np.mean(amp_Xd,axis=1)
    amp_XdP = amp_Xd ** p
    amp_Xna = np.mean(amp_Xn,axis=1)
    amp_XnaP = amp_Xna ** p
    for i in range(0,n):
        SNR[:,i] = np.sum(amp_XdP[:,i]) / np.sum(amp_XnaP)
        SNR[:,i] = 10*np.log10(SNR[:,i])
        alpha[:,i] = alpha0       #hold alpha constant
        amp_XpP[:,i] = amp_XdP[:,i] - alpha[:,i]*(amp_XnaP)  
        belowthreshold = amp_XpP[:,i] < beta*amp_XdP[:,i]
        amp_XpP[belowthreshold,i] = beta*amp_XdP[belowthreshold,i]
        amp_Xp[:,i] = amp_XpP[:,i] ** (1/p)  
    return amp_Xp, SNR, alpha 


def over_subtraction(amp_Xd, amp_Xn, p): 
    #alpha over subtraction factor, value greater than or equal to 1
    #beta spectral floor parameter value between 0 to 1
    m, n = amp_Xd.shape 
    SNR = np.zeros((1,n))
    alpha = np.zeros((1,n))
    amp_Xp = np.zeros((m,n))
    amp_Xda = np.mean(amp_Xd,axis=1)
    amp_XdP = amp_Xd ** p
    amp_Xna = np.mean(amp_Xn,axis=1)
    amp_XnaP = amp_Xna ** p
    alpha0=4 #vary between 3-6 (Beruiti et al'79), normally taken as 4 (Kamath & Loizou'02)
    beta=0.2 #should be between 0-1.
    #result = np.transpose(np.where(np.locgical_and(freqs_d>0.5,freqs_d<10)))
    for i in range(0,n):
        SNR[:,i] = np.sum(amp_XdP[:,i]) / np.sum(amp_XnaP)
        SNR[:,i] = 10*np.log10(SNR[:,i]) #convert snr to decibels
        if SNR[:,i] < -5:
            alpha[:,i] = alpha0+(3/4)
        elif SNR[:,i] >= -5 and SNR[:,i] < 20:
            alpha[:,i] = alpha0-(3/20*SNR[:,i])
        elif SNR[:,i] >= 20:
            alpha[:,i] = alpha0-3 
            #amp_Xp[i,:] = (amp_Xd[i,:] ** p) - ((alpha[i])*(amp_Xna[i] ** p)) #Hassani 2011
        amp_Xp[:,i] = (amp_Xd[:,i] ** p) - (alpha)*(amp_Xna ** p) # Berouti
#        amp_Xp[:,i] = (amp_Xd[:,i] ** p) - (amp_Xna ** p) # Fukane 2011
        for j in range(0,m):
            if amp_Xp[j,i] > (beta+(alpha[0,i]))*(amp_Xna[j] ** p): # Fukane 2011
                amp_Xp[j,i] = amp_Xp[j,i]
            else:
                amp_Xp[j,i] = (beta)*(amp_Xna[j] ** p)
        amp_Xp[:,i] = amp_Xp[:,i] ** (1/p)  
    return amp_Xp, SNR, alpha   

def nonlin_subtraction(amp_Xd, amp_Xn): #mainly from Lockwood
    m, n = amp_Xd.shape 
    alpha = np.zeros((n))
    rho = np.zeros((m,n))
    phi = np.zeros((m,n))
    amp_Xp = np.zeros((m,n))
    amp_Xds = np.zeros((m,n))
    amp_Xda = np.mean(amp_Xd,axis=1)
    amp_Xna = np.max(amp_Xn,axis=1)
    #amp_Xna = np.mean(amp_Xn,axis=1)
    beta=0.1 #Fukane 2011
    gamma = 0.5 # scaling factor, r in Fukane, gamma in Lockwood, is this the smoothing factor in Upadhyay taken as 0.5?
    muy = 0.3 #should be between 0.1-0.5
    mud = 0.7 #should be between 0.5-0.9
    for i in range(0,n):
#        alpha = np.max(amp_Xna) # Lockwood calculates this for the last 40 frames, not sure this is necessary in our case - yet.  
        alpha = amp_Xna # I think this makes more sense
        for j in range(0,m):
            amp_Xds[j,i] = (muy)*amp_Xd[j,(i-1)]+(1-muy)*amp_Xd[j,i] #smoothed estimate of degraded signal, same should be for the noise.
            rho[j,i] = amp_Xds[j,j]/amp_Xna[j]
            phi[j,i] = alpha[j] / (1 + (gamma*rho[j,i]))
            amp_Xp[j,i] = (amp_Xds[j,i]) - phi[j,i]
            if amp_Xds[j,i] > phi[j,i] + ((beta)*(amp_Xna[j])):
                amp_Xp[j,i] = amp_Xp[j,i]
            else:
                amp_Xp[j,i] = (beta)*(amp_Xds[j,i])
    np.savetxt('amp_Xna.out', amp_Xna, delimiter=',', newline="\n")   # X is an array
    return amp_Xp, phi, alpha, rho   


def mulban_subtraction(amp_Xd, amp_Xn, tro, freqs): #mainly from Upadhyay and Karmakar 2013
#find a better way to do the powers
    p = 2 
    m, n = amp_Xd.shape
    SNR = np.zeros((1,n))
    alpha = np.zeros((1,n))
    delta = np.zeros((m))
    amp_Xp = np.zeros((m,n))
    amp_XpP = np.zeros((m,n))
    amp_XdP = amp_Xd ** p
    amp_Xna = np.mean(amp_Xn,axis=1)
    amp_XnaP = amp_Xna ** p
    beta=0.002 #in Kamath and Loizou
    FS = tro.stats.sampling_rate
    #delta :tweaking factor that can be individually set for each frequency band to customize the noise removal properties.
    for i in range(0,n):
        for j in range(0,m):
             #if freqs[j] > 9 and freqs[j] <= 10:
             #   idx = np.transpose(np.where(np.logical_and(freqs>9,freqs<=10)))
             if freqs[j] <= 1:
                idx = np.transpose(np.where(np.logical_and(freqs>0,freqs<=1)))
                delta = 1/1000
                SNR[:,i] = np.sum(amp_XdP[idx,i]) / np.sum(amp_XnaP[idx])
                SNR[:,i] = 10*np.log10(SNR[:,i])
                #print("SNR", SNR[:,i]) 
                if SNR[:,i] < -5: 
                    alpha[:,i] = 5 # or 4.75
                elif SNR[:,i] >= -5 and SNR[:,i] <= 20:
#                    alpha[:,i] = alpha_max + ((SNR[:,i] - SNR_min)*((alpha_min - alpha_max)/(SNR_max - SNR_min))) #took this somewhere else. 
                    alpha[:,i] = 4 - ((3/20)*SNR[:,i]) #in Kamath and Loizous
                elif SNR[:,i] > 20:
                    alpha[:,i]= 1
                amp_Xp[idx,i] = (amp_Xd[idx,i] ** p) - (alpha[:,i])*(delta)*(amp_Xna[idx] ** p) # NEED TO TEST AFTER THIS
                noi = amp_Xp[idx,i].size #number of index
                for idx in range(0,noi):
                    #if amp_XpP[k,i] > (beta)*(amp_Xd[k,i] ** p): #this is different in Kamath and Loizous but same with Loizous's book  
                    if amp_Xp[idx,i] > 0: #from Kamath and Loizous
                        amp_Xp[idx,i] = amp_Xp[idx,i]
                    else:
                        amp_Xp[idx,i] = (beta)*(amp_Xd[idx,i] ** p)
                amp_Xp[idx,i] = amp_Xp[idx,i] + (0.05*(amp_Xd[idx,i] ** p)) #a small amount of noisy spectrum is introduced back
                amp_Xp[idx,i] = amp_Xp[idx,i] ** (1/p)
             elif freqs[j] > 1 and freqs[j]<=((FS/2)-2): #khz #in Kamath and Loizous
                idx = np.transpose(np.where(np.logical_and(freqs>1,freqs<=((FS/2)-2))))
                delta = 2.5
                SNR[:,i] = np.sum(amp_XdP[idx,i]) / np.sum(amp_XnaP[idx])
                SNR[:,i] = 10*np.log10(SNR[:,i]) 
                if SNR[:,i] < -5: 
                    alpha[:,i] = 5 # or 4.75
                elif SNR[:,i] >= -5 and SNR[:,i] <= 20:
#                    alpha[:,i] = alpha_max + ((SNR[:,i] - SNR_min)*((alpha_min - alpha_max)/(SNR_max - SNR_min))) #took this somewhere else. 
                    alpha[:,i] = 4 - ((3/20)*SNR[:,i]) #in Kamath and Loizous
                elif SNR[:,i] > 20:
                    alpha[:,i]= 1
                amp_Xp[idx,i] = (amp_Xd[idx,i] ** p) - (alpha[:,i])*(delta)*(amp_Xna[idx] ** p) # NEED TO TEST AFTER THIS
                print(amp_Xp[idx,i], delta)
                noi = amp_Xp[idx,i].size #number of index
                for idx in range(0,noi):
                    if amp_Xp[idx,i] > 0: #from Kamath and Loizous
                        amp_Xp[idx,i] = amp_Xp[idx,i]
                    else:
                        amp_Xp[idx,i] = (beta)*(amp_Xd[idx,i] ** p)
                amp_Xp[idx,i] = amp_Xp[idx,i] + (0.05*(amp_Xd[idx,i] ** p)) #a small amount of noisy spectrum is introduced back
                amp_Xp[idx,i] = amp_Xp[idx,i] ** (1/p)  
             elif freqs[j] > ((FS/2)-2): #khz #in Kamath and Loizous
                idx = np.transpose(np.where(np.logical_and(freqs>((FS/2)-2),freqs<=20)))
                delta = 1.5/1000
                SNR[:,i] = np.sum(amp_XdP[idx,i]) / np.sum(amp_XnaP[idx])
                SNR[:,i] = 10*np.log10(SNR[:,i]) 
                if SNR[:,i] < -5: 
                    alpha[:,i] = 5 # or 4.75
                elif SNR[:,i] >= -5 and SNR[:,i] <= 20:
#                    alpha[:,i] = alpha_max + ((SNR[:,i] - SNR_min)*((alpha_min - alpha_max)/(SNR_max - SNR_min))) #took this somewhere else. 
                    alpha[:,i] = 4 - ((3/20)*SNR[:,i]) #in Kamath and Loizous
                elif SNR[:,i] > 20:
                    alpha[:,i]= 1
                amp_Xp[idx,i] = (amp_Xd[idx,i] ** p) - (alpha[:,i])*(delta)*(amp_Xna[idx] ** p) # NEED TO TEST AFTER THIS
                noi = amp_Xp[idx,i].size #number of index
                for idx in range(0,noi):
                    if amp_Xp[idx,i] > 0: #from Kamath and Loizous
                        amp_Xp[idx,i] = amp_Xp[idx,i]
                    else:
                        amp_Xp[idx,i] = (beta)*(amp_Xd[idx,i] ** p)
                amp_Xp[idx,i] = amp_Xp[idx,i] + (0.05*(amp_Xd[idx,i] ** p)) #a small amount of noisy spectrum is introduced back
                amp_Xp[idx,i] = amp_Xp[idx,i] ** (1/p)  
        #print(np.sum(amp_Xd[j,i]))
    #amp_Xp = amp_Xp ** (1/p)             # square root has to come AFTER the negatives are removed
    np.savetxt('SNR.out', SNR, delimiter=',', newline="\n")  
    np.savetxt('alpha.out', alpha, delimiter=',', newline="\n")   
    np.savetxt('amp_Xp.out', amp_Xp, delimiter=',', newline="\n")  
    np.savetxt('amp_Xd.out', amp_Xd, delimiter=',', newline="\n")   
    np.savetxt('amp_Xna.out', amp_Xna, delimiter=',', newline="\n")  
    np.savetxt('freqs.out', freqs, delimiter=',', newline="\n")  
    return amp_Xp
