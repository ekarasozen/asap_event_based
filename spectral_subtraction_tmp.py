import numpy as np
#Xn: signal with noise
#N: noise estimate - known for now (D in Fukane and Sahare)
    
def simple_subtraction(amp_Xd, amp_Xn, p): 
    #if p = 1; amplitude/magnitude subtraction
    #if p = 2; power subtraction
    minamp = 1000                # I picked this value randomly
    m, n = amp_Xd.shape 
    amp_Xp = np.zeros((m,n))
    amp_Xna = np.mean(amp_Xn,axis=1)
    for i in range(0,n):
        amp_Xp[:,i] = (amp_Xd[:,i] ** p) - (amp_Xna ** p)
    result = np.where(amp_Xp<minamp**p)  # find low or negative values
    amp_Xp[result] = minamp**p           # set minimum value
    amp_Xp = amp_Xp ** (1/p)             # square root has to come AFTER the negatives are removed
    return amp_Xp 


