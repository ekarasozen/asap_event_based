# CREATE SUBTRACTION PLOT
#
#assuming D = 1 to simplyfy plotting
#if Y > alpha * D
#     X = Y - alpha * D
#else
#    X = beta * D

import matplotlib.pyplot as plt                                                                                                                                                               
import numpy as np        


# SET UP VARIABLES
Y = np.logspace(-2,3,200)                                                                                                                                                                   
D = 1
beta = 0.25
rho = Y/D
alpha = 14.24/(1+0.01*(rho**4))       #alpha4b
#alpha = 4.16/(1+0.01*rho)            #alpha4b


# PERFORM THE TWO PARTS OF SUBTRACION
X = beta * D * np.ones(Y.shape)
high_snr = Y >= alpha
X[high_snr] = Y[high_snr] - alpha[high_snr]
R = Y - X

 
 
# MAKE PLOT                                                                                                                                                      
fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize=(6,9))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
ax1.plot(rho,Y,color='k',label='Y',lw=2)  
ax1.stackplot(rho, X, R, labels=('X','R'), colors={'gray','r'})
ax1.set_xscale('log') 
crossover = np.logspace(-2,3,50)       
ax1.legend(loc='upper right')
ax1.grid()
ax1.set_ylabel('amplitude (normalized by 1/D)')                                                                                                                                                                       
ax1.set(xlim=[0.3,70])
ax1.set(ylim=[0,70])
ax1.set_title('14.24/(1+0.01*rho**4) & beta=0.25')                                                                                                                                                                                                                                                                                                                                

ax2.plot(rho,Y,color='k',label='Y',lw=2)  
ax2.stackplot(rho, X, R, labels=('X','R'), colors={'gray','r'})
ax2.set_xscale('log')
ax2.legend(loc='upper right')
ax2.set_ylabel('amplitude (normalized by 1/D)')                                                                                                                                                                       
ax2.set(xlim=[0.3,70])
ax2.set(ylim=[0,5])
ax2.grid()
         
ax3.stackplot(rho, R, labels=('R'), colors={'r'})
ax3.set_xscale('log')
ax3.legend(loc='upper right')
ax3.set_ylabel('amplitude (normalized by 1/D)')                                                                                                                                                                       
ax3.set_xlabel('rho (same as Y/D)')
ax3.set(xlim=[0.3,70])
ax3.set(ylim=[0,5])
ax3.grid()
ax3.set_title('How much amplitude is actually being subtracted from Y')                                                                                                                                                                                                                                                                                                                                
                                                                                                                                                          
plt.show()        


