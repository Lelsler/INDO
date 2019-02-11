#### Laura Elsler: April 2017
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.integrate import quad
import matplotlib.gridspec as gridspec
import seaborn as sns
import scipy

### Parameters #################################################################
tmax = 50 # model runs
# Monsoon seasons
a = 10 #
b = 10 #
d = 10 #
# species
g = 0.2 #
k = 100 #
# effort
alpha =
beta = 1. #
gamma = .25 #
c = 0.4 # variable cost of fishing
# catch
q = 0.03 #
# trader rules
f =
# reef health
m =
r =
u =
# community rules
h =
j =
l =

### Variables ##################################################################
# monsoon seasons
M = np.zeros(tmax) # monsoon seasons
# species
N = np.zeros(tmax) # biomass
# effort
E = np.zeros(tmax) # effort
# catch
C = np.zeros(tmax) # catch species
# trader rules & relations
R_t = np.zeros(tmax) # trader rules/demand
# reef health
H = np.zeros(tmax) # reef health
# income and price for fisher
P = np.zeros(tmax) # price
I = np.zeros(tmax) # fisher income
# community rules & relations
R_c = np.zeros(tmax) # community rules

### Initial values #############################################################
M[0] = 100
N[0] = 0.5
E[0] = 0.5
C[0] = 0.5
R_t[0] = 2
H[0] = 2
P[0] = 0.5
I[0] = 0.5
R_c[0] = 2

### Define Model ###############################################################
def model(a,b,d,g,k,alpha,beta,gamma,c,q,f,m,r,u,h,j,l):
    for t in np.arange(0,tmax-1):
        # monsoon
        M[t+1] = tau[t]= a + b *np.cos(t) + d *np.sin(t)
        print M[t], "M"

        # population logistic growth dynamics
        N[t+1] = N[t] + g*N[t] * (1- (N[t]/(k* H[t]))) -C[t]
        print N[t], "N"

        # effort
        E[t+1] = E[t]*np.exp(alpha*(gamma**(1/beta))*((C[t])**((beta-1)/beta))-c*E[t])
        print E[t], "E"

        # catch
        C[t] = q*E[t]*N[t]
        print C[t], "C"

        # trader rules
        R_t[t+1] = f * np.exp(-C[t])
        print R_t[t], "Rt"

        # reef health
        H[t+1] = H[t]+ (m/(r+ u* R_t[t]* E[t]))
        print H[t], "H"

        # price
        P[t] = gamma* (C[t])**(-beta)
        if P[t] >= 10:
            P[t]= 10
        if P[t] < 1:
            P[t]= 1
        print P[t], "P"

        # income
        I[t+1] = C[t]* P[t]- E[t]*c

        # community rules
        R_c[t+1] = h* np.exp(j* C[t]+ l* H[t])
        print R_c[t], "Rc"

    return M, N, E, C, R_t, H, P, I, R_c # output variables

##### Run the model ############################################################

OUT1 = np.zeros(I.shape[0])
OUT2 = np.zeros(I.shape[0])
OUT3 = np.zeros(I.shape[0])
OUT4 = np.zeros(I.shape[0])
OUT5 = np.zeros(I.shape[0])
OUT6 = np.zeros(I.shape[0])
OUT7 = np.zeros(I.shape[0])
OUT8 = np.zeros(I.shape[0])
OUT9 = np.zeros(I.shape[0])

for i in np.arange(0,tmax):
        M, N, E, C, R_t, H, P, I, R_c = model(a,b,d,g,k,alpha,beta,gamma,c,q,f,m,r,u,h,j,l)
        OUT1[i]= M[i]
        OUT2[i]= N[i]
        OUT3[i]= E[i]
        OUT4[i]= C[i]
        OUT5[i]= R_t[i]
        OUT6[i]= H[i]
        OUT7[i]= P[i]
        OUT8[i]= I[i]
        OUT9[i]= R_c[i]

# np.save("./Desktop/RQ3_P.npy", OUT7)

#####! PLOT ORIGINAL MODEL
fig = plt.figure()
plt.plot(N)
plt.plot(C)
plt.xlim(0,tmax-3)
plt.title("Test",fontsize=17)
plt.xlabel("Time period",fontsize=15)
plt.ylabel("Species biomass",fontsize=15)
plt.legend(['biomass', 'catch'], loc='best')
#fig.savefig('fish.png',dpi=300)
plt.show()
