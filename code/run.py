# -*- coding: utf-8 -*-
'''
This file contains all the problem information.
'''
import numpy as np
from scipy.stats import geom, binom, gamma
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D  

import matplotlib
font = {'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 13}

matplotlib.rc('font', **font)

num_seats_def = 150
prng = np.random.RandomState(1) #Pseudorandom number generator
demand_par_a_def = 60.2#prng.randint(100, size=1).astype(float)
demand_par_b_def = 10.0#prng.randint(5, size=1).astype(float)
demand_par_c_def = 1.0
epsilons_support_def = 10
pmin_def = 1.0

p=0.3
prob_eps_rem=(1-geom.cdf(epsilons_support_def+1, p))/(epsilons_support_def*2+1)
prob_eps = []
for i in range(0, epsilons_support_def+1):
    if i ==0:
        prob_eps.append(geom.pmf(i+1, p) + prob_eps_rem)
    else:
        prob_eps.append(geom.pmf(i+1, p)/2 + prob_eps_rem)
        
prob_eps_rev = prob_eps.copy()
prob_eps_rev.reverse()
prob_eps_rev.pop()
prob_eps_def= prob_eps_rev + prob_eps 

p_r_min_def = 0.8 * pmin_def
gamma_def = 0.9
num_stages_def = 12

class MDP():
    '''
    Create problem parameter info, including number of seets,
    and the price-dependent-demand model.
    The demand models are assumed linear in price, of the form,
    D(p) = a - bp. Here a and b iare positive scalars.
    Epsilons are the additive demand noise. The full demand model is 
    D_t(p_t) = a - b p_t + epsilon_t. 
    '''
    def __init__(self, num_seats = num_seats_def ,\
                 demand_par_a = demand_par_a_def,\
                 demand_par_b = demand_par_b_def,\
                 epsilons_support = epsilons_support_def,\
                 pmin = pmin_def,\
                 prob_eps   = prob_eps_def,\
                 p_r_min = p_r_min_def,\
                 gamma = gamma_def,
                 num_stages = num_stages_def,
                 demand_par_c = demand_par_c_def,
                 ):
        
        self.t = 1
        self.num_seats = num_seats
        self.observation = self.num_seats
        self.nS = self.num_seats+1
        self.states = np.arange(self.num_seats+1)
        self.demand_par_a = demand_par_a
        self.demand_par_b = demand_par_b
        self.demand_par_c = demand_par_c
        self.epsilons_support = epsilons_support
        self.epsilons = np.arange(- self.epsilons_support, self.epsilons_support+1)
        self.nE = self.epsilons_support*2 +1
        self.pmin = pmin
        self.p_r_min = p_r_min
        self.pmax = (self.demand_par_a - self.epsilons_support +\
                     self.demand_par_c * self.p_r_min)/ self.demand_par_b
#        self.pmax = (self.demand_par_a - self.epsilons_support)/(self.demand_par_b -self.demand_par_c)
        self.p_r_max = 1*self.pmax
        self.dmin = self.D(self.pmax, self.p_r_min)
        self.dmax = self.D(self.pmin, self.p_r_max)
        self.d_range = self.dmax - self.dmin + 1
        self.prob_eps = prob_eps
        delta = 0.1
        self.p_range = np.round(np.arange(self.pmin,self.pmax+1e-3, delta), 1)
        self.p_r_range = np.round(np.arange(self.p_r_min,self.p_r_max+1e-3, delta), 1)
        self.prob_r_v  = np.array([((p_r -0.9*self.p_r_min)/
            (self.p_r_max - 0.8*self.p_r_min))*0.95 for p_r in self.p_r_range])
        self.prob_r_ = dict(zip( self.p_r_range,self.prob_r_v ))
        
        temp=[]
        self.prob_r_dic={}
        n = np.arange(self.num_seats+1)
        for k in n:
            for  p in self.p_r_range:
                temp.append(binom.pmf(np.arange(0,k+1),k,self.prob_r_[p]))
                self.prob_r_dic[k,p]= temp[-1]
        self.prob_r = np.array(temp)
        
        self.actions = np.array([(p, p_r) for p in self.p_range\
                                 for p_r in self.p_r_range])
    
        self.nA = self.actions.shape[0]
        self.f=np.zeros((self.nS, self.nA, self.nE))
        self.gamma = gamma
        self.num_stages = num_stages_def
        
    def backward_step(self, t, s, p, pr, value):
        ''' Calculates one backward step 
        '''
        if t == self.num_stages:
            d = self.D(p,pr)
#            print('d=',d)
#            print('p=',p)
#            print('pr=',pr)
            E = np.tile(self.epsilons,self.num_seats-s+1)
            R = np.arange(self.num_seats-s+1)
            R = np.repeat(R, self.nE)
            S = np.ones(self.nE * (self.num_seats-s+1)) * s
            D = np.ones(self.nE * (self.num_seats-s+1)) * d
            R_prob = np.repeat(self.prob_r_dic[self.num_seats-s,pr], self.nE)
            E_prob = np.tile(np.array(self.prob_eps), self.num_seats-s+1)
            W = np.minimum(S, D + E)
            
#            print(W)
            NS_DIC[s,d,pr] = (S + R - W).astype(int)
#            print(NS_DIC[s,d,pr][np.where(NS_DIC[s,d,pr]==101)])
            V_part = self.gamma * value[t, NS_DIC[s,d,pr]]
            RW_DIC[s,d,pr] = p * W - pr * R 
            PROB_DIC[s,d,pr] = R_prob * E_prob
            V = np.dot(PROB_DIC[s,d,pr], RW_DIC[s,d,pr] + V_part)
        else:
            d = self.D(p,pr)
            V_part = self.gamma * value[t, NS_DIC[s,d,pr]]
            V = np.dot(PROB_DIC[s,d,pr], RW_DIC[s,d,pr] + V_part)
            
        return V
    
    def BI(self):
        '''
        Backward Induction
        '''
        global RW_DIC , PROB_DIC, NS_DIC
        RW_DIC = {}
        PROB_DIC = {}
        NS_DIC = {}
        value = np.zeros((self.num_stages+1, self.nS))
        policy = np.zeros((self.num_stages+1, self.nS, 2))
        for t in range(self.num_stages -1, -1, -1):
            print('t=',t)
            for s in self.states:
                v = []
                for a_idx in range(self.nA):
                    a = self.actions[a_idx]
                    v.append(self.backward_step(t+1, s, a[0], a[1], value))
                vmax = np.max(v)
                idx = np.where(vmax==v)#np.array([i for i,x in enumerate(v) if abs(vmax - x)<10e-8])
                a_max = np.max(self.actions[idx], axis=0)
#                if t==11 :
#                    print('v[0]=',v[0],'v[-1]', v[-1],'vmax',vmax,'idx',idx, '\n')
            
#                idx = np.argmax(v)    
#                policy[t,s,:] = [self.P(self.actions[idx][0], self.actions[idx][1]),
#                                          self.actions[idx][1]]
                policy[t,s,:] = [a_max[0],a_max[1]]
                value[t,s] = vmax
        self.value = value
        self.policy = policy
                
                    
        
        
    def D(self, p, pr):
        '''
        Deterministic demand function: returns a demand scalar (rounded to the 
        nearest integer) coresponding to the demand for the 
        given price input
        '''
        d= np.rint(self.demand_par_a - self.demand_par_b*p + self.demand_par_c*pr).astype(int)
#        d= self.demand_par_a - self.demand_par_b*p
        return d
 
    def P(self, d, pr):
        '''
        Returns a price vector of each station for the given vector demand input
        '''
        p=(self.demand_par_a - d +self.demand_par_c*pr)/self.demand_par_b
        return p         
  
    def R(self, k,n, p_r):
        ''' Returns the probability of k out of n cutomers returning their tickets '''
        prob = ((p_r -0.9*self.p_r_min)/(self.p_r_max - 0.8*self.p_r_min))*0.95
        return binom.pmf(k,n,prob)
    
    def step(self,a):
        epsilon = prng.choice(self.epsilons , p=self.prob_eps)
        r = prng.choice(self.num_seats-self.observation+1,
                        p=self.prob_r_dic[self.num_seats-self.observation,a[1]])
        self.observation += r - np.minimum(self.observation, a[0] + epsilon)
        self.observation = self.observation.astype(int)
        self.t +=1
        return r

        
    def reset(self):
        self.observation = self.num_seats
        self.t = 1
    
    def simulate_policy(self, policy):
        pol =policy.copy()
        times = 5000
        m = np.zeros((times,self.num_stages))
        rp = np.zeros((times,self.num_stages))
        S = np.zeros((times,self.num_stages+1))
        NR = np.zeros((times,self.num_stages+1))
        for i in range(times):
            self.reset()
            print('i=',i)
            p = []
            pr = []
            s = []
            nr = []
            s.append(self.observation)
            nr.append(0)
            while self.t <= int(self.num_stages):
#                print('t=',self.t)
                
                action =  pol[self.t-1,int(self.observation)].copy()
                p.append(action[0])
                pr.append(action[1])
                
                action[0] = self.D(action[0],action[1])
                nr.append(self.step(action))
                s.append(self.observation)
                
            m[i,:]=p
            rp[i,:]=pr
            S[i,:]=s
            NR[i,:]=nr
        mean_p =np.mean(m, axis=0)  
        mean_pr =np.mean(rp, axis=0) 
        mean_s =np.mean(S, axis=0)
        mean_nr = np.mean(NR, axis=0)
        return mean_p , mean_pr, mean_s, mean_nr, m, rp, S, NR   
 
z1 = []
z2 = []
for i in range(m.num_stages):
    for j in range(m.num_seats+1):
        z1.append(m.policy[i,j][0])
        z2.append(m.policy[i,j][1])
        


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.arange(1,m.num_stages+1)
y = np.arange(m.num_seats+1)
X, Y = np.meshgrid(y, x)
z1 = np.array(z1)
Z1 = z1.reshape(Y.shape)
z2 = np.array(z2)
Z2 = z2.reshape(Y.shape)

ax.plot_surface(Y, X, Z1, label='$p_t$')
ax.plot_surface(Y, X, Z2, label='$p_{rt}$')
plt.xlabel('Stage')
ax.set_ylabel('No. of remaining seats')
ax.set_zlabel('Price units')
ax.set_title('Case 2: Policy Plots')
plt.legend(loc='best') 
#           
for t in range(m.num_stages):
    plt.plot(m.policy[t,:][:,0], label='$p_'+str(t)+'$')
    plt.plot(m.policy[t,:][:,1], label='$p_r'+str(t)+'$')
plt.xlabel('Number of remaining seats')
plt.ylabel('Price unit')
#plt.title('Policy Plots')
plt.legend(loc='best') 
plt.show
#
#
mean_p , mean_pr, mean_s, mean_nr, mm, rp, S, NR =m.simulate_policy(m.policy)
#
#plt.plot(np.arange(1, len(mean_p)+1),mean_p, label='Mean $p_{t}$')
#plt.plot(np.arange(1, len(mean_pr)+1),mean_pr, label='Mean $p_{rt}$')
#plt.xlabel('Stage')
#plt.ylabel('Price unit')
#plt.title('Case 2: Average Optimal Price')
#plt.legend(loc='best')
##
##
#plt.plot(np.arange(0,m.num_seats),m.value[0,:])
#plt.xlabel('Number of remaining seats')
#plt.ylabel('Value')
#plt.title('Case 2: Optimal Value Funtion')
#plt.legend(loc='best')
#
#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.plot3D(np.arange(m.num_stages), m.states, m.policy[:,:][:,0] )
#x = []
#y = []
#z1 = []
#z2 = []
#for t in range(12):
#    for s in range(101):
#        x.append(t)
#        y.append(s)
#        z1.append(m.policy[t,s][0])
#        z2.append(m.policy[t,s][1])
#
#  
#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.scatter(x,y,z1)
#ax.plot3D(x,y,z1)

#plt.plot(n.value[0,:])
#plt.plot(m.value[0,:])
#
#plt.plot(m.policy[0,:][:,0])
#plt.plot(m.policy[0,:][:,1])
#plt.plot(n.policy[0,:][:,0])
#plt.plot(n.policy[0,:][:,1])