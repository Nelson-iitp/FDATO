    
#~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+
# DEFINE WORKFLOW & INFRA CONSTANTS 
    
#        wfD   range of Data units in a workflow
#        wfV   range of Task units in a workflow
#        wfL   range of initial task locations in a workflow
#        
#        E   no of edge servers
#        C   no of cloud servers
#        A   no of actions possible = 1+E+C
#        
#        DR  Data Rate :                  Time Delay in moving a unit of data b/w two locations
#        DE  Data Energy :                Energy consumed in handling per unit of Data
#        VR  Task Rate :                  Time Delay in executing a unit of task
#        VE  Task Energy :                Energy consumed in executing a unit of task

#        randomize()   a function that can uniformly randomize infra constants DR VR DE VE

#~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+~~+


import numpy as np
import matplotlib.pyplot as plt
from math import floor
from scipy.sparse.csgraph import floyd_warshall

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Basic Shared functions
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def effective_bandwidth(M):
    """ function for finding shortest path and effective Data Rate (bandwidth) b/w edge or cloud servers 
        return dist_matrix """
    lm = len(M)
    for i in range(lm):
        for j in range(i+1, lm):
            if M[i,j]!=0:
                M[i,j] = 1/M[i,j]
            M[j,i] = M[i,j]
    dist_matrix, predecessors = floyd_warshall(csgraph=M, 
                directed=False,return_predecessors=True)
    return dist_matrix #<--- this turns the diagonals to inf  #print(dist_matrix)
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def int2base(num, base, digs):
    """ convert base-10 integer to base-n array of fixed no. of digits 
    return array (of length = digs)"""
    res = np.zeros(digs, dtype=np.int32)
    q = num
    for i in range(digs):
        res[i]=q%base
        q = floor(q/base)
    return res
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def base2int(arr, base):
    """ convert array from given base to base-10  --> return integer"""
    res = 0
    for i in range(len(arr)):
        res+=(base**i)*arr[i]
    return res
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def strA(arr, start="[", sep=",", end="]"):
    """ returns a string representation of an array/list for printing """
    res=start
    for i in range(len(arr)):
        res+=str(arr[i])+sep
    return res + end
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=   
def strD(arr, sep="\n", caption=""):
    """ returns a string representation of a dict object for printing """
    res="=-=-=-=-==-=-=-=-="+sep+"DICT: "+caption+sep+"=-=-=-=-==-=-=-=-="+sep
    for i in arr:
        res+=str(i) + "\t\t" + str(arr[i]) + sep
    return res + "=-=-=-=-==-=-=-=-="+sep

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# core classes & functions
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def COST(infra, workflow, solution):
    iLoc = int(workflow.V[0])
    L = [x for x in solution]
    L.insert(0, iLoc)
    L.append(iLoc)
    c = 0
    
    for i in range(1, workflow.T+1):
        a = L[i]
        c +=   (workflow.D[i-1] * infra.DR[L[i-1], a] + \
                workflow.D[i-1] * infra.DE[a] + \
                workflow.V[i] * infra.VR[a] + \
                workflow.V[i] * infra.VE[a] + \
                workflow.D[i] * infra.DE[a] + \
                workflow.D[i] * infra.DR[a, L[i+1]] * int(workflow.T==i))
    return c

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class BASELINE:
    def __init__(self):
        pass
    
    def geteff(self, r): # efficiency
        return 1- ((r-self.min_cost) / self.cost_delta)

    def total_pies(self, infra, workflow):
        return infra.A**workflow.T
    def allpie(self, infra, workflow, verbose=0):
        """ Compare Cost for all policies on current workflow and find the min & max costs
            return TP, costs, min_count, min_cost, min_pies, max_count, max_cost, max_pies """
        self.A = infra.A
        self.T = workflow.T
        self.TP = self.A**self.T
        costs = []
        for i in range(self.TP):
            costs.append( COST (infra, workflow, int2base(i, self.A, self.T)) )
        costs = np.array(costs)
        min_cost, max_cost = np.min(costs), np.max(costs)
        min_pies, max_pies = np.where(costs==min_cost)[0], np.where(costs==max_cost)[0]
        min_count, max_count = len(min_pies), len(max_pies)

        self.costs = costs
        self.min_cost = min_cost
        self.min_count = min_count
        self.min_pies = min_pies
        self.max_cost = max_cost
        self.max_count = max_count
        self.max_pies = max_pies
        self.cost_delta = (self.max_cost-self.min_cost)

        if verbose>0:
            self.render()
        return # TP, costs, min_count, min_cost, min_pies, max_count, max_cost, max_pies

    def render(self):
        print('TOTAL-POLICIES:', self.TP)
        print('\nMIN-Cost:', self.min_cost)
        print('MIN-Cost COUNT:', self.min_count)
        for i in range(self.min_count):
            print('[',self.min_pies[i],']', int2base(self.min_pies[i], self.A, self.T))

        print('\nMAX-Cost:', self.max_cost)
        print('MAX-Cost COUNT:', self.max_count)
        for i in range(self.max_count):
            print('[',self.max_pies[i],']', int2base(self.max_pies[i], self.A, self.T))
        
        fig, ax = plt.subplots(1,2, figsize=(16,4))
        ax[0].plot(self.costs)
        ax[0].scatter(self.min_pies, np.zeros(self.min_count)+self.min_cost, color='green')
        ax[0].scatter(self.max_pies, np.zeros(self.max_count)+self.max_cost, color='red')
        _=ax[1].hist(self.costs, bins=50)
        plt.show()
        return

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class WORKFLOW:

    def __init__(self, arg_T, wfD=[1,100], wfV=[1,60], wfL=[0,0], seed=None):
        self.T = arg_T
        self.wfD = np.array(wfD)  # all data size
        self.wfV = np.array(wfV)   # all task size
        self.wfL = np.array(wfL)    # initial location 
        self.prng = np.random.default_rng(seed)
        self.new_flow()
        
    def new_flow(self):
        self.V = self.prng.uniform(self.wfV[0], self.wfV[1], size=self.T+1)
        self.V[0] = self.prng.integers(self.wfL[0],self.wfL[1]+1) # initiak work destination location  
        self.D = self.prng.uniform(self.wfD[0], self.wfD[1], size=self.T+1)
        self.D[0] = self.prng.uniform(self.wfD[0] + (self.wfD[1]-self.wfD[0])/2, self.wfD[1]) # initial data is a bit higher
        return
        
    def render(self):
        return ('V:{}\nD:{}'.format(self.V, self.D))
        
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class INFRA:

    def __init__(self, seed=None):
        self.prng = np.random.default_rng(seed)
        pass
                           
    def render(self):
        return ('E,C,A: {},{},{}\nVR:{}\nVE:{}\nDE:{}\nDR:\n{}'.format(self.E, self.C, self.A, self.VR, self.VE, self.DE, self.DR))

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# pre-defined infra objects
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def infra_1():
    x=INFRA()
    
    x.E = 1                     # no of Edge servers
    x.C = 1                     # no of Cloud servers
    x.A = x.E + x.C +1          # action space

    # 1 [DR] Data Rate(delay) # DR contains delay per unit of data
    gw8_DR = 1 
    x.DR = effective_bandwidth(np.array([
    #	#i0         e1			c2			#
    [	0,          5,  	    0,	   		], # i0
    [	0,          0,	        300,		], # e1
    [	0,          0,			0,	      	], # c2
    ], dtype='float') ) * gw8_DR 
    x.DRi = np.copy(x.DR)
    x.DRrL, x.DRrH = 0.5, 1.5
    

    # 2 [DE] Data Eng # DE contains energy cost per unit of data
    gw8_DE = 1
    ar_DE = np.array ([0.56,        0.52, 		0.55, 		], dtype='float')
    w8_DE = np.array ([	1,         1,		    1,		    ], dtype='float') * gw8_DE
    x.DE = np.array(np.multiply(ar_DE, w8_DE)) 
    x.DEi = np.copy(x.DE)
    x.DErL, x.DErH = 0.5, 1.5
    
    
    # 3 [VR] Task Rate(delay) # VR contains delay per unit of computation
    gw8_VR = 1
    x.VR =  np.array([	1/4.5,       1/5,       1/10,		], dtype='float') * gw8_VR
    x.VRi = np.copy(x.VR)
    x.VRrL, x.VRrH = 0.5, 1.5 # randomize ratio


    # 4 [VE] Task Eng # # VE contains energy cost per unit of computation
    gw8_VE = 1
    ar_VE = np.array ([0.56,         0.52, 		    0.55, 		], dtype='float')
    w8_VE = np.array ([1,          1,		    1,		], dtype='float') * gw8_VE
    x.VE = np.array(np.multiply(ar_VE, w8_VE)) 
    x.VEi = np.copy(x.VE)
    x.VErL, x.VErH = 0.5, 1.5 # randomize ratio

    
    #-=============================================
    return x
    #-=============================================
    
    
def infra_2():
    x=INFRA()
    
    x.E = 3                             # no of Edge servers
    x.C = 2                             # no of Cloud servers
    x.A = x.E + x.C +1                  # action space


    # 1 [DR] Data Rate(delay) # DR contains delay per unit of data
    gw8_DR = 1 
    x.DR = effective_bandwidth(np.array([
    #	#i0         e1			e2          e3          c4          c5	   #
    [	0,          5,  	    5, 	   		 5,          0,  	    0,  ], # i0
    [	0,          0,	        300,		 0,          300,  	    0   ], # e1
    [	0,          0,	        0,	    	 300,        300,  	    0,  ], # e2
    [	0,          0,	        0,	    	 0,          0,  	    300,], # e3
    [	0,          0,			0,	      	 0,          0,  	    500,], # c4
    [	0,          0,			0,	      	 0,          0,  	    0,  ], # c5
    ], dtype='float') ) * gw8_DR 
    x.DRi = np.copy(x.DR)
    x.DRrL, x.DRrH =0.5, 1.5


    # 2 [DE] Data Eng # DE contains energy cost per unit of data
    gw8_DE = 1
    ar_DE = np.array ([0.56,        0.522, 		0.521, 		0.522, 		0.55, 		0.55, 		], dtype='float')
    w8_DE = np.array ([	1,         1,		    1,		    1,		    1,		    1,		    ], dtype='float') * gw8_DE
    x.DE = np.array(np.multiply(ar_DE, w8_DE)) 
    x.DEi = np.copy(x.DE)
    x.DErL, x.DErH = 0.5, 1.5
    
    # 3 [VR] Task Rate(delay) # VR contains delay per unit of computation
    gw8_VR = 1
    x.VR =  np.array([	1/4.5,       1/5,       1/5.05,       1/5,       1/10,		1/10,		], dtype='float') * gw8_VR
    x.VRi = np.copy(x.VR)
    x.VRrL, x.VRrH = 0.5, 1.5 # randomize ratio


    # 4 [VE] Task Eng # # VE contains energy cost per unit of computation
    gw8_VE = 1
    ar_VE = np.array ([0.56,       0.52, 		 0.52, 		 0.52, 		 0.55, 		0.55, 		], dtype='float')
    w8_VE = np.array ([1,          1,		    1,		    1,		    1,		    1,		    ], dtype='float') * gw8_VE
    x.VE = np.array(np.multiply(ar_VE, w8_VE)) 
    x.VEi = np.copy(x.VE)
    x.VErL, x.VErH = 0.5, 1.5 # randomize ratio
    
    
    #-=============================================
    return x
    #-=============================================
    
    
def infra_3():

    x=INFRA()
    
    x.E = 5                             # no of Edge servers
    x.C = 2                             # no of Cloud servers
    x.A = x.E + x.C +1                  # action space

    # 1 [DR] Data Rate(delay) # DR contains delay per unit of data
    gw8_DR = 1 
    x.DR = effective_bandwidth(np.array([
    #	#i0         e1			e2          e3          e4          e5          c6          c7	   #
    [	0,          5,  	    5, 	   		 5,          5,  	    5,          0,          0,  ], # i0
    [	0,          0,	        300,		 0,          0,  	    0,          300,  	    0,  ], # e1
    [	0,          0,	        0,	    	 300,        0,  	    0,          300,  	    0,  ], # e2
    [	0,          0,	        0,	    	 0,          300,  	    0,          300,  	    0,  ], # e3
    [	0,          0,			0,	      	 0,          0,  	    300,        300,  	    0,  ], # e4
    [	0,          0,			0,	      	 0,          0,  	    0,          0,  	    300,  ], # e5
    [	0,          0,			0,	      	 0,          0,  	    0,          0,  	    500,  ], # c6
    [	0,          0,			0,	      	 0,          0,  	    0,          0,  	    0,  ], # c7
    ], dtype='float') ) * gw8_DR 
    x.DRi = np.copy(x.DR)
    x.DRrL, x.DRrH = 0.5, 1.5


    # 2 [DE] Data Eng # DE contains energy cost per unit of data
    gw8_DE = 1
    ar_DE = np.array ([0.56,        0.522, 		0.521, 		0.522, 		0.521, 		0.522, 		0.55, 		0.55, 		], dtype='float')
    w8_DE = np.array ([	1,         1,		    1,		    1,		    1,		    1,		    1,		    1,		    ], dtype='float') * gw8_DE
    x.DE = np.array(np.multiply(ar_DE, w8_DE)) 
    x.DEi = np.copy(x.DE)
    x.DErL, x.DErH = 0.5, 1.5
    
    # 3 [VR] Task Rate(delay) # VR contains delay per unit of computation
    gw8_VR = 1
    x.VR =  np.array([	1/4.5,       1/5,       1/5.05,       1/5,       1/5.05,       1/5,       1/10,		1/10,		], dtype='float') * gw8_VR
    x.VRi = np.copy(x.VR)
    x.VRrL, x.VRrH =0.5, 1.5 # randomize ratio


    # 4 [VE] Task Eng # # VE contains energy cost per unit of computation
    gw8_VE = 1
    ar_VE = np.array ([0.56,       0.52, 		 0.52, 		 0.52, 		 0.52, 		 0.52, 		 0.55, 		0.55, 		], dtype='float')
    w8_VE = np.array ([1,          1,		    1,		    1,		    1,		    1,		    1,		    1,		    ], dtype='float') * gw8_VE
    x.VE = np.array(np.multiply(ar_VE, w8_VE)) 
    x.VEi = np.copy(x.VE)
    x.VErL, x.VErH = 0.5, 1.5 # randomize ratio
    
    #-=============================================
    return x
    #-=============================================
    
    
def infra_4():

    x=INFRA()
    
    x.E = 8                             # no of Edge servers
    x.C = 3                             # no of Cloud servers
    x.A = x.E + x.C +1                  # action space

    # 1 [DR] Data Rate(delay) # DR contains delay per unit of data
    gw8_DR = 1 
    x.DR = effective_bandwidth(np.array([
    #	#i0         e1			e2          e3          e4          e5          e6              e7              e8              c9          c10          c11	   #
    [	0,          5,  	    5, 	   		 5,          5,  	    5,          5,              5,              5,              0,          0,          0,  ], # i0
    [	0,          0,	        300,		 0,          0,  	    0,          0,              0,              0,             300,         0,  	    0,  ], # e1
    [	0,          0,	        0,	    	 300,        0,  	    0,          0,              0,              0,             300,         0,  	    0,  ], # e2
    [	0,          0,	        0,	    	 0,          300,  	    0,          0,              0,              0,             300,         0,  	    0,  ], # e3
    [	0,          0,			0,	      	 0,          0,  	    0,          0,              0,              0,              0,  	    300,         0,  ], # e4
    [	0,          0,			0,	      	 0,          0,  	    0,          300,            0,              0,              0,        300,  	    300,  ], # e5
    [	0,          0,			0,	      	 0,          0,  	    0,          0,              300,            0,              0,          0,  	    300,  ], # e6
    [	0,          0,			0,	      	 0,          0,  	    0,          0,              0,              300,              0,          0,  	    300,  ], # e7
    [	0,          0,			0,	      	 0,          0,  	    0,          0,              0,              0,              0,          0,  	    300,  ], # e8
    [	0,          0,			0,	      	 0,          0,  	    0,          0,              0,              0,              0,          500,  	    0,  ], # c9
    [	0,          0,			0,	      	 0,          0,  	    0,          0,              0,              0,              0,          0,  	    500,  ], # c10
    [	0,          0,			0,	      	 0,          0,  	    0,          0,              0,              0,              0,          0,  	    0,  ], # c11
    ], dtype='float') ) * gw8_DR 
    x.DRi = np.copy(x.DR)
    x.DRrL, x.DRrH =0.5, 1.5


    # 2 [DE] Data Eng # DE contains energy cost per unit of data
    gw8_DE = 1
    ar_DE = np.array ([0.56,        0.522, 		0.521, 		0.522, 		0.521, 		0.522, 		0.522, 		0.521, 		0.522, 		0.55, 		0.55, 		0.55, 		], dtype='float')
    w8_DE = np.array ([	1,         1,		    1,		    1,		    1,		    1,		    1,		    1,		    1,		    1,		    1,		    1,		    ], dtype='float') * gw8_DE
    x.DE = np.array(np.multiply(ar_DE, w8_DE)) 
    x.DEi = np.copy(x.DE)
    x.DErL, x.DErH = 0.5, 1.5
    
    # 3 [VR] Task Rate(delay) # VR contains delay per unit of computation
    gw8_VR = 1
    x.VR =  np.array([	1/4.5,       1/5,       1/5.05,       1/5,       1/5.05,       1/5.05,       1/5,       1/5.05,       1/5,       1/10,		1/10,		1/10,		], dtype='float') * gw8_VR
    x.VRi = np.copy(x.VR)
    x.VRrL, x.VRrH = 0.5, 1.5 # randomize ratio


    # 4 [VE] Task Eng # # VE contains energy cost per unit of computation
    gw8_VE = 1
    ar_VE = np.array ([0.56,       0.52, 		 0.52, 		 0.52, 		 0.52, 		 0.52, 		 0.52, 		 0.52, 		 0.52, 		 0.55, 		0.55, 		0.55, 		], dtype='float')
    w8_VE = np.array ([1,          1,		    1,		    1,		    1,		    1,		    1,		    1,		    1,		    1,		    1,		    1,		    ], dtype='float') * gw8_VE
    x.VE = np.array(np.multiply(ar_VE, w8_VE)) 
    x.VEi = np.copy(x.VE)
    x.VErL, x.VErH = 0.5, 1.5 # randomize ratio
    
    #-=============================================
    return x
    #-=============================================
    
    
    
