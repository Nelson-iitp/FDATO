
# Global Section
import numpy as np
import matplotlib.pyplot as plt
import datetime
now = datetime.datetime.now

def strA(arr, start="[", sep=",", end="]"):
    """ returns a string representation of an array/list for printing """
    res=start
    for i in range(len(arr)):
        res+=str(arr[i])+sep
    return res + end

def l2(x): # L2 norm of difference vector (Euclidian distance b/w vectors)
    return np.sum(x**2)**0.5

class FDA:
    def __init__(self, n_dim, lb, ub, costF, alpha, beta, seed=None):

        """
        Args:
            
            n_dim       dimension of soultion space
            lb          lower bound
            ub          upper bound
            costF       a function like:  lambda soultion: cost 
            alpha       initial population size
            beta        neighbourhood size (no of neighbours)
        """
        self.prng = np.random.default_rng(seed)
        # problmen formulation
        self.n_dim, self.lb, self.ub = n_dim, lb, ub
        self.cost = costF
        self.alpha = alpha
        self.beta = beta

    def get_random_flow(self): # a function for generating a random solution
        return self.lb + self.prng.uniform(0,1,size=self.n_dim)*(self.ub-self.lb)

    def optimize(self, MAXITER, verbose=1, plot=True):
        W_var_history, C_history, V_history = [], [], []

        # Initial flow population
        self.Flow_X = [ self.get_random_flow() for _ in range(self.alpha) ] 
        if verbose>0:
            print('\nInitial flow:\n', strA(self.Flow_X, start='', sep='\n',end=''))

        start_time = now()
        print('\nBegin {} Iterations...'.format(MAXITER))
        for ITER in range(1, MAXITER): 

            if verbose>0:
                print('[ITER {}]'.format(ITER))
                if verbose>2:
                    print('flow:\n', strA(self.Flow_X, start='\n', sep='\n',end='\n'))

            Flow_newX = [] #<--- for new set of flows
            
            #  fitness for all flows
            Flow_fitness = np.array([self.cost(x) for x in self.Flow_X])
            if verbose>2:
                print('fitness:', Flow_fitness)

            #  best flow out of current flows (in Flow_X)
            best_flow_at = np.argmin(Flow_fitness)
            best_flow = self.Flow_X[best_flow_at]
            best_flow_cost = Flow_fitness[best_flow_at]
            if verbose>1:
                print('best-flow:', best_flow, ' cost:', best_flow_cost)
            C_history.append(best_flow_cost)

            # calulate 'W' which is required for calulating a 'delta' for each flow in Flow_X
            rand_bar = self.prng.uniform(0,1,size=self.n_dim)
            randn = self.prng.normal(0,1)
            iter_ratio = ITER/MAXITER

            W =     ( 1-iter_ratio ) ** ( 2 * randn ) * \
                    ( (iter_ratio * rand_bar) * rand_bar)
            W_var_history.append(np.var(W))

            flow_velocity = []
            for i,flow_i in enumerate(self.Flow_X):

                rand, x_rand = self.prng.uniform(0,1), self.get_random_flow()
                delta = rand*(x_rand-flow_i) * l2(best_flow-flow_i) * W

                # create beta neighbours
                Neighbour_X = [ flow_i + self.prng.normal(0,1)*delta \
                            for _ in range(self.beta) ]

                # cal neighbour fitness
                Neighbour_fitness = np.array( [self.cost(n) for n in Neighbour_X] )

                best_neighbour_at = np.argmin(Neighbour_fitness)
                best_neighbour = Neighbour_X[best_neighbour_at]
                best_neighbour_cost = Neighbour_fitness[best_neighbour_at]
                #print('\tbest-neighbour:', best_neighbour, ' cost:', best_neighbour_cost)


                s0=[] #<---- slopes
                for j in range(self.beta):
                    num = Flow_fitness[i]-Neighbour_fitness[j]

                    #@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=
                    # **[NOTE:1]
                    #@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=
                    s0.append( num/np.abs(flow_i - Neighbour_X[j]) )  
                    #s0.append( num/l2(flow_i - Neighbour_X[j]) )  # if useing, also change line [121, 132]
                    #@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=@=#=

                if best_neighbour_cost < Flow_fitness[i]:
                    V = self.prng.normal(0,1)*s0[best_neighbour_at] #<---- flope of best neighbour 
                    new_flow_i = flow_i + \
                                V * ( (flow_i-best_neighbour)/l2(flow_i-best_neighbour))
                else:
                    V = np.zeros(self.n_dim) + (1/self.n_dim**0.5) #<---- so that log(V) is zero
                    r = i
                    while r==i:
                        r = self.prng.integers(0, len(self.Flow_X))
                    if Flow_fitness[r]<Flow_fitness[i]:
                        randn_bar=self.prng.normal(0,1,size=self.n_dim)
                        new_flow_i = flow_i + randn_bar*(self.Flow_X[r]-flow_i) # Note, flow_i == Flow_X[i]
                    else:
                        rand_n = self.prng.uniform(0,1)
                        new_flow_i = flow_i + 2*rand_n*(best_flow - flow_i)

                flow_velocity.append(l2(V))
                Flow_newX.append(new_flow_i)
            # end for (all flows)

            V_history.append(flow_velocity)
            Flow_newfitness = np.array([self.cost(x) for x in Flow_newX])    

            for i in range(self.alpha):
                if Flow_newfitness[i] < Flow_fitness[i]:
                    self.Flow_X[i] = Flow_newX[i]
            
            #print('\n')
        # end for (all iterations)

        # Results
        Flow_fitness = np.array([self.cost(x) for x in self.Flow_X])
        best_flow_at = np.argmin(Flow_fitness)
        if verbose>0:
            print('--------------------------------------------')
            print('Final flows:', strA(self.Flow_X, start='\n', sep='\n',end=''))
            print('Final fitness:', Flow_fitness)
            print('--------------------------------------------')

        if plot:
            # results plot
            fig, ax = plt.subplots(2,1, figsize=(12,12))
            ax[0].plot(C_history, linewidth=0.6, color='tab:green')
            ax[0].set_title('Cost History')

            # plot the variance in logarithmic scale, 
            W_log_var = np.log(W_var_history)
            ax[1].plot(W_log_var, linewidth=0.6, color='tab:red') 
            ax[1].grid(axis='both')
            ax[1].set_title('W Var History')
            plt.show()

            # plot flow velocity history for all flows in population
            flow_velocity_history = np.array(V_history)
            fig, ax = plt.subplots(self.alpha, 1, figsize=(12,6), sharex=True)
            for i in range(self.alpha):
                ax[i].plot(np.log(flow_velocity_history[:,i]), label='flow_'+str(i), linewidth=0.5)
                ax[i].set_ylabel('flow_'+str(i))
            fig.suptitle('Population Velocity')
            plt.show()

        best_flow = self.Flow_X[best_flow_at]
        best_flow_cost = Flow_fitness[best_flow_at]
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Best-Flow [{}]:\t'.format(best_flow_at), best_flow)
        print('Minimum-Cost:\t', best_flow_cost)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

        end_time = now()
        print('\nFinished!\nElapsed Time: {}\n'.format(end_time-start_time))

        return best_flow, best_flow_cost, self.Flow_X, Flow_fitness, W_var_history, C_history, V_history 