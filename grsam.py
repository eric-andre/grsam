################################################################
###                         grsam.py
###       generalized recursive smooth ambiguity model
###
### An implementation in Python of the model proposed in
###   Ju & Miao (2012). Ambiguity, Learning, and Asset Returns.
###   Econometrica, 80(2), 559‑591.
###
### Implementation by Eric André and Silvia Faroni, 2023
### available as open source under the terms of the MIT License.
#################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns ; sns.set()
import scipy.stats as stats
import scipy.optimize as opt
from tqdm.notebook import tqdm


class Portfolio():
    """
    A class for solving dynamic optimal consumption and investment problems
    in the Ju and Miao (2012) framework.
    
    This version allows for two risky assets which are jointly Normally distributed.
    The covariance matrix is known but there is a bivariate Normal prior for the mean of the distribution.
    """
    
    #######################
    ### Initializations ###
    #######################
    
    def __init__(self, horizon):
        """
        horizon is the maturity in years
        time runs from t=0 to t=horizon hence contains horizon+1 elements
        """
        self.horizon = horizon
        
    def init_utility(self, beta, rho, gamma, eta):
        self.beta = beta
        self.rho = rho
        self.gamma = gamma
        self.eta = eta
                
    def init_assets(self, risk_free_rate, mean, covariance):
        """
        `rf` stores the *gross* risk-free rate
        `mean` is the unobserved *real* expected *excess* return
        """
        self.rf = 1 + risk_free_rate
        self.mean = mean
        self.n_assets = mean.shape[0]
        self.cov = covariance
        self.precision = np.linalg.inv(covariance)
        
    def init_belief(self, mean_belief, covariance_belief):
        """
        Initialize the prior on the assets' Normal distribution's means vector
        """
        self.mean_belief_init = mean_belief
        self.cov_belief_init = covariance_belief
        self.precision_belief_init = np.linalg.inv(covariance_belief)
    
    def init_results(self):
        """
        Pre allocation of arrays to store optimal controls and value functions
        Note that there is no portfolio choice at t=horizon, hence the second dimension of theta
        """
        self.theta = np.zeros((self.n_simul, self.horizon, self.n_assets), dtype='float64')
        self.ct = np.zeros((self.n_simul, self.horizon+1), dtype='float64')
        self.G = np.zeros((self.n_simul, self.horizon+1), dtype='float64')
        self.H = np.zeros((self.n_simul, self.horizon+1), dtype='float64')
        self.reg_res = {}
    
    def standard_initialization(self):
        """
        Initialize the class with standard values to perform tests
        """
        ### utility ###
        # parameters taken from Bansal, R., & Yaron, A. (2004).
        # Risks for the Long Run : A Potential Resolution of Asset Pricing Puzzles.
        # The Journal of Finance, 59(4), 1481‑1509.
        # https://doi.org/10.1111/j.1540-6261.2004.00670.x
        self.init_utility(beta=.975, rho=1/1.5, gamma=2, eta=8.864)
        
        ### assets ###
        m = np.array([0.08, 0.08])
        vol1 = 0.2
        vol2 = 0.2
        rho = 0.5
        cov = rho * vol1 * vol2
        sigma = np.array([[vol1**2, cov],
                          [cov, vol2**2]])
        self.init_assets(risk_free_rate=0.02, mean=m, covariance=sigma)
        
        ### prior ###
        m_belief0 = m + np.array([0.05, 0.05])
        factor1 = 1/10
        factor2 = 1/4
        sigma_belief0 = np.array([[factor1*vol1**2, 0],
                                  [0, factor2*vol2**2]])
        
        self.init_belief(mean_belief=m_belief0, covariance_belief=sigma_belief0)
        
        
    ####################
    ### Sample paths ###
    ####################
    
    def simulate_paths(self, n_simul, seed=42):
        """
        `returns` is an (n_simul x horizon+1 x n_assets) matrix
        with zeros at time t=0
        """
        self.n_simul = n_simul
        self.rng = np.random.default_rng(seed)
        returns_random = self.rng.multivariate_normal(
            self.mean,
            self.cov,
            size=(self.n_simul, self.horizon)
        )
        returns_init = np.zeros((self.n_simul, 1, self.n_assets))
        self.returns = np.hstack((returns_init, returns_random))
        
    def display_paths(self, n_display, legend=False):
        x = np.arange(self.horizon+1)
        mean = self.returns[:,1:,0].mean(axis=0)
        std = self.returns[:,1:,0].std(axis=0)
        q = stats.norm.ppf(.975)
        
        indices = self.rng.choice(np.arange(self.n_simul),
                                  size=n_display,
                                  replace=False
                                 )

        fig, ax = plt.subplots(figsize=(12, 5))
        for i in indices:
            ax.plot(x, self.returns[i, :, 0].cumsum(), lw=.5, label=f"sim{i}")
        #ax.plot(x, mean, color='red', label="mean over simulations")
        #ax.fill_between(x, mean+q*std, mean-q*std,
        #                color='y', alpha=.2, label="95% CI")
        ax.set_xticks(x)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_xlabel("time")
        ax.set_ylabel("returns")
        ax.set_title("simulated cumulative log returns")
        if legend:
            ax.legend(fontsize=8, ncol=2);

    def display_paths_hist(self, time=0):
        """
        histogram of simulated returns at a given time accross all paths
        """
        if time == 0:
            shp = self.returns.shape
            new_shp = (shp[0]*shp[1], shp[2])
            data = self.returns.reshape(new_shp)
            title = "all times"
        else:
            data = self.returns[:, time, :]
            title = f"returns at time {time}"

        df = pd.DataFrame(data=data,
                          columns=['asset 1', 'asset 2'])
        g = sns.jointplot(data=df, x='asset 1', y='asset 2',
                          height=5, marker='.', alpha=.2)
        g.refline(x=self.mean[0], y=self.mean[1], linewidth=1)
        g.figure.suptitle(title)
        g.figure.tight_layout()


    ################
    ### Learning ###
    ################
    
    def learning(self):
        """
        `mean_belief` is an (n_simul x horizon+1 x n_assets) matrix
        `cov_belief` is an (horizon+1 x n_assets x n_assets) matrix
        `precision_belief` is an (horizon+1 x n_assets x n_assets) matrix
        values at time t=0 are the initial prior before learning
        Values at time t are the posteriors after observation of `returns` at time t.
        """
        self.mean_belief = np.zeros((self.n_simul, self.horizon+1, self.n_assets))
        
        # sigma_belief does not change for each simulation, so we just compute for each horizon
        self.cov_belief = np.zeros((self.horizon+1, self.n_assets, self.n_assets))
        self.precision_belief = np.zeros((self.horizon+1, self.n_assets, self.n_assets))
        
        # returns[t] are the observed returns which are used to update the posterior at date t+1
        # due to indexing of arrays, the posterior at date t+1 is at index t.
        t = 0
        self.precision_belief[t,] = self.precision_belief_init 
        self.cov_belief[t,] = np.linalg.inv(self.precision_belief[t,])
        self.mean_belief[:,t,:] = self.mean_belief_init

        for t in range(1, self.horizon+1):
            self.precision_belief[t,] = self.precision_belief[t-1,] + self.precision
            self.cov_belief[t,] = np.linalg.inv(self.precision_belief[t,])
            mean_updated = self.cov_belief[t,] @ self.precision @ self.returns[:,t,:].T
            mean_updated += self.cov_belief[t,] @ self.precision_belief[t-1,] @ self.mean_belief[:,t-1,:].T
            self.mean_belief[:,t,:] = mean_updated.T
                
    def display_posterior(self):
        x = np.arange(self.horizon+1)
        titles = ["Means of the posterior distribution of the mean excess return",
                  "Variances of the posterior distribution of the mean excess return",
                  "Correlation"]
        ylabels = ["returns",
                   "variances",
                   "correlation"]
        
        fig, axs = plt.subplots(3, figsize=(10,10), sharex=True)
        for i in range(self.n_assets):
            axs[0].plot(x[0], self.mean_belief_init[i],
                        color='C'+str(i), marker='o', ls='',
                        label=f"prior asset {i+1}")
            axs[0].plot(x[1:], self.mean_belief[:,1:,:].mean(axis=0)[:,i],
                        color='C'+str(i), 
                        label=f"posterior asset {i+1}")
            axs[0].axhline(self.mean[i],
                           ls=':', color='C'+str(i),
                           label=f"true mean asset {i+1}")
            axs[1].plot(x[0], self.cov_belief_init[i,i],
                        color='C'+str(i), marker='o', ls='',
                        label=f"prior asset {i+1}")
            axs[1].plot(x[1:], self.cov_belief[1:,i,i],
                        color='C'+str(i), 
                        label=f"posterior asset {i+1}")
        
        corr = self.cov_belief[:,0,1] / np.sqrt(self.cov_belief[:,0,0] * self.cov_belief[:,1,1])
        true_corr = self.cov[0,1] / np.sqrt(self.cov[0,0] * self.cov[1,1])
        axs[2].plot(x, corr, label=f"correlation assets 1 and 2")
        axs[2].axhline(true_corr,
                       ls=':', color='C0',
                       label="true correlation")
        
        for i, ax in enumerate(axs):
            ax.set_title(titles[i])
            ax.set_ylabel(ylabels[i])
            ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
        axs[0].set_ylim(bottom=0)
        axs[1].set_ylim(bottom=0)
        axs[2].set_ylim((-1,1))
        axs[2].set_xticks(x[::2])
        axs[2].set_xlabel("time")        

        
    ################################
    ### Gauss-Hermite Quadrature ###
    ################################
    
    def init_GaussHermite(self, n_Gauss):
        """
        Compute GH nodes and weights
        Create meshgrids for double integration in the bivariate case
        """
        x, w = np.polynomial.hermite.hermgauss(n_Gauss)
        self.x_gauss = x
        self.w_gauss = w
        
        X1, X2, X3, X4 = np.meshgrid(x, x, x, x)
        self.X_p1 = X1
        self.X_p2 = X2
        self.X_n1 = X3
        self.X_n2 = X4
        
        W1, W2 = np.meshgrid(w, w)
        self.W_n1 = W1
        self.W_n2 = W2
    
    def init_corr_nodes(self, cov):
        """
        For single integration and inner integral of double integration.
        Only for 2x2 covariance matrices.
        Use the correlation structure to generate correlated nodes.
        Means must be added to the correlated nodes
        """
        corr = cov[0,1] / np.sqrt(cov[0,0] * cov[1,1])

        # self.X_p1[:,:,0,0] is the same as the 2-dim matrix generated by meshgrid(x,x)
        R1 = np.sqrt(2*cov[0,0]) * self.X_p1[:,:,0,0]
        
        R2 = corr * self.X_p1[:,:,0,0]
        R2 += np.sqrt(1-corr**2) * self.X_p2[:,:,0,0]
        R2 *= np.sqrt(2*cov[1,1])
        
        self.R1_s = R1
        self.R2_s = R2
    
    def init_corr_nodes_double(self, cov_outer): # , cov_inner
        """
        For outer integral of double integration.
        Only for 2x2 covariance matrices.
        Use the correlation structure to generate correlated nodes.
        Means must be added to the correlated nodes
        """
        corr_outer = cov_outer[0,1]
        corr_outer /= np.sqrt(cov_outer[0,0]*cov_outer[1,1])

        R1 = np.sqrt(2*cov_outer[0,0]) * self.X_p1
        R1 += self.R1_s
        
        R2 = corr_outer * np.sqrt(2*cov_outer[1,1]) * self.X_p1
        R2 += np.sqrt(2*cov_outer[1,1]*(1-corr_outer**2)) * self.X_p2
        R2 += self.R2_s
        
        self.R1_d = R1
        self.R2_d = R2
    
    def init_nodes_predictive(self):
        """
        For simple integration with predictive probability
        """
        self.init_corr_nodes(cov=self.cov_belief_init + self.cov)
    
    def inner_quadrature(self, X):
        """
        X is an (n_Gauss x n_Gauss x n_Gauss x n_Gauss) matrix
        which contains the values of a function of the bivariate Normal random variables
        at the quadrature nodes
        """
        Y = X * self.W_n2 * self.W_n1
        return Y.sum(axis=3).sum(axis=2) / np.pi
      
    def outer_quadrature(self, X):
        """
        X is an (n_Gauss x n_Gauss) matrix
        which contains the values of a function of the bivariate Normal random variables
        at the quadrature nodes
        """
        Y = X * self.W_n1 * self.W_n2
        return Y.sum() / np.pi

    
    ###########################
    ### Portfolio Returns  ###
    ##########################
    
    def gross_return(self, theta, mean):
        """
        Compute portfolio gross returns at all Gauss Hermite nodes
        `mean` is the mean of the belief distribution
        Returns `Ret` a ndarray of size (n_Gauss x n_Gauss x n_Gauss x n_Gauss) for double quadrature.
        """
        R1 = np.exp(mean[0] + self.R1_d) - self.rf
        R2 = np.exp(mean[1] + self.R2_d) - self.rf
        Ret = self.rf + theta[0] * R1 + theta[1] * R2
        return Ret
        
    def portf_return(self, theta, returns):
        """
        Compute portfolio gross returns from simulated assets returns
        Returns a scalar
        """
        return self.rf + np.dot(theta, np.exp(returns)-self.rf)
    
    ##################################
    ### Optimize Without Learning  ###
    ##################################
    
    def init_constraints(self):
        """
        Portfolio weights constraints for optimization
        lb <= A @ [theta1 theta2]' <= ub
        """
        A = np.array([[1,1],
                      [1,0],
                      [0,1]])
        lb = np.array([-np.inf,0,0])
        ub = np.array([1,np.inf,np.inf])
        self.cons = opt.LinearConstraint(A=A, lb=lb, ub=ub) # contains constraint + bounds
        
        
    def tildeH(self, theta, mean_belief):
        """
        Compute the KMM certainty equivalent.
        Returns the opposite of \tilde{H} for use with scipy.optimize.minimize.
        Expectations computed with double quadrature.
        mean_belief: 2-dim mean vector of the prior belief.
        """
        ret = self.gross_return(theta, mean_belief)
        ret **= 1 - self.gamma
        E = self.inner_quadrature(ret)
        E **= (1-self.eta) / (1-self.gamma)
        H = self.outer_quadrature(E)
        H **= 1 / (1-self.eta)
        return -H
    
    def tildeH_Jac(self, theta, mean_belief):
        """
        Return the Jacobian of tildeH,
        that is the Jacobian of the opposite of the KMM certainty equivalent.
        Expectations computed with double quadrature.
        mean_belief: 2-dim mean vector of the prior belief.
        """
        # excess returns
        R1 = np.exp(mean_belief[0] + self.R1_d) - self.rf
        R2 = np.exp(mean_belief[1] + self.R2_d) - self.rf
        # portfolio return
        ret = self.rf + theta[0]*R1 + theta[1]*R2
        # KMM certainty equivalent
        E = self.inner_quadrature(ret**(1-self.gamma))
        E **= 1 / (1-self.gamma)
        KMM = self.outer_quadrature(E**(1-self.eta))
        KMM **= 1 / (1-self.eta)
        # second components of the derivatives
        ret **= -self.gamma
        dE1 = self.inner_quadrature(ret*R1)
        dE2 = self.inner_quadrature(ret*R2)
        E **= self.gamma - self.eta
        E1 = self.outer_quadrature(E*dE1)
        E2 = self.outer_quadrature(E*dE2)
        return -KMM**self.eta * np.array([E1, E2])
        
    def OptimizeNoLearning(self, n_Gauss):
        
        #Initialization
        self.init_GaussHermite(n_Gauss=n_Gauss)
        self.init_corr_nodes(cov=self.cov)
        self.init_corr_nodes_double(cov_outer=self.cov_belief_init)
        self.init_constraints()
        
        #optimization without learning = theta constant
        res = opt.minimize(self.tildeH, (0.25,0.25),
                           args=(self.mean_belief_init),
                           constraints=self.cons)
        self.theta[:,:,:] = res.x
        self.H_tilde = -res.fun
 
        #at time T
        self.G[:,-1] = 1
        self.ct[:,-1] = 1
        
        for t in range(self.horizon-1, -1, -1):
            self.H[:,t] = self.H_tilde * self.G[:,t+1]
            self.ct[:,t] = (1+(self.beta*self.H[:,t]**(1-self.rho))**(1/self.rho)) ** (-1)
            self.G[:,t] = self.ct[:,t] ** (-self.rho/(1-self.rho))

    
    ########################################
    ### Optimize Portfolio with learning ###
    ########################################   
    
    def polyB(self, B, t):
        
        if B.all==1:
            return t
        else:
            tmp = B ** (t+1)
            tmp -= 1
            tmp /= (B-1)
            return tmp

    def Ht_dynamic(self, theta, mean, value_fun):
        """
        `mean`: mean_belief
        `value_fun`: polynomial approximation of G_{t+1} computed on all GH nodes
                     must be a ndarray of size (n_Gauss x n_Gauss x n_Gauss x n_Gauss) 
        """
        #return of portfolio
        ret = self.gross_return(theta, mean)
        ret *= value_fun
        ret **= 1 - self.gamma
        E = self.inner_quadrature(ret)
        E **= (1-self.eta) / (1-self.gamma)
        H = self.outer_quadrature(E)
        H **= 1 / (1-self.eta)
        
        return -H
    
    def OptimizeLearning(self, alpha, n_Gauss1):
        #initialization
        self.init_results()        
        self.init_GaussHermite(n_Gauss=n_Gauss1) #to be initialized ones 
        self.init_corr_nodes(cov=self.cov)#to be initialized ones
        self.init_constraints()
        theta_0 = np.array([0.5, 0.5])
        
        #Time T
        self.ct[:,-1] = 1
        self.G[:,-1] = 1
        
        #Time T-1
        #to be initialized for each period
        self.init_corr_nodes_double(cov_outer=self.cov_belief[self.horizon - 1,:,:])  
        
        
        for i in range(self.n_simul):
            #min function
            try:
                res = opt.minimize(self.tildeH, theta_0, #same function as in the "no learning"
                                   args=(self.mean_belief[i,self.horizon - 1,:]),
                                   constraints=self.cons,
                                  )
                self.theta[i,self.horizon - 1,:] = res.x
                H_max = -res.fun
            except ValueError:
                print(f"Value error at time {self.horizon - 1} with sample {i}")
                self.theta[i,self.horizon - 1,:] = np.nan
                H_max = np.nan
                
            #compute ct, Gt, Ht
            self.H[i,self.horizon - 1] = H_max
            self.ct[i,self.horizon - 1] = (1+(self.beta * H_max**(1-self.rho))**(1/self.rho))**(-1)
            self.G[i,self.horizon - 1] = self.ct[i,self.horizon - 1]**(-self.rho/(1-self.rho))


        for t in range(self.horizon-2,-1,-1):
            self.init_corr_nodes_double(cov_outer=self.cov_belief[t,:,:]) #to be initialized for each period  %nodes for gauss hermite quadd


            A=self.cov_belief[t+1,:,:]@self.precision
            B=self.cov_belief[t+1,:,:]@self.precision_belief[t,:,:]
    
            for i in range(self.n_simul):
                
                #the update of the belief using the points for gauss hermite quadrature
                belief=np.array(self.mean_belief[i,t,:]).reshape(self.mean_belief[i,t,:].size,1)
                S1=A[0,0]*(self.R1_d+self.mean_belief[i,t,0])+A[0,1]*(self.R2_d+self.mean_belief[i,t,1])
                S2=A[1,0]*(self.R1_d+self.mean_belief[i,t,0])+A[1,1]*(self.R2_d+self.mean_belief[i,t,1])
                S3=B@belief

                b1=S3[0]+S1
                b2=S3[1]+S2
                
                sum_mean = b1+b2
                sum_var = self.cov_belief[t+1,0,0]+self.cov_belief[t+1,1,1]
                risk_adj1 = b1 / self.cov_belief[t+1,0,0]
                risk_adj2 = b2/ self.cov_belief[t+1,1,1]
                risk_sum = risk_adj1+risk_adj2
                ratio = (alpha[0]
                     + alpha[1]*sum_mean
                     + alpha[2]*sum_var
                     + alpha[3]*risk_sum
                )

                Gtplus = self.polyB(ratio,(self.horizon-(t+1))) ** (self.rho/(1-self.rho))
                #min function
                try:
                    res=opt.minimize(self.Ht_dynamic, 
                           theta_0,
                           args=(self.mean_belief[i,t,:], Gtplus),
                           constraints=self.cons,
                           method = 'trust-constr')
                    self.theta[i,t,:]=res.x
                    H_max = -res.fun
                except ValueError:
                    print(f"Value error at time {t} with sample {i}")
                    self.theta[i,self.horizon - 1,:] = np.nan
                    H_max = np.nan
                
                #compute ct, Gt, Ht
                self.H[i,t] = H_max
                self.ct[i,t] = (1+(self.beta * H_max**(1-self.rho))**(1/self.rho))**(-1)
                self.G[i,t] = self.ct[i,t]**(-self.rho/(1-self.rho))



    
    ########################################
    ### Plot results ###
    ########################################   
    
    def compute_wealth(self, starting_wealth):
        self.wealth = np.zeros((self.n_simul, self.horizon+1))
        self.real_consumption = np.zeros((self.n_simul, self.horizon+1))
        self.wealth[:,0] = starting_wealth
        self.real_consumption[:,0] = starting_wealth * self.ct[:,0]
        for t in range(self.horizon):
            for s in range(self.n_simul):
                portf_gross_ret = self.portf_return(self.theta[s,t,:], self.returns[s,t+1,:])
                self.wealth[s,t+1] = self.wealth[s,t] * (1-self.ct[s,t]) * portf_gross_ret
                self.real_consumption[s,t+1] = self.ct[s,t+1] * self.wealth[s,t+1]

    def plot_wealth(self):
        t = np.arange(self.horizon+1)
        cons = self.real_consumption.mean(axis=0)
        w_start = self.wealth.mean(axis=0)
        w_end = w_start - cons
        pv = np.roll(w_start, -1) - w_end
        pv[-1] = 0
        
        fig, (ax_w, ax_c) = plt.subplots(1, 2, figsize=(12,4))
        ax_w.plot(t, w_start, label="start of period")
        ax_w.plot(t, w_end, label="end of period")
        ax_w.set_title("Wealth over time")
        ax_w.set_xlabel("time")
        ax_w.legend(fontsize='small')
        ax_c.plot(cons, label="consumption")
        ax_c.plot(pv, label="gains on portfolio")
        ax_c.set_title("Changes in wealth over time")
        ax_c.set_xlabel("time")
        ax_c.legend(fontsize='small')

    def plot_optimal_control(self):
        fig, axs = plt.subplots(2, 2, figsize=(10,8))
    
        axs[0,0].plot(self.theta[:,:,0].mean(axis=0),
                      label="Asset 1"
                     )
        axs[0,0].plot(self.theta[:,:,1].mean(axis=0),
                      label="Asset 2"
                     )
        axs[0,0].plot(self.theta_static[:,:,0].mean(axis=0),
                      c='g', ls=':', lw=1,
                      label="KMM Asset 1"
                     )
        axs[0,0].plot(self.theta_static[:,:,1].mean(axis=0),
                      c='g', ls='--', lw=1,
                      label="KMM Asset 2"
                     )
        axs[0,0].set_ylim(0,1.05)
        axs[0,0].set_title("Asset weights")
        axs[0,0].legend()
        
        axs[0,1].plot(self.theta[:,:,:].sum(axis=2).mean(axis=0),
                      label="Participation"
                     )
        axs[0,1].plot(self.theta_static[:,:,:].sum(axis=1).mean(axis=0),
                      c='g', ls=':', lw=1,
                      label="Participation KMM"
                     )
        axs[0,1].set_title("Participation")
        axs[0,1].set_ylim(0,1.05)
        axs[0,1].legend()
    
        axs[1,0].plot(self.ct.mean(axis=0))
        axs[1,0].set_title("Consumption")
        
        times = np.arange(self.horizon+1)
        times[0] = 1
        times = (1-self.rho)/times[::-1]
        axs[1,1].plot((self.G.mean(axis=0)**times)[:-1])
        axs[1,1].set_title("Value function")

    
    def plot_value_functions_over_time(self):
        titles = ["function H", "function G", "Consumption (% of present wealth)"]
        functions = [self.H, self.G, self.ct]
        t = np.arange(self.horizon+1)
        fig, axs = plt.subplots(2, 2, figsize=(12,8))
        for ax, fun, title in zip(axs.ravel(), functions, titles):
            ax.plot(t, fun.mean(axis=0))
            ax.set_title(title)
        ax = axs[1,1]
        ax.plot(self.theta[:,:,0].mean(axis=0), label="asset 1")
        ax.plot(self.theta[:,:,1].mean(axis=0), label="asset 2")
        ax.plot(self.theta[:,:,:].mean(axis=0).sum(axis=1), label="participation")
        ax.set_title("portfolio composition")
        ax.set_ylim([-0.05,1.05])
        ax.legend(fontsize='small')
        fig.supxlabel("time")

    def plot_value_functions(self, time):
        titles = ["function H", "function G", "Consumption"]
        functions = [self.H, self.G, self.ct]
        fig, axs = plt.subplots(1, 3, figsize=(15,4))
        for (ax, title, fun) in zip(axs, titles, functions):
            ax.hist(fun[:, time])
            ax.set_title(title)
        return fig

    def plot_portfolios(self):
        time=self.horizon-1
        fig = plt.figure(figsize=(18,4))
        ax1 = fig.add_subplot(141)
        ax2 = fig.add_subplot(142, sharey=ax1)
        ax3 = fig.add_subplot(143)
        ax4 = fig.add_subplot(144)
        
        ax1.hist(self.theta[:, time, 0], bins=20)
        ax2.hist(self.theta[:, time, 1], bins=20)
        ax3.hist(self.theta[:, time, :].sum(axis=1), bins=20)
        ax4.scatter(self.theta[:, time, 0], self.theta[:, time, 1], marker='.', alpha=.1)
        ax1.set_title("theta 1")
        ax2.set_title("theta 2")
        ax3.set_title("share of risky assets")
        ax4.set(title="portfolios", xlabel="theta1", ylabel="theta2")
        return fig

        
    ##############################################
    ### Static portfolio with two risky assets ###
    ##############################################
    
    def CRRA_EU(self, theta):
        """
        With simple quadrature
        """
        R1 = np.exp(self.mean_belief_init[0] + self.R1_s) - self.rf
        R2 = np.exp(self.mean_belief_init[1] + self.R2_s) - self.rf
        ret = self.rf + theta[0] * R1 + theta[1] * R2
        ret **= 1 - self.gamma
        EU = self.outer_quadrature(ret)
        return EU / (1-self.gamma)

    def CRRA_EU_double_quad(self, theta):
        """
        With double quadrature
        """
        ret = self.gross_return(theta, self.mean_belief_init)
        ret **= 1 - self.gamma
        E = self.inner_quadrature(ret)
        EU = self.outer_quadrature(E)
        return EU / (1-self.gamma)

    
    def KMM_Hess(self, theta, mean_belief):
        """
        Return the Hessian of KMM utility of portfolio
        """
        # excess returns
        e1 = np.exp(mean_belief[0] + self.R1_d) - self.rf
        e2 = np.exp(mean_belief[1] + self.R2_d) - self.rf
        # portfolio return
        ret = self.rf + theta[0]*e1 + theta[1]*e2
        # Harmonic mean of portfolio return
        E = self.inner_quadrature(ret**(1-self.gamma))
        E **= 1 / (1-self.gamma)
        # KMM certainty equivalent
        KMM = self.outer_quadrature(E**(1-self.eta))
        KMM **= 1 / (1-self.eta)
        # E[R^-\gamma e_i]
        ret_pow = ret ** (-self.gamma)
        E1 = self.inner_quadrature(ret_pow*e1)
        E2 = self.inner_quadrature(ret_pow*e2)
        # k_i
        E_pow = E ** (self.gamma-self.eta)
        k1 = self.outer_quadrature(E_pow*E1)
        k2 = self.outer_quadrature(E_pow*E2)
        # derivatives of k_i
        ret_pow = ret ** (-self.gamma-1)
        integrand = (self.gamma-self.eta) * E**(self.gamma-1)
        integrand_11 = integrand * E1**2
        integrand_11 -= self.gamma * self.inner_quadrature(ret_pow*e1**2)
        integrand_12 = integrand * E1*E2
        integrand_12 -= self.gamma * self.inner_quadrature(ret_pow*e1*e2)
        integrand_22 = integrand * E2**2
        integrand_22 -= self.gamma * self.inner_quadrature(ret_pow*e2**2)
        integrand_11 *= E_pow
        integrand_12 *= E_pow
        integrand_22 *= E_pow
        h11 = self.outer_quadrature(integrand_11)
        h12 = self.outer_quadrature(integrand_12)
        h22 = self.outer_quadrature(integrand_22)
        # final step
        h11 += self.eta * KMM**(self.eta-1) * k1**2
        h12 += self.eta * KMM**(self.eta-1) * k1*k2
        h22 += self.eta * KMM**(self.eta-1) * k2**2
        h11 *= KMM**self.eta
        h12 *= KMM**self.eta
        h22 *= KMM**self.eta

        return np.array([[h11, h12],[h12, h22]])
        
    def compute_static_weights(self):
        self.init_GaussHermite(n_Gauss=11)
        self.init_corr_nodes(cov=self.cov)
        self.init_constraints()
        self.theta_static = np.zeros((self.n_simul, self.horizon, self.n_assets))

        for t in tqdm(range(self.horizon), desc="time", total=self.horizon):
            self.init_corr_nodes_double(cov_outer=self.cov_belief[t,])
            theta_0 = np.array([1.,1.]) / 3
            for s in range(self.n_simul):
                res = opt.minimize(self.tildeH,
                                   theta_0,
                                   args=(self.mean_belief[s,t,:]),
                                   constraints=self.cons,
                                   jac=self.tildeH_Jac,
                                  )
                self.theta_static[s,t,] = res.x
                theta_0 = res.x
                
    def plot_static_portfolios(self):
        times = np.arange(self.horizon)
        mean1 = self.theta_static[:,:,0].mean(axis=0)
        std1 = self.theta_static[:,:,0].std(axis=0)
        mean2 = self.theta_static[:,:,1].mean(axis=0)
        std2 = self.theta_static[:,:,1].std(axis=0)
        participation = self.theta_static[:,:,:].sum(axis=2).mean(axis=0)
        participation_std = self.theta_static[:,:,:].sum(axis=2).std(axis=0)
    
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7,8),
                                       sharex=True, sharey=True
                                      )
        ax1.plot(mean1, label="Asset 1")
        ax1.fill_between(times, mean1+std1, mean1-std1,
                         color='C0', alpha=.2
                        )
        ax1.plot(mean2, label="Asset 2")
        ax1.fill_between(times, mean2+std2, mean2-std2,
                         color='C1', alpha=.2
                        )
        ax1.set_ylim([-0.05, 1.1])
        ax1.set_title("Assets' weights with $\pm$ std dev")
        ax1.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
        ax2.plot(participation)
        ax2.fill_between(times,
                         participation+participation_std,
                         participation-participation_std,
                         color='C0', alpha=.2
                        )
        ax2.set_title("Participation with $\pm$ std dev")
