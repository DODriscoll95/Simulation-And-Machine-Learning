import numpy as np
from scipy.stats import norm

# defining our volatility function

# input arguments in all cases are:
# S is the spot price of the underlying asset
# K1 is the strike price of the option
# Sb is the barrier value
# T is the time to strike in years
# r is the risk free rate
# sigma values are defined according to our local volatility function with overall volatility named sigma 

def volatility(S,t,sigma_0,sigma_1,sigma_2):
    # for testing, set sigma_1 = sigma_2 = 0
    # leave this in place and uncomment as necessary
    #sigma_1 = 0.0
    #sigma_2 = 0.0
    sigma = (sigma_0*(1+sigma_1*np.cos(np.pi*2*t)))*((1 + sigma_2*np.exp(-S/50)))
    return(sigma)


#Down and out naive monte carlo

def SDE_downandout_put(S0,Sb, K, T, r, Npaths,sigma_0,sigma_1,sigma_2, seed = 98765):
    
    #set random number generator
    rg = np.random.default_rng(seed)
    
    #Euler time stepping of one day 260 days in a year for T years = Nsteps
    Nsteps = int(260 * T)
    
    t, dt = np.linspace(0, T, Nsteps + 1, retstep=True)
    S = np.zeros((Nsteps + 1, Npaths))
    #Setting dW for the GBM
    dW = np.sqrt(dt) * rg.normal(0,1,(Nsteps, Npaths))
    
    # Time step starting from initial condition
    S[0,:] = S0  
    for n in range(Nsteps):
        #Setting stock price at t+1 
        S[n+1,:] = S[n,:] * (1 + r*dt + volatility(S[n,:], t[n],sigma_0,sigma_1,sigma_2) * dW[n,:])
    
    # discounted payoff based on S at final time
    min_of_S = np.amin(S,axis = 0)
    
    #Check If barrier conditions were statisfied
    indicator = np.heaviside(min_of_S-Sb,0)
    fST = np.exp(-r*T)*np.maximum(K-S[Nsteps,:],0)*indicator
    
    price = np.mean(fST)
    variance = np.var(fST)
    return price, variance


#Down and out Put Antithetic variance reduction
def SDE_downandout_put_ant(S0,Sb, K, T, r, Npaths, sigma_0, sigma_1, sigma_2, seed = 98765):
    #As before
    rg = np.random.default_rng(seed)
    Nsteps = int(260 * T)
    t, dt = np.linspace(0, T, Nsteps + 1, retstep=True)
    #Set two arrays this time for two stock price paths
    Sp = np.zeros((Nsteps + 1, Npaths))
    Sm = np.zeros((Nsteps + 1, Npaths))
    dW = np.sqrt(dt) * rg.normal(0,1,(Nsteps, Npaths))
    
    # Time step starting from initial condition
    Sp[0,:] = S0  
    Sm[0,:] = S0  
    for n in range(Nsteps):
        #Plus and Minus paths for the antithetic variance reduciton
        Sp[n+1,:] = Sp[n,:] * (1 + r*dt + volatility(Sp[n,:], t[n],sigma_0,sigma_1,sigma_2) * dW[n,:])
        Sm[n+1,:] = Sm[n,:] * (1 + r*dt - volatility(Sm[n,:], t[n],sigma_0,sigma_1,sigma_2) * dW[n,:])
    
    #Minimum values Inidcators for barrier conditions for each path
    min_of_Sp = np.amin(Sp,axis = 0)
    indicatorp = np.heaviside(min_of_Sp-Sb ,0)
    min_of_Sm = np.amin(Sm,axis = 0)
    indicatorm = np.heaviside(min_of_Sm-Sb ,0)
    
    
    # discounted payoff based on S at final time
    fSTp = np.exp(-r*T)*np.maximum(K-Sp[Nsteps,:],0)*indicatorp 
    fSTm = np.exp(-r*T)*np.maximum(K-Sm[Nsteps,:],0)*indicatorm 

    #Average of the two path payoffs
    Z = (fSTp + fSTm)/2.0
    
    price = np.mean(Z)
    variance = np.var(Z)
    return price, variance


def MC_euro_put_delta(S, K, T, r, sigma, N, seed = 98765):

    dS = 0.5
    # construct random generator 
    rg = np.random.default_rng(seed)
    X = rg.normal(0,1,N)
    
    S0 = S + dS
    ST_up = S0*np.exp((r-0.5*sigma**2)*T+sigma*np.sqrt(T)*X)
    fST_up = np.exp(-r*T)*np.maximum(K-ST_up,0)
    
    S0 = S - dS
    ST_dn = S0*np.exp((r-0.5*sigma**2)*T+sigma*np.sqrt(T)*X)
    fST_dn = np.exp(-r*T)*np.maximum(K-ST_dn,0)
  
    dfST_dS = (fST_up-fST_dn)/(2*dS)
    
    delta = np.mean(dfST_dS)
    variance = np.var(dfST_dS)
    return delta, variance

#Delta of Down and out Put
def SDE_downandout_put_delta(S0,Sb, K, T, r, Npaths, sigma_0, sigma_1, sigma_2, seed = 98765):
    
    rg = np.random.default_rng(seed)
    dS = 0.5
    
    Nsteps = int(260 * T)
    
    t, dt = np.linspace(0, T, Nsteps + 1, retstep=True)
    Su = np.zeros((Nsteps + 1, Npaths))
    Sd = np.zeros((Nsteps + 1, Npaths))
    dW = np.sqrt(dt) * rg.normal(0,1,(Nsteps, Npaths))
    
    
    #Offsetting starting price for different paths
    Su[0,:] = S0+dS  
    Sd[0,:] = S0-dS
    for n in range(Nsteps):
        Su[n+1,:] = Su[n,:] * (1 + r*dt + volatility(Su[n,:], t[n],sigma_0,sigma_1,sigma_2) * dW[n,:])
        Sd[n+1,:] = Sd[n,:] * (1 + r*dt + volatility(Sd[n,:], t[n],sigma_0,sigma_1,sigma_2) * dW[n,:]) 
    
    min_of_Su = np.amin(Su,axis = 0)
    indicatoru = np.heaviside(min_of_Su-Sb ,0)
    min_of_Sd = np.amin(Sd,axis = 0)
    indicatord = np.heaviside(min_of_Sd-Sb ,0)
    
    
    # discounted payoff based on S at final time for both paths
    fSTu = np.exp(-r*T)*np.maximum(K-Su[Nsteps,:],0)*indicatoru 
    fSTd = np.exp(-r*T)*np.maximum(K-Sd[Nsteps,:],0)*indicatord
    
    #Finite Difference
    dfST_dS = (fSTu - fSTd)/(2*dS)
    
    delta = np.mean(dfST_dS)
    variance = np.var(dfST_dS)
    return(delta,variance)




