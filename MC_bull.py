import numpy as np
from scipy.stats import norm

# Monte Carlo functions for computing bull call spread prices and corresponding deltas

# input arguments in all cases are:
# S is the spot price of the underlying asset
# K1 is the strike price of the call option that we purchase
# K2 is the strike price of the call option that we sell
# T is the time to strike in years
# r is the risk free rate
# sigma is the volatility 
# seed is an optional argument to set the seed of the random number generator

# return is the option price and variance or delta and variance

#Bull call spread naive method (no variance reduction)

def MC_bull_call_naive(S, K1, K2, T, r, sigma, N, seed = 98765):
    #setting our random generating seed
    rg = np.random.default_rng(seed)
    #Setting stock price at time 0
    S0 = S
    #Generating N random numbers from a standard normal dist
    X = rg.normal(0,1,N)
    #setting stock price according to GBM formula
    ST = S0*np.exp((r-0.5*sigma**2)*T + sigma*np.sqrt(T)*X)
    
    # Discounted payoff of option we purchase
    fST = np.exp(-r*T)*((np.maximum(ST-K1,0))-(np.maximum(ST-K2,0)))
  
    #Price and variance of the bull call spread we are interested in
    price = np.mean(fST)
    variance = np.var(fST)
    
    #SEM = np.sqrt(variance/N)
    return(price,variance)

#Variance reduction
#Antithetic variance reduction

def MC_bull_call_ant(S, K1, K2, T, r, sigma, N, seed = 98765):

    # Set parameters
    S0 = S
    
    # construct random generator 
    rg = np.random.default_rng(seed)
    
    X = rg.normal(0,1,N)
    
    #here is the antithetic variance reduction technique
    STp = S0*np.exp((r-0.5*sigma**2)*T+sigma*np.sqrt(T)*X)
    STm = S0*np.exp((r-0.5*sigma**2)*T-sigma*np.sqrt(T)*X)
    fSTp = np.exp(-r*T)*((np.maximum(STp-K1,0)) - (np.maximum(STp-K2,0)))
    fSTm = np.exp(-r*T)*((np.maximum(STm-K1,0)) - (np.maximum(STm-K2,0)))
    
    #getting average of the 2 payoff paths
    Z = (fSTp+fSTm)/2.
    
    price = np.mean(Z)
    variance = np.var(Z)
    return price, variance

#Control Variates

def MC_bull_call_con(S,K1,K2,T,r,sigma,N, seed = 98765):

    # Set parameters
    S0 = S
        
    # construct random generator 
    rg = np.random.default_rng(seed)
    
    # known mean and variance of g=S(T)=ST
    mean_ST = S0 * np.exp(r*T)
    var_ST = mean_ST**2 * (np.exp(sigma**2*T)-1)
    
    # generate normally distributed random numbers
    X = rg.normal(0,1,N)
    #generate stock price according to GBM
    ST = S0*np.exp((r-0.5*sigma**2)*T+sigma*np.sqrt(T)*X)
    fST = np.exp(-r*T)*((np.maximum(ST-K1,0)) - (np.maximum(ST-K2,0)))
    #Control variate key part
    #covariance
    cov_fST_ST = np.mean(fST*ST) - np.mean(fST)*np.mean(ST)
    #choosing c
    c = cov_fST_ST/var_ST
    
    f_c = fST-c*(ST-mean_ST)
    
    price = np.mean(f_c)
    variance = np.var(f_c)
    return price, variance

# importance sampling

def MC_bull_call_imp(S,K1,K2,T,r,sigma,N, seed = 98765):

    # Set parameters
    S0 = S
        
    # construct random generator 
    rg = np.random.default_rng(seed)
    
    y1 = norm.cdf((np.log(K1/S0) - (r-0.5*sigma**2)*T)/(sigma*np.sqrt(T)))
    
    # The if statement below is not strictly necessary, but if y1 ~= 1 then for a call option
    # we are very far out of the money. The way importance sampling works it will always 
    # produce a small non-zero option price for y1 ~= 1. It is better to just set to zero. 
    if (y1 > 0.999):
        price = 0
        variance = 0
    else:
        rands = rg.uniform(0,1,N)
        
        Y1 = y1 + (1-y1)*rands
        X1 = norm.ppf(Y1)
        ST1 = S0*np.exp((r-0.5*sigma**2)*T+sigma*np.sqrt(T)*X1)
        
        fST1 = (1-y1)*np.exp(-r*T)*((ST1-K1)-np.maximum(ST1-K2,0))
        price = np.mean(fST1)
        variance = np.var(fST1)

    return price, variance


##########################
# BULL CALL SPREAD DELTA #
##########################

# path recycling with antiethetic vartiance reduction of delta

  
def MC_bull_call_delta_ant_path(S, K1,K2 , T, r, sigma, N, seed = 98765):
    

    # Set parameters
    S0 = S
    dS = 0.5
    # construct random generator 
    rg = np.random.default_rng(seed)
                            
    X = rg.normal(0,1,N)
    
    #Antithetic and Path Recycling
    STpr = (S0+dS)*np.exp((r-0.5*sigma**2)*T+sigma*np.sqrt(T)*X)
    STmr = (S0+dS)*np.exp((r-0.5*sigma**2)*T-sigma*np.sqrt(T)*X)
    STpl = (S0-dS)*np.exp((r-0.5*sigma**2)*T+sigma*np.sqrt(T)*X)
    STml = (S0-dS)*np.exp((r-0.5*sigma**2)*T-sigma*np.sqrt(T)*X)
    
    fSTpr = np.exp(-r*T)*(np.maximum(STpr-K1,0) - np.maximum(STpr-K2,0))
    fSTmr = np.exp(-r*T)*(np.maximum(STmr-K1,0) - np.maximum(STmr-K2,0)) 
    
    fSTpl = np.exp(-r*T)*(np.maximum(STpl-K1,0) - np.maximum(STpl-K2,0))        
    fSTml = np.exp(-r*T)*(np.maximum(STml-K1,0) - np.maximum(STml-K2,0))        
            
    fSTr = (fSTpr + fSTmr)/(2.0)
    fSTl = (fSTpl + fSTml)/(2.0)        
 
    df = (fSTr-fSTl)/(2*dS)
    delta = np.mean(df)
    variance = np.var(df)
    
    return(delta,variance)        
