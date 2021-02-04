import numpy as np
from scipy.stats import norm

# BlackScholes functions for computing call and put option prices and corresponding deltas
# Also calculates the price of a bull call spread
# input arguments in all cases are:
# S is the spot price of the underlying asset
# K1 is the strike price of the option we buy
# K2 is the strike price of the option we sell
# T is the time to strike in years
# r is the risk free rate
# sigma is the volatility 

# returns is the option price
# Euro Call (no variance reduction)

#For illustration here's the BS Calculation for a european call

def BS_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2 /2) * T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)


#Bullcall spread

#Here K1 is the strike price of the option we purchase, K2 is the strike of the options we sell
# K1 < K2

# Calculate the payoff for both options using BS and take one from the other

# Overall pay off = max(ST - K1,0) - max(ST - K2,0)

def BS_bull_call(S,K1,K2,T,r,sigma):
    
    
    K1d1 = (np.log(S/K1) + (r + sigma**2 /2) * T)/(sigma*np.sqrt(T))
    K1d2 = K1d1 - sigma * np.sqrt(T)
    
    K2d1 = (np.log(S/K2) + (r + sigma**2 /2) * T)/(sigma*np.sqrt(T))
    K2d2 = K2d1 - sigma * np.sqrt(T)
    
    S1 = S * norm.cdf(K1d1) - K1 * np.exp(-r*T) * norm.cdf(K1d2)
    S2 = S * norm.cdf(K2d1) - K2 * np.exp(-r*T) * norm.cdf(K2d2)
    if (K1 < K2):
        price = S1 - S2
        return(price)
    else:
        return("Try again, K1 is not less than K2")

#Delta of bull call using BS
#Essentially taking the delta of the option we short from the delta of the option we buy
    
def BS_bull_call_delta(S,K1,K2,T,r,sigma):
    
    K1d1 = (np.log(S/K1) + (r + sigma**2 /2) * T)/(sigma*np.sqrt(T))
    K1d2 = K1d1 - sigma * np.sqrt(T)
    K2d1 = (np.log(S/K2) + (r + sigma**2 /2) * T)/(sigma*np.sqrt(T))
    K2d2 = K2d1 - sigma * np.sqrt(T)
    
    delta = (norm.cdf(K1d1) - norm.cdf(K2d1))
    return(delta)