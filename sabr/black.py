import numpy as np
from scipy.stats import norm

def black76_price(time, strike, forward, volatility, discount_factor):
    """
    Calculate Black76 price for a swaption
    
    Args:
        time: Time to expiry
        strike: Strike price
        forward: Forward price
        volatility: Volatility
        discount_factor: Discount factor
    
    Returns:
        Option price
    """
    if volatility <= 0 or time <= 0:
        return 0.0
        
    stddev = volatility * np.sqrt(time)
    d1 = (np.log(forward/strike) + 0.5 * stddev * stddev) / stddev
    d2 = d1 - stddev
    
    return discount_factor * (forward * norm.cdf(d1) - strike * norm.cdf(d2))

def black76_implied_vol(time, strike, forward, price, discount_factor, 
                       initial_guess=0.2, tolerance=1e-5, max_iter=100):
    """
    Calculate Black76 implied volatility using Newton-Raphson
    """
    if price <= 0:
        return 0.0
        
    vol = initial_guess
    for i in range(max_iter):
        price_diff = black76_price(time, strike, forward, vol, discount_factor) - price
        
        if abs(price_diff) < tolerance:
            return vol
            
        # Calculate vega
        stddev = vol * np.sqrt(time)
        d1 = (np.log(forward/strike) + 0.5 * stddev * stddev) / stddev
        vega = forward * np.sqrt(time) * norm.pdf(d1) * discount_factor
        
        if abs(vega) < 1e-10:
            return vol
            
        vol = vol - price_diff/vega
        
        if vol <= 0:
            vol = initial_guess/2
            
    return vol 