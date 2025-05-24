#Import Necessary Libraries 
import numpy as np
from scipy.stats import norm

def black_scholes_price(option_type, current_price, strike_price, time_to_maturity, interest_rate, sigma):
    """
    Parameters:
        option_type (str): 'call' or 'put'
        S (float): Current stock price
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free rate (annualized)
        sigma (float): Volatility of underlying (annualized)
    Returns:
        float: Option price
    """
    d1 = (np.log(current_price / strike_price) + (interest_rate + 0.5 * sigma ** 2) * time_to_maturity) / (sigma * np.sqrt(time_to_maturity))
    d2 = d1 - sigma * np.sqrt(time_to_maturity)

    if option_type == 'call':
        price = current_price * norm.cdf(d1) - strike_price * np.exp(-interest_rate * time_to_maturity) * norm.cdf(d2)
    elif option_type == 'put':
        price = strike_price * np.exp(-interest_rate * time_to_maturity) * norm.cdf(-d2) - current_price * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    
    return price


def binomial_tree_price(current_price, strike_price, time_to_maturity, interest_rate, sigma, option_type, n_steps):
    """
    Calculate the price of an American or European option (as N approaches infinity, price converges to blackscholes price) using a binomial tree.
    current_price: Current stock price (S)
    strike_price: Strike price of the option (K)
    time_to_maturity: Time to maturity in years (T)
    interest_rate: Risk-free interest rate (r)
    sigma: Implied Volatility of the underlying asset (σ)
    option_type: 'call' for call option, 'put' for put option
    n_steps: Number of steps in the binomial tree (N)

    """
    dt= T / n_steps # Time step
    u = np.exp(sigma * sqrt(dt)) # up factor 
    d = np.exp(-sigma * sqrt(dt)) # down factor 
    p = (np.exp(r * dt) - d) / (u - d) # risk-neutral probability
    discount_factor = np.exp(-r * dt) # discount factor per step

    # Create the stock price tree at terminal value 
    ST = np.array([S * (u**j) * d**(n_steps-j) for j in range (n_steps + 1)]) # Stock prices at maturity

    # Create the option value tree at terminal value
    if option_type == 'call':
        option_value = np.maximum(ST - strike_price, 0) # Call option payoff

    elif option_type == 'put':
        option_value = np.maximum(strike_price - ST, 0) # Put option payoff
    
    else: 
        raise ValueError("option_type must be 'call' or 'put'")
    
    # Backward induction to calculate option price at each node
    for i in range(n_steps - 1, -1, -1):
        for j in range(i + 1):
            ST = current_price * (u**j) * (d**(i-j)) # Stock price at node 