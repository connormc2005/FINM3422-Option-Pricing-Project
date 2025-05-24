#Import Necessary Libraries 
import numpy as np

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