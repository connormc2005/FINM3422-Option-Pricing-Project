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


def monte_carlo_price(option_type, current_price, strike_price, time_to_maturity, interest_rate, sigma, n_paths=10000):
    Z = np.random.standard_normal(n_paths)
    final_stock_prices = current_price * np.exp(
        (interest_rate - 0.5 * sigma ** 2) * time_to_maturity + sigma * np.sqrt(time_to_maturity) * Z
    )

    if option_type == 'call':
        payoffs = np.maximum(final_stock_prices - strike_price, 0)
    elif option_type == 'put':
        payoffs = np.maximum(strike_price - final_stock_prices, 0)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    return np.exp(-interest_rate * time_to_maturity) * np.mean(payoffs)


def monte_carlo_barrier_price(current_price, strike_price, time_to_maturity, interest_rate, sigma, barrier_price, n_paths=10000, n_steps=100):
    dt = time_to_maturity / n_steps
    discount_factor = np.exp(-interest_rate * time_to_maturity)
    payoffs = []

    for _ in range(n_paths):
        prices = [current_price]
        for _ in range(n_steps):
            Z = np.random.normal()
            next_price = prices[-1] * np.exp(
                (interest_rate - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
            )
            prices.append(next_price)
        if max(prices) >= barrier_price:
            payoff = max(prices[-1] - strike_price, 0)
        else:
            payoff = 0
        payoffs.append(payoff)

    return discount_factor * np.mean(payoffs)

def binomial_barrier_price_up_and_in(
    option_type,
    current_price,
    strike_price,
    time_to_maturity,
    interest_rate,
    sigma,
    barrier_price,
    n_steps=100
):
    """
    Binomial tree pricer for European up-and-in barrier options.
    
    Parameters:
        option_type (str): 'call' or 'put'
        current_price (float): initial stock price
        strike_price (float): strike price of the option
        time_to_maturity (float): time to maturity in years
        interest_rate (float): annual risk-free rate
        sigma (float): annual volatility
        barrier_price (float): barrier level (must be breached during option life)
        n_steps (int): number of time steps in the tree

    Returns:
        float: Option price
    """

    dt = time_to_maturity / n_steps
    up = np.exp(sigma * np.sqrt(dt))
    down = 1 / up
    discount = np.exp(-interest_rate * dt)
    p = (np.exp(interest_rate * dt) - down) / (up - down)

    # Stock tree: store max price seen along each path
    stock_tree = [[0.0] * (i + 1) for i in range(n_steps + 1)]
    max_tree = [[0.0] * (i + 1) for i in range(n_steps + 1)]

    for i in range(n_steps + 1):
        for j in range(i + 1):
            price = current_price * (up ** j) * (down ** (i - j))
            stock_tree[i][j] = price
            max_price = max(price, barrier_price) if i == 0 else max(price, max_tree[i - 1][j if j == i else j])
            max_tree[i][j] = max_price

    # Terminal payoff (only if barrier was breached)
    option_tree = [[0.0] * (i + 1) for i in range(n_steps + 1)]
    for j in range(n_steps + 1):
        final_price = stock_tree[n_steps][j]
        max_price = max_tree[n_steps][j]

        if max_price >= barrier_price:
            if option_type == 'call':
                option_tree[n_steps][j] = max(final_price - strike_price, 0)
            elif option_type == 'put':
                option_tree[n_steps][j] = max(strike_price - final_price, 0)
        else:
            option_tree[n_steps][j] = 0.0

    # Backward induction
    for i in range(n_steps - 1, -1, -1):
        for j in range(i + 1):
            max_price = max_tree[i][j]
            if max_price >= barrier_price:
                continuation = discount * (
                    p * option_tree[i + 1][j + 1] +
                    (1 - p) * option_tree[i + 1][j]
                )
                option_tree[i][j] = continuation
            else:
                option_tree[i][j] = 0.0

    return option_tree[0][0]
