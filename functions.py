import numpy as np
import math
from scipy.stats import norm

def black_scholes_price(option_type, current_price, strike_price, time_to_maturity, interest_rate, sigma):
    """
    Calculate Black-Scholes option price
    
    Parameters:
        option_type (str): 'call' or 'put'
        current_price (float): Current stock price
        strike_price (float): Strike price
        time_to_maturity (float): Time to maturity (in years)
        interest_rate (float): Risk-free rate (annualized)
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


def binomial_tree_price(
    current_price,    # S: Current stock price (today's observable price) 
    strike_price,     # K: The strike price of the option
    time_to_maturity, # T: Time to expiration in years (e.g., 0.5 for six months)
    interest_rate,    # r: Continuously-compounded risk-free rate (e.g., 0.025 for 2.5%)
    sigma,            # σ: Implied volatility of the underlying (in decimal form, e.g., 0.20)
    option_type,      # 'call' or 'put'
    american,         # Boolean: True if American-style, False if European
    n_steps           # N: Number of time steps in the binomial tree
):
    """
    Calculate the price of a European or American option using a Cox-Ross-Rubinstein binomial tree.

    Arguments:
    - current_price:      S, current price of the underlying asset.
    - strike_price:       K, the option strike price.
    - time_to_maturity:   T, in years (e.g. 1.0 means one year until expiry).
    - interest_rate:      r, continuously-compounded annual risk-free rate (as a decimal).
    - sigma:              σ, annualized volatility of the underlying (as a decimal).
    - option_type:        'call' or 'put'.
    - american:           True for American option (can exercise early), False for European.
    - n_steps:            N, number of binomial steps. As N→∞, result converges to Black-Scholes.

    Returns:
    - The computed option price (a float).
    """

    # 1) Compute the size of each time step (Δt = T / N).
    dt = time_to_maturity / n_steps
    #    - We split the total time to maturity into N equal intervals.
    #    - For example, if T = 1 year and N = 100, each step represents 0.01 years (≈3.65 days).

    # 2) Compute up-factor (u) and down-factor (d) using the standard CRR model:
    #       u = exp(σ * sqrt(Δt))
    #       d = exp(−σ * sqrt(Δt))
    #    - Intuition: In each small step of length Δt, the stock either moves up by factor u or down by factor d.
    #    - These factors are chosen so that the variance over Δt matches σ^2 * Δt.
    u = math.exp(sigma * math.sqrt(dt))
    d = math.exp(-sigma * math.sqrt(dt))
    #    - Note that d = 1/u. You can check: d = exp(-σ√Δt) = 1 / exp(σ√Δt) = 1 / u.

    # 3) Compute the risk-neutral probability p of an "up" move in each step:
    #       p = (e^{r Δt} − d) / (u − d)
    #    - Under risk-neutral pricing, the expected growth of the stock is risk-free rate.
    #    - e^{r Δt} is the growth factor on cash over Δt.
    #    - p must lie between 0 and 1 if u > e^{r Δt} > d, which is usually true in practice.
    p = (math.exp(interest_rate * dt) - d) / (u - d)

    # 4) The discount factor per single step is e^{−r Δt}, used to back discount expected payoffs:
    discount = math.exp(-interest_rate * dt)

    # 5) Build the array of stock prices at maturity (time T) for all possible up–down paths.
    #    - At expiry (step N), the stock could have gone up j times and down (N − j) times, for j = 0,1,...,N.
    #    - If j = number of up-moves, then number of down-moves is (N−j).
    #    - Stock price at node j at time T: S * u^j * d^{(N−j)}.
    terminal_ST = np.array([
        current_price * (u ** j) * (d ** (n_steps - j))
        for j in range(n_steps + 1)
    ])
    #    - This creates a length-(N+1) array: [S·d^N, S·u·d^{N−1}, S·u^2·d^{N−2}, ..., S·u^N].

    # 6) Compute the option payoff at each terminal node:
    if option_type.lower() == 'call':
        # For a call at expiry: max(ST − K, 0)
        option_values = np.maximum(terminal_ST - strike_price, 0.0)
    elif option_type.lower() == 'put':
        # For a put at expiry: max(K − ST, 0)
        option_values = np.maximum(strike_price - terminal_ST, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # 7) Now perform backward induction to roll the option values back from time N to time 0.
    #    We will overwrite option_values layer by layer:
    #    - At step i (from N−1 down to 0), there are (i+1) nodes: j = 0,1,...,i.
    #    - At node (i,j), stock price is S * u^j * d^{(i−j)}.
    #    - The continuation value = discounted expected value: e^{−r Δt} [p · V_{i+1,j+1} + (1−p) · V_{i+1,j}].
    #    - For American options, we must compare that continuation value to the exercise value:
    #          exercise_value = max( S_node − K, 0 ) for call, or max( K − S_node, 0 ) for put.
    #    - The node value = max(continuation_value, exercise_value) if American; otherwise = continuation_value (European).

    # Loop i from N−1 down to 0:
    for i in range(n_steps - 1, -1, -1):
        # At level i, there are (i+1) possible states (j = 0..i).
        # We create a new array next_values of length (i+1) to store option values at step i.
        new_values = np.zeros(i + 1)

        for j in range(i + 1):
            # 7a) Compute the stock price at node (i, j):
            #      j = number of up-moves so far, so (i − j) = number of down-moves so far.
            ST_node = current_price * (u ** j) * (d ** (i - j))

            # 7b) Continuation value by risk-neutral expectation:
            #      = e^{−r Δt} [ p * value_if_up + (1−p) * value_if_down ]
            #      At time (i+1), "up" leads to node (i+1, j+1), "down" leads to (i+1, j).
            cont_value = (
                discount * (p * option_values[j + 1] + (1 - p) * option_values[j])
            )

            if american:
                # 7c) If it's American, compute immediate exercise payoff at this node:
                if option_type.lower() == 'call':
                    exercise_value = max(ST_node - strike_price, 0.0)
                else:  # put
                    exercise_value = max(strike_price - ST_node, 0.0)

                # 7d) The option value is the maximum of (continuation, exercise):
                node_value = max(cont_value, exercise_value)
            else:
                # 7e) If it's European, you cannot exercise early → must take continuation value
                node_value = cont_value

            new_values[j] = node_value

        # 7f) Move one step back: replace option_values with new_values for next iteration
        option_values = new_values.copy()

    # At the end of backward induction, option_values has length 1: it's the value at time 0.
    return float(option_values[0])

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
