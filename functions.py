
import numpy as np
from scipy.stats import norm

# =============================
# 🔹 BLACK-SCHOLES PRICER
# =============================

def black_scholes_price(option_type, S, K, T, r, sigma):
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
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    
    return price

# =============================
# 🔹 BINOMIAL TREE PRICER
# =============================

def binomial_tree_price(option_type, S, K, T, r, sigma, N=100):
    """
    Parameters:
        option_type (str): 'call' or 'put'
        S (float): Initial stock price
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free rate
        sigma (float): Volatility
        N (int): Number of steps in the tree
    Returns:
        float: Option price using CRR binomial model (supports American options)
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    q = (np.exp(r * dt) - d) / (u - d)
    
    # Stock price tree
    ST = np.array([S * (u ** j) * (d ** (N - j)) for j in range(N + 1)])

    # Option value at maturity
    if option_type == 'call':
        option_values = np.maximum(0, ST - K)
    elif option_type == 'put':
        option_values = np.maximum(0, K - ST)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    # Work backwards through tree
    for i in range(N - 1, -1, -1):
        option_values = np.exp(-r * dt) * (q * option_values[1:] + (1 - q) * option_values[:-1])
        ST = ST[:i+1] / u
        if option_type == 'call':
            option_values = np.maximum(option_values, ST - K)
        else:
            option_values = np.maximum(option_values, K - ST)

    return option_values[0]

# =============================
# 🔹 MONTE CARLO PRICER
# =============================

def monte_carlo_price(option_type, S, K, T, r, sigma, n_paths=10000):
    """
    Monte Carlo pricer for European call/put options
    """
    Z = np.random.standard_normal(n_paths)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    if option_type == 'call':
        payoffs = np.maximum(ST - K, 0)
    elif option_type == 'put':
        payoffs = np.maximum(K - ST, 0)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    return np.exp(-r * T) * np.mean(payoffs)

# =============================
# 🔹 MONTE CARLO BARRIER OPTION
# =============================

def monte_carlo_barrier_price(S, K, T, r, sigma, barrier, n_paths=10000, steps=100):
    """
    European up-and-in barrier call option
    Barrier must be breached during path
    """
    dt = T / steps
    discount_factor = np.exp(-r * T)

    payoffs = []
    for _ in range(n_paths):
        prices = [S]
        for _ in range(steps):
            Z = np.random.normal()
            S_next = prices[-1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
            prices.append(S_next)
        if max(prices) >= barrier:
            payoff = max(prices[-1] - K, 0)
        else:
            payoff = 0
        payoffs.append(payoff)

    return discount_factor * np.mean(payoffs)

# =============================
# 🔹 MONTE CARLO BASKET OPTION
# =============================

def monte_carlo_basket_price(spot_prices, weights, K, T, r, sigmas, n_paths=10000):
    """
    Basket option pricer assuming independence (no correlation)
    """
    n_assets = len(spot_prices)
    payoffs = []

    for _ in range(n_paths):
        ST = []
        for i in range(n_assets):
            Z = np.random.normal()
            ST_i = spot_prices[i] * np.exp((r - 0.5 * sigmas[i] ** 2) * T + sigmas[i] * np.sqrt(T) * Z)
            ST.append(ST_i)
        basket_value = np.dot(weights, ST)
        payoff = max(basket_value - K, 0)
        payoffs.append(payoff)

    return np.exp(-r * T) * np.mean(payoffs)