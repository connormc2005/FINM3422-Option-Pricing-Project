import numpy as np
import math
from scipy.stats import norm

def black_scholes_price(option_type, current_price, strike_price, time_to_maturity, interest_rate, sigma, dividend_yield=0.0):
    """
    Calculate Black-Scholes option price
    
    Parameters:
        option_type (str): 'call' or 'put'
        current_price (float): Current stock price
        strike_price (float): Strike price
        time_to_maturity (float): Time to maturity (in years)
        interest_rate (float): Risk-free rate (annualized)
        sigma (float): Volatility of underlying (annualized)
        dividend_yield (float): Continuous annualized dividend yield (e.g., 0.02 for 2%)
    
    Returns:
        float: Option price
    """
    # Adjust current_price for continuous dividend yield. S_adj = S * e^(-qT)
    # This S_adj is the stock price effectively used for a non-dividend paying stock.
    S_adj = current_price * np.exp(-dividend_yield * time_to_maturity)

    # d1 uses S (original current_price) and (interest_rate - dividend_yield) for the drift
    d1 = (np.log(current_price / strike_price) + (interest_rate - dividend_yield + 0.5 * sigma ** 2) * time_to_maturity) / (sigma * np.sqrt(time_to_maturity))
    d2 = d1 - sigma * np.sqrt(time_to_maturity)

    if option_type == 'call':

        price = S_adj * norm.cdf(d1) - strike_price * np.exp(-interest_rate * time_to_maturity) * norm.cdf(d2)
    elif option_type == 'put':
        # Put price: K * e^(-rT) * N(-d2) - S * e^(-qT) * N(-d1)
        price = strike_price * np.exp(-interest_rate * time_to_maturity) * norm.cdf(-d2) - S_adj * norm.cdf(-d1)
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
    n_steps,          # N: Number of time steps in the binomial tree
    dividend_yield=0.0  # q: Continuous annualized dividend yield
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
    #       p = (e^{(r-q) Δt} − d) / (u − d)
    #    - Under risk-neutral pricing, the expected growth of the stock is risk-free rate.
    #    - e^{(r-q) Δt} is the growth factor on cash over Δt.
    #    - p must lie between 0 and 1 if u > e^{(r-q) Δt} > d, which is usually true.
    p = (math.exp((interest_rate - dividend_yield) * dt) - d) / (u - d)

    # Ensure p is within a valid range to prevent issues with u,d,r,q combinations
    if not (0 < p < 1):
        # print(f"Warning: Binomial p = {p:.4f} is outside (0,1). Check r, q, sigma, dt.")
        p = np.clip(p, 0.00001, 0.99999)

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

import numpy as np

def binomial_barrier_price_up_and_in(
    option_type,
    current_price,
    strike_price,
    time_to_maturity,
    interest_rate,
    sigma,
    barrier_price,
    n_steps=100,
    dividend_yield=0.0
):
    dt = time_to_maturity / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    discount = np.exp(-interest_rate * dt)
    p = (np.exp((interest_rate - dividend_yield) * dt) - d) / (u - d)

    # Ensure p is within a valid range to prevent issues with u,d,r,q combinations
    if not (0 < p < 1):
        # This can happen if (r-q)*dt is too large or too small relative to moves u and d.
        # For simplicity, clipping or raising an error. Clipping might hide model issues.
        # A more robust solution might involve adjusting u and d or using a different tree type.
        # print(f"Warning: Risk-neutral probability p = {p:.4f} is outside (0,1). Check parameters r, q, sigma, dt.")
        p = np.clip(p, 0.00001, 0.99999) # Clipping p to avoid math errors, though this indicates a parameter issue

    # Stock tree (S) and barrier breach tracker (True/False)
    S = [[0.0 for _ in range(i + 1)] for i in range(n_steps + 1)]
    barrier_breached = [[False for _ in range(i + 1)] for i in range(n_steps + 1)]

    # Initialize stock prices and barrier breach flags
    for i in range(n_steps + 1):
        for j in range(i + 1):
            S_ij = current_price * (u ** j) * (d ** (i - j))
            S[i][j] = S_ij

            # For i == 0, we're at the root node
            if i == 0:
                barrier_breached[i][j] = S_ij >= barrier_price
            else:
                from_left = False
                from_right = False

                if j < i:
                    from_left = barrier_breached[i - 1][j]
                if j > 0:
                    from_right = barrier_breached[i - 1][j - 1]

                # Breach occurs if this node hits the barrier or any ancestor did
                barrier_breached[i][j] = (S_ij >= barrier_price) or from_left or from_right

    # Terminal payoff - only if barrier was breached
    V = [[0.0 for _ in range(i + 1)] for i in range(n_steps + 1)]
    for j in range(n_steps + 1):
        if barrier_breached[n_steps][j]:
            ST = S[n_steps][j]
            if option_type == 'call':
                V[n_steps][j] = max(ST - strike_price, 0)
            elif option_type == 'put':
                V[n_steps][j] = max(strike_price - ST, 0)

    # Backward induction
    for i in range(n_steps - 1, -1, -1):
        for j in range(i + 1):
            # Calculate continuation value regardless of current breach status
            continuation_value = discount * (
                p * V[i + 1][j + 1] +
                (1 - p) * V[i + 1][j]
            )
            
            # For up-and-in barrier: option has value if barrier was breached
            # OR if there's potential value from paths that breach the barrier later
            if barrier_breached[i][j]:
                # Barrier already breached - option is active
                V[i][j] = continuation_value
            else:
                # Barrier not yet breached - only has value from future breach paths
                V[i][j] = continuation_value

    return V[0][0]

def monte_carlo_price(option_type, current_price, strike_price, time_to_maturity, interest_rate, sigma, n_paths=10000):
    """
    Monte Carlo pricing for European options using GBM simulation.
    """
    # Generate random normal values for all paths at once
    Z = np.random.normal(0, 1, size=n_paths)
    
    # Calculate final stock prices using GBM formula
    final_stock_prices = current_price * np.exp(
        (interest_rate - 0.5 * sigma ** 2) * time_to_maturity + sigma * np.sqrt(time_to_maturity) * Z
    )

    # Calculate payoffs
    if option_type == 'call':
        payoffs = np.maximum(final_stock_prices - strike_price, 0)
    elif option_type == 'put':
        payoffs = np.maximum(strike_price - final_stock_prices, 0)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    # Discount back to present value
    return np.exp(-interest_rate * time_to_maturity) * np.mean(payoffs)

def monte_carlo_barrier_price(current_price, strike_price, time_to_maturity, interest_rate, sigma, barrier_price, n_paths=10000, n_steps=100, dividend_yield=0.0):
    """
    Monte Carlo pricing for up-and-in barrier options using GBM path simulation.
    """
    dt = time_to_maturity / n_steps
    discount_factor = np.exp(-interest_rate * time_to_maturity)
    
    # Initialize price matrix (rows=time steps, columns=paths)
    S = np.zeros((n_steps + 1, n_paths))
    S[0] = current_price
    
    # Generate random normal values for all paths and time steps at once
    # Consider antithetic variates for variance reduction if stability is an issue
    # Z_half = np.random.normal(0, 1, size=(n_steps, n_paths // 2))
    # Z = np.concatenate((Z_half, -Z_half), axis=1)
    # if n_paths % 2 != 0: # If odd n_paths, add one more column from new randoms
    #     Z_extra = np.random.normal(0, 1, size=(n_steps,1))
    #     Z = np.concatenate((Z, Z_extra), axis=1)
    Z = np.random.normal(0, 1, size=(n_steps, n_paths))
    
    # Simulate all paths using GBM with dividend yield adjustment
    drift = (interest_rate - dividend_yield - 0.5 * sigma**2) * dt
    diffusion_stoch = sigma * np.sqrt(dt) * Z

    for i in range(1, n_steps + 1):
        S[i] = S[i-1] * np.exp(drift + diffusion_stoch[i-1])
    
    # Check barrier breach for each path
    barrier_breached = np.max(S, axis=0) >= barrier_price
    
    # Calculate payoffs only for paths that breached the barrier
    final_prices = S[-1]  # Final prices for all paths
    payoffs = np.where(barrier_breached, np.maximum(final_prices - strike_price, 0), 0)
    
    return discount_factor * np.mean(payoffs)

def monte_carlo_basket_price(current_prices, weights, strike_price, time_to_maturity, interest_rate, sigma, correlation_matrix, option_type='call', n_paths=10000, dividend_yields=None):
    """
    Monte Carlo pricing for European basket options using GBM simulation.
    
    Parameters:
    ...
    sigma (list/np.array): List of annualized volatilities for each asset.
    ...
    dividend_yields (list/np.array, optional): List of continuous annualized dividend yields for each asset. Defaults to zeros.
    """
    n_assets = len(current_prices)
    dt = time_to_maturity # For a simple European option, we simulate directly to maturity
    
    # Ensure current_prices, weights, sigma (volatilities), and dividend_yields are numpy arrays
    current_prices = np.array(current_prices)
    weights = np.array(weights)
    sigmas_arr = np.array(sigma) # Renaming to avoid conflict if 'sigma' is used for single asset elsewhere

    if dividend_yields is None:
        dividend_yields_arr = np.zeros(n_assets)
    else:
        dividend_yields_arr = np.array(dividend_yields)
        if len(dividend_yields_arr) != n_assets:
            raise ValueError("Length of dividend_yields must match number of assets.")

    # Cholesky decomposition of the correlation matrix
    L = np.linalg.cholesky(correlation_matrix)
    
    # Generate correlated random normal values for all paths and assets
    # Z_uncorrelated shape: (n_assets, n_paths)
    Z_uncorrelated = np.random.normal(0, 1, size=(n_assets, n_paths))
    Z_correlated = L @ Z_uncorrelated # Correlated random numbers
    
    final_asset_prices = np.zeros((n_assets, n_paths))
    
    for i in range(n_assets):
        # Adjusted drift for GBM with dividends: (r - q_i - 0.5 * sigma_i^2)T is correct for log S_T/S_0
        drift = (interest_rate - dividend_yields_arr[i] - 0.5 * sigmas_arr[i]**2) * time_to_maturity
        diffusion = sigmas_arr[i] * np.sqrt(time_to_maturity) * Z_correlated[i, :]
        final_asset_prices[i, :] = current_prices[i] * np.exp(drift + diffusion)
        
    # Calculate basket value at maturity for each path
    basket_values_at_maturity = np.sum(final_asset_prices * weights[:, np.newaxis], axis=0)
    
    # Calculate option payoff for each path
    if option_type.lower() == 'call':
        payoffs = np.maximum(basket_values_at_maturity - strike_price, 0)
    elif option_type.lower() == 'put':
        payoffs = np.maximum(strike_price - basket_values_at_maturity, 0)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
        
    # Discounted average payoff
    option_price = np.exp(-interest_rate * time_to_maturity) * np.mean(payoffs)
    
    return option_price

def get_zero_rate(time_to_maturity):
    """
    Bootstrap yield curve to get zero rate for any maturity
    Converts bank bill discount yields and par yields to zero rates
    """
    maturities = [1/12, 3/12, 6/12, 1, 2, 3, 5]  # Time periods we have data for
    bb_yields = [0.037925, 0.03797, 0.03855]  # Bank bill discount yields (short term)
    gov_yields = [0.0348, 0.03374, 0.03431, 0.03659]  # Government par yields (long term)
    zero_rates = []  # Will store our calculated zero rates
    
    # Convert bank bills to zero rates using formula: r = (F/P)^(1/T) - 1
    for i in range(3):  # Process 1M, 3M, 6M bank bills
        T = maturities[i]
        price = 1 - bb_yields[i] * T  # Bank bills sold at discount to face value
        zero_rate = (1/price)**(1/T) - 1  # Extract pure zero rate
        zero_rates.append(zero_rate)
    
    # Bootstrap government bonds (they pay semi-annual coupons)
    for i in range(4):  # Process 1Y, 2Y, 3Y, 5Y bonds
        maturity = maturities[i + 3]
        par_yield = gov_yields[i]
        
        if maturity == 1:  # Special case for 1-year bond
            coupon = par_yield * 100 / 2  # Coupon paid every 6 months
            from scipy.optimize import fsolve
            
            def bond_price_error(z):  # Function to solve bond pricing equation
                discount_factor = 1 + z/2  # Semi-annual compounding
                pv = coupon / discount_factor + (coupon + 100) / (discount_factor**2)
                return pv - 100  # Bond trades at par (price = 100)
            
            z_1y = fsolve(bond_price_error, par_yield)[0]  # Solve for zero rate
            zero_rates.append(z_1y)
        else:  # Multi-year bonds need bootstrapping
            coupon = par_yield * 100 / 2  # Semi-annual coupon payment
            periods = int(maturity * 2)  # Number of coupon payments
            
            pv_coupons = 0  # Present value of all intermediate coupons
            for p in range(1, periods):  # Discount each coupon except the last
                t = p / 2  # Time when this coupon is paid
                z = _get_rate(t, maturities[:len(zero_rates)], zero_rates)  # Get zero rate for this time
                pv_coupons += coupon / (1 + z/2)**p  # Discount this coupon
            
            # Solve for the final zero rate that makes bond price = 100
            final_payment = coupon + 100  # Last coupon plus principal repayment
            remaining_pv = 100 - pv_coupons  # What the final payment must be worth today
            discount_factor = final_payment / remaining_pv
            z = 2 * (discount_factor**(1/periods) - 1)  # Convert to annual rate
            zero_rates.append(z)
    
    return _get_rate(time_to_maturity, maturities, zero_rates)  # Find rate for requested maturity

def _get_rate(target_time, times, rates):
    """Helper function: uses linear interpolation to find rate for any time period"""
    if target_time <= times[0]:
        return rates[0]  # If before our data, use first rate
    if target_time >= times[-1]:
        return rates[-1]  # If after our data, use last rate
    
    # Find which two data points our target falls between
    for i in range(len(times) - 1):
        if times[i] <= target_time <= times[i + 1]:
            # Calculate how far between the two points we are (0 to 1)
            weight = (target_time - times[i]) / (times[i + 1] - times[i])
            # Blend the two rates based on the weight
            return rates[i] + weight * (rates[i + 1] - rates[i])
    
    return rates[-1]  # Fallback

def black_scholes_greeks(option_type, current_price, strike_price, time_to_maturity, interest_rate, sigma):
    """
    Calculate Black-Scholes Greeks
    
    Returns:
        dict: Dictionary containing delta, gamma, theta, vega, rho
    """
    d1 = (np.log(current_price / strike_price) + (interest_rate + 0.5 * sigma ** 2) * time_to_maturity) / (sigma * np.sqrt(time_to_maturity))
    d2 = d1 - sigma * np.sqrt(time_to_maturity)
    
    # Common terms
    nd1 = norm.cdf(d1)
    nd2 = norm.cdf(d2)
    npd1 = norm.pdf(d1)
    
    if option_type == 'call':
        delta = nd1
        theta = (-(current_price * npd1 * sigma) / (2 * np.sqrt(time_to_maturity)) 
                - interest_rate * strike_price * np.exp(-interest_rate * time_to_maturity) * nd2)
        rho = strike_price * time_to_maturity * np.exp(-interest_rate * time_to_maturity) * nd2
    elif option_type == 'put':
        delta = nd1 - 1
        theta = (-(current_price * npd1 * sigma) / (2 * np.sqrt(time_to_maturity)) 
                + interest_rate * strike_price * np.exp(-interest_rate * time_to_maturity) * norm.cdf(-d2))
        rho = -strike_price * time_to_maturity * np.exp(-interest_rate * time_to_maturity) * norm.cdf(-d2)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    
    # Gamma and Vega are the same for calls and puts
    gamma = npd1 / (current_price * sigma * np.sqrt(time_to_maturity))
    vega = current_price * npd1 * np.sqrt(time_to_maturity)