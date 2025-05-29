import numpy as np
import math
from scipy.stats import norm

def black_scholes_price(option_type, current_price, strike_price, time_to_maturity, interest_rate, sigma, dividend_yield=0.0):
    """
    Calculate Black-Scholes option price using the famous pricing formula
    
    The Black-Scholes model assumes stock prices follow geometric Brownian motion
    and provides an analytical (exact mathematical) solution for European options.
    """
    # Adjust stock price for dividends: companies paying dividends reduce stock value by dividend amount
    S_adj = current_price * np.exp(-dividend_yield * time_to_maturity)  # e^(-qT) discounts future dividends

    # Calculate d1 and d2: these are standardized variables from the Black-Scholes formula
    # They represent how many standard deviations the stock price is from the strike price
    d1 = (np.log(current_price / strike_price) + (interest_rate - dividend_yield + 0.5 * sigma ** 2) * time_to_maturity) / (sigma * np.sqrt(time_to_maturity))
    d2 = d1 - sigma * np.sqrt(time_to_maturity)  # d2 = d1 - σ√T

    if option_type == 'call':
        # Call option: right to BUY at strike price. Worth more when stock price > strike price
        price = S_adj * norm.cdf(d1) - strike_price * np.exp(-interest_rate * time_to_maturity) * norm.cdf(d2)
    elif option_type == 'put':
        # Put option: right to SELL at strike price. Worth more when stock price < strike price
        price = strike_price * np.exp(-interest_rate * time_to_maturity) * norm.cdf(-d2) - S_adj * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    
    return price


def binomial_tree_price(
    current_price,    # S: Current stock price 
    strike_price,     # K: Strike price 
    time_to_maturity, # T: Time until expiration in years 
    interest_rate,    # r: Risk-free interest rate 
    sigma,            # σ: Volatility 
    option_type,      # 'call' (right to buy) or 'put' (right to sell)
    american,         # True = can exercise early, False = only at expiration
    n_steps,          # Number of time periods to split the tree into (more steps = more accurate)
    dividend_yield=0.0  # Annual dividend rate (e.g., 0.03 = 3% dividend yield)
):
    """
    Price options using Cox-Ross-Rubinstein binomial tree method
    
    This method models stock price movements as a series of up/down moves in a tree structure.
    Each time step, stock can move up by factor 'u' or down by factor 'd'.
    We work backwards from expiration to find today's option value.
    """
    
    dt = time_to_maturity / n_steps  # Time step size: divide total time into n equal periods
    
    # Calculate up and down movement factors using Cox-Ross-Rubinstein formulas
    # These ensure the tree matches the stock's volatility over time
    u = math.exp(sigma * math.sqrt(dt))      # Up factor: stock multiplied by this when it goes up
    d = math.exp(-sigma * math.sqrt(dt))     # Down factor: stock multiplied by this when it goes down (d = 1/u)
    
    # Risk-neutral probability: not the real probability, but mathematical probability that makes pricing work
    # In risk-neutral world, all assets earn the risk-free rate on average
    p = (math.exp((interest_rate - dividend_yield) * dt) - d) / (u - d)
    
    # Safety check: probability must be between 0 and 1, otherwise our model parameters are inconsistent
    if not (0 < p < 1):
        p = np.clip(p, 0.00001, 0.99999)  # np.clip forces value into valid range

    discount = math.exp(-interest_rate * dt)  # Factor to discount future value back one time step

    # Build array of all possible stock prices at expiration (terminal nodes of the tree)
    # After n_steps: j up-moves and (n_steps - j) down-moves gives S * u^j * d^(n_steps-j)
    terminal_ST = np.array([
        current_price * (u ** j) * (d ** (n_steps - j))
        for j in range(n_steps + 1)  # j goes from 0 to n_steps (n_steps+1 possible outcomes)
    ])

    # Calculate option payoff at each terminal node (what option is worth at expiration)
    if option_type.lower() == 'call':
        # Call option payoff: max(Stock_Price - Strike_Price, 0)
        # Only worth something if stock price > strike price
        option_values = np.maximum(terminal_ST - strike_price, 0.0)
    elif option_type.lower() == 'put':
        # Put option payoff: max(Strike_Price - Stock_Price, 0)
        # Only worth something if stock price < strike price
        option_values = np.maximum(strike_price - terminal_ST, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # Backward induction: work backwards through tree from expiration to today
    # At each node, calculate option value based on future possible values
    for i in range(n_steps - 1, -1, -1):  # Count backwards from n_steps-1 to 0
        new_values = np.zeros(i + 1)  # At time step i, there are i+1 possible stock price nodes

        for j in range(i + 1):  # Loop through each node at this time step
            # Calculate stock price at this node (j up-moves, i-j down-moves so far)
            ST_node = current_price * (u ** j) * (d ** (i - j))
            
            # Continuation value: expected option value if we DON'T exercise today
            # Take probability-weighted average of up and down outcomes, then discount back
            cont_value = discount * (p * option_values[j + 1] + (1 - p) * option_values[j])

            if american:
                # For American options: compare continuation value vs immediate exercise value
                if option_type.lower() == 'call':
                    exercise_value = max(ST_node - strike_price, 0.0)  # Value if we exercise call today
                else:  # put
                    exercise_value = max(strike_price - ST_node, 0.0)  # Value if we exercise put today
                
                # Take the maximum: either exercise now or wait (optimal decision)
                node_value = max(cont_value, exercise_value)
            else:
                # European options: can only exercise at expiration, so take continuation value
                node_value = cont_value

            new_values[j] = node_value

        option_values = new_values.copy()  # Move to next time step backwards

    return float(option_values[0])  # Return option value at root node (today's price)

import numpy as np

def binomial_barrier_price_up_and_in(
    option_type,
    current_price,
    strike_price,
    time_to_maturity,
    interest_rate,
    sigma,
    barrier_price,      # Barrier level: option only becomes active if stock hits this price
    n_steps=100,
    dividend_yield=0.0
):
    """
    Price up-and-in barrier options using binomial tree
    
    Barrier option: only becomes a regular option if stock price touches the barrier level.
    Up-and-in: barrier is above current stock price, option activates if stock moves up to barrier.
    """
    dt = time_to_maturity / n_steps
    u = np.exp(sigma * np.sqrt(dt))      # Up movement factor
    d = 1 / u                            # Down movement factor (inverse of up)
    discount = np.exp(-interest_rate * dt)
    p = (np.exp((interest_rate - dividend_yield) * dt) - d) / (u - d)  # Risk-neutral probability

    # Ensure probability is valid (between 0 and 1)
    if not (0 < p < 1):
        p = np.clip(p, 0.00001, 0.99999)

    # Initialize 2D lists: [time_step][up_moves] for stock prices and barrier tracking
    # Python list comprehension creates nested lists efficiently
    S = [[0.0 for _ in range(i + 1)] for i in range(n_steps + 1)]
    barrier_breached = [[False for _ in range(i + 1)] for i in range(n_steps + 1)]

    # Build stock price tree and track whether barrier has been hit along each path
    for i in range(n_steps + 1):      # Time step (0 to n_steps)
        for j in range(i + 1):        # Number of up-moves (0 to i)
            # Calculate stock price: start * (up_factor^up_moves) * (down_factor^down_moves)
            S_ij = current_price * (u ** j) * (d ** (i - j))
            S[i][j] = S_ij

            if i == 0:  # At start (root node)
                barrier_breached[i][j] = S_ij >= barrier_price  # Check if we start above barrier
            else:
                # Check if barrier breached: either at this node OR inherited from parent nodes
                from_left = barrier_breached[i - 1][j] if j < i else False      # Came from down-move
                from_right = barrier_breached[i - 1][j - 1] if j > 0 else False # Came from up-move
                barrier_breached[i][j] = (S_ij >= barrier_price) or from_left or from_right

    # Calculate terminal payoffs: option only has value if barrier was breached during path
    V = [[0.0 for _ in range(i + 1)] for i in range(n_steps + 1)]
    for j in range(n_steps + 1):
        if barrier_breached[n_steps][j]:  # Only if barrier was hit during this path
            ST = S[n_steps][j]
            if option_type == 'call':
                V[n_steps][j] = max(ST - strike_price, 0)  # Call payoff
            elif option_type == 'put':
                V[n_steps][j] = max(strike_price - ST, 0)  # Put payoff

    # Backward induction: work backwards calculating continuation values
    for i in range(n_steps - 1, -1, -1):  # Go backwards through time
        for j in range(i + 1):
            # Calculate expected value: probability-weighted average of future outcomes, discounted
            continuation_value = discount * (p * V[i + 1][j + 1] + (1 - p) * V[i + 1][j])
            
            # For up-and-in barriers, option value equals continuation value
            # (No early exercise for this type of barrier option)
            V[i][j] = continuation_value

    return V[0][0]  # Return today's option value

def monte_carlo_price(option_type, current_price, strike_price, time_to_maturity, interest_rate, sigma, n_paths=10000):
    """
    Price European options using Monte Carlo simulation
    
    Monte Carlo method: simulate many possible future stock price paths, calculate average payoff.
    Based on geometric Brownian motion: stock prices follow random walk with drift.
    """
    # Generate random numbers from standard normal distribution (mean=0, std=1)
    # Each number represents one random price path
    Z = np.random.normal(0, 1, size=n_paths)
    
    # Apply geometric Brownian motion formula to get final stock prices
    # S_T = S_0 * exp((r - 0.5*σ²)*T + σ*√T*Z)
    # The (r - 0.5*σ²) term is the risk-neutral drift, σ*√T*Z is the random component
    final_stock_prices = current_price * np.exp(
        (interest_rate - 0.5 * sigma ** 2) * time_to_maturity + sigma * np.sqrt(time_to_maturity) * Z
    )

    # Calculate option payoff for each simulated path
    if option_type == 'call':
        payoffs = np.maximum(final_stock_prices - strike_price, 0)  # Call: max(S-K, 0)
    elif option_type == 'put':
        payoffs = np.maximum(strike_price - final_stock_prices, 0)  # Put: max(K-S, 0)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    # Option price = average payoff discounted to present value
    return np.exp(-interest_rate * time_to_maturity) * np.mean(payoffs)

def monte_carlo_barrier_price(current_price, strike_price, time_to_maturity, interest_rate, sigma, barrier_price, n_paths=10000, n_steps=100, dividend_yield=0.0):
    """
    Price up-and-in barrier options using Monte Carlo path simulation
    
    Unlike regular Monte Carlo that only simulates final price, this tracks full price path
    to check if barrier was hit at any point during the option's life.
    """
    dt = time_to_maturity / n_steps  # Time step size for path simulation
    discount_factor = np.exp(-interest_rate * time_to_maturity)
    
    # Initialize price matrix: rows = time steps, columns = simulation paths
    # S[i][j] = stock price at time step i for path j
    S = np.zeros((n_steps + 1, n_paths))
    S[0] = current_price  # All paths start at current stock price
    
    # Generate matrix of random numbers: one for each time step and path
    Z = np.random.normal(0, 1, size=(n_steps, n_paths))
    
    # Simulate stock price paths using geometric Brownian motion with small time steps
    drift = (interest_rate - dividend_yield - 0.5 * sigma**2) * dt      # Deterministic part
    diffusion_stoch = sigma * np.sqrt(dt) * Z                           # Random part

    # Build each time step of all paths simultaneously (vectorized computation)
    for i in range(1, n_steps + 1):
        S[i] = S[i-1] * np.exp(drift + diffusion_stoch[i-1])  # Each step: S_new = S_old * exp(drift + random)
    
    # Check barrier breach: for each path, did stock price ever exceed barrier?
    # np.max finds maximum price along each path (column), >= checks if it hit barrier
    barrier_breached = np.max(S, axis=0) >= barrier_price
    
    # Calculate payoffs: only paths that hit barrier get regular option payoff
    final_prices = S[-1]  # Final stock prices for all paths
    # np.where: if barrier_breached, calculate payoff; otherwise payoff = 0
    payoffs = np.where(barrier_breached, np.maximum(final_prices - strike_price, 0), 0)
    
    return discount_factor * np.mean(payoffs)  # Average payoff, discounted to present

def monte_carlo_basket_price(current_prices, weights, strike_price, time_to_maturity, interest_rate, sigma, correlation_matrix, option_type='call', n_paths=10000, dividend_yields=None):
    """
    Price basket options using Monte Carlo simulation with correlated assets
    
    Basket option: payoff depends on weighted combination of multiple stocks.
    Challenge: stocks are correlated (move together), need to model this relationship.
    """
    n_assets = len(current_prices)
    
    # Convert inputs to numpy arrays for efficient mathematical operations
    current_prices = np.array(current_prices)
    weights = np.array(weights)
    sigmas_arr = np.array(sigma)  # Individual volatilities for each asset

    # Handle dividend yields: if not provided, assume no dividends
    if dividend_yields is None:
        dividend_yields_arr = np.zeros(n_assets)
    else:
        dividend_yields_arr = np.array(dividend_yields)
        if len(dividend_yields_arr) != n_assets:
            raise ValueError("Length of dividend_yields must match number of assets.")

    # Generate correlated random numbers using Cholesky decomposition
    # Cholesky breaks down correlation matrix: if A = L*L^T, then L*Z gives correlated randoms
    L = np.linalg.cholesky(correlation_matrix)
    Z_uncorrelated = np.random.normal(0, 1, size=(n_assets, n_paths))  # Independent random numbers
    Z_correlated = L @ Z_uncorrelated  # Matrix multiplication gives correlated random numbers
    
    final_asset_prices = np.zeros((n_assets, n_paths))
    
    # Simulate final price for each asset using geometric Brownian motion
    for i in range(n_assets):
        # Each asset follows its own GBM with its own volatility and dividend yield
        drift = (interest_rate - dividend_yields_arr[i] - 0.5 * sigmas_arr[i]**2) * time_to_maturity
        diffusion = sigmas_arr[i] * np.sqrt(time_to_maturity) * Z_correlated[i, :]
        final_asset_prices[i, :] = current_prices[i] * np.exp(drift + diffusion)
        
    # Calculate basket value at maturity: weighted sum of all asset prices
    # weights[:, np.newaxis] reshapes weights for broadcasting across all paths
    basket_values_at_maturity = np.sum(final_asset_prices * weights[:, np.newaxis], axis=0)
    
    # Calculate basket option payoff for each simulated path
    if option_type.lower() == 'call':
        payoffs = np.maximum(basket_values_at_maturity - strike_price, 0)  # Call on basket
    elif option_type.lower() == 'put':
        payoffs = np.maximum(strike_price - basket_values_at_maturity, 0)  # Put on basket
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
        
    return np.exp(-interest_rate * time_to_maturity) * np.mean(payoffs)  # Discounted average payoff

def get_zero_rate(time_to_maturity):
    """
    Bootstrap yield curve to extract pure interest rates for any maturity
    
    Problem: Market gives us different types of rates (discount rates, coupon bonds).
    Solution: Convert everything to zero-coupon rates (pure interest rates with no coupons).
    """
    # Market data: different maturities and their corresponding rates
    maturities = [1/12, 3/12, 6/12, 1, 2, 3, 5]  # 1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y
    bb_yields = [0.037925, 0.03797, 0.03855]      # Bank bill DISCOUNT yields (short-term)
    gov_yields = [0.0348, 0.03374, 0.03431, 0.03659]  # Government bond PAR yields (long-term)
    zero_rates = []  # Will store the pure zero rates we calculate
    
    # Step 1: Convert bank bill discount yields to zero rates
    # Bank bills are sold at discount, mature at face value (no coupons)
    for i in range(3):  # Process 1M, 3M, 6M bank bills
        T = maturities[i]
        price = 1 - bb_yields[i] * T  # Discount formula: Price = Face_Value - Discount
        zero_rate = (1/price)**(1/T) - 1  # Compound interest formula: (FV/PV)^(1/T) - 1
        zero_rates.append(zero_rate)
    
    # Step 2: Bootstrap government bonds (they pay semi-annual coupons)
    # Bootstrapping: use known short rates to find unknown long rates
    for i in range(4):  # Process 1Y, 2Y, 3Y, 5Y government bonds
        maturity = maturities[i + 3]
        par_yield = gov_yields[i]  # Par yield: coupon rate when bond trades at face value
        
        if maturity == 1:  # Special case: 1-year bond (simpler calculation)
            coupon = par_yield * 100 / 2  # Semi-annual coupon payment
            from scipy.optimize import fsolve
            
            # Bond pricing equation: PV = Coupon/(1+r/2) + (Coupon+Principal)/(1+r/2)²
            def bond_price_error(z):  # Function to find zero rate that makes bond price = 100
                discount_factor = 1 + z/2  # Semi-annual compounding
                pv = coupon / discount_factor + (coupon + 100) / (discount_factor**2)
                return pv - 100  # Error: difference from par value (100)
            
            z_1y = fsolve(bond_price_error, par_yield)[0]  # Numerical solver finds zero rate
            zero_rates.append(z_1y)
        else:  # Multi-year bonds: use bootstrapping technique
            coupon = par_yield * 100 / 2  # Semi-annual coupon
            periods = int(maturity * 2)   # Number of coupon payments
            
            # Calculate present value of all intermediate coupons using known rates
            pv_coupons = 0
            for p in range(1, periods):  # All payments except the last
                t = p / 2  # Time of this coupon payment
                z = _get_rate(t, maturities[:len(zero_rates)], zero_rates)  # Interpolate known rate
                pv_coupons += coupon / (1 + z/2)**p  # Discount this coupon payment
            
            # Solve for final zero rate: remaining PV must equal final payment discounted
            final_payment = coupon + 100  # Last coupon + principal repayment
            remaining_pv = 100 - pv_coupons  # What final payment must be worth today
            discount_factor = final_payment / remaining_pv
            z = 2 * (discount_factor**(1/periods) - 1)  # Convert to annual rate
            zero_rates.append(z)
    
    return _get_rate(time_to_maturity, maturities, zero_rates)  # Interpolate for requested maturity

def _get_rate(target_time, times, rates):
    """
    Linear interpolation: estimate rate for any time between known data points
    
    If we know rates at time 1 and time 2, we can estimate rate at time 1.5
    by taking weighted average based on how close 1.5 is to each endpoint.
    """
    if target_time <= times[0]:
        return rates[0]  # Before first data point: use first rate
    if target_time >= times[-1]:
        return rates[-1]  # After last data point: use last rate
    
    # Find which two data points bracket our target time
    for i in range(len(times) - 1):
        if times[i] <= target_time <= times[i + 1]:
            # Linear interpolation formula: rate = rate1 + weight * (rate2 - rate1)
            weight = (target_time - times[i]) / (times[i + 1] - times[i])  # How close to second point (0 to 1)
            return rates[i] + weight * (rates[i + 1] - rates[i])
    
    return rates[-1]  # Fallback (shouldn't reach here)

def black_scholes_greeks(option_type, current_price, strike_price, time_to_maturity, interest_rate, sigma):
    """
    Calculate Black-Scholes Greeks: sensitivities of option price to various factors
    
    Greeks measure risk: how much option price changes when market conditions change.
    Essential for risk management and hedging strategies.
    """
    # Calculate the famous d1 and d2 parameters from Black-Scholes formula
    d1 = (np.log(current_price / strike_price) + (interest_rate + 0.5 * sigma ** 2) * time_to_maturity) / (sigma * np.sqrt(time_to_maturity))
    d2 = d1 - sigma * np.sqrt(time_to_maturity)
    
    # Standard normal distribution functions: cdf = cumulative, pdf = probability density
    nd1 = norm.cdf(d1)   # N(d1): probability that normal random variable ≤ d1
    nd2 = norm.cdf(d2)   # N(d2): probability that normal random variable ≤ d2
    npd1 = norm.pdf(d1)  # n(d1): probability density at d1 (height of bell curve)
    
    if option_type == 'call':
        delta = nd1  # Delta: how much option price changes per $1 change in stock price
        # Theta: how much option price changes per day (time decay)
        theta = (-(current_price * npd1 * sigma) / (2 * np.sqrt(time_to_maturity)) 
                - interest_rate * strike_price * np.exp(-interest_rate * time_to_maturity) * nd2)
        # Rho: how much option price changes per 1% change in interest rate
        rho = strike_price * time_to_maturity * np.exp(-interest_rate * time_to_maturity) * nd2
    elif option_type == 'put':
        delta = nd1 - 1  # Put delta is negative (puts gain value when stock falls)
        theta = (-(current_price * npd1 * sigma) / (2 * np.sqrt(time_to_maturity)) 
                + interest_rate * strike_price * np.exp(-interest_rate * time_to_maturity) * norm.cdf(-d2))
        rho = -strike_price * time_to_maturity * np.exp(-interest_rate * time_to_maturity) * norm.cdf(-d2)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    
    # These Greeks are the same for calls and puts
    gamma = npd1 / (current_price * sigma * np.sqrt(time_to_maturity))  # Gamma: convexity (how delta changes)
    vega = current_price * npd1 * np.sqrt(time_to_maturity)             # Vega: volatility sensitivity
    
    return {
        'delta': delta,  # Price sensitivity: ∂V/∂S
        'gamma': gamma,  # Delta sensitivity: ∂²V/∂S²
        'theta': theta,  # Time decay: ∂V/∂t
        'vega': vega,    # Volatility sensitivity: ∂V/∂σ
        'rho': rho       # Interest rate sensitivity: ∂V/∂r
    }