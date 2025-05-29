import numpy as np
from datetime import date
from functions import (
    black_scholes_price,
    binomial_tree_price,
    monte_carlo_price,
    monte_carlo_barrier_price,
    monte_carlo_basket_price,
    binomial_barrier_price_up_and_in,
    get_zero_rate
)

class Option:
    """
    Base option class: foundation for all option types
    
    This is the parent class that defines common attributes and methods.
    All specific option types (European, American, etc.) inherit from this class.
    """
    def __init__(self, current_price, strike_price, expiry, option_type, today_date):
        # Instance variables: each option object will have its own copy of these values
        self.current_price = current_price    # Current market price of the underlying asset
        self.strike_price = strike_price      # Price at which option can be exercised
        self.expiry = expiry                  # Expiration date (datetime.date object)
        self.option_type = option_type        # 'call' or 'put'
        self.today_date = today_date          # Today's date for time calculations

    def time_to_maturity(self):
        """
        Calculate time remaining until expiration in years
        
        Uses .days to get integer days between dates, then converts to years.
        365.25 accounts for leap years (more accurate than 365).
        """
        return (self.expiry - self.today_date).days / 365.25


class EuropeanOption(Option):
    """
    European option: can only be exercised at expiration date
    
    Uses Black-Scholes analytical formula for pricing.
    Inheritance: EuropeanOption inherits all methods from Option class.
    """
    def __init__(self, current_price, strike_price, expiry, option_type, today_date, sigma, interest_rate=None, dividend_yield=0.0):
        # Call parent class constructor using super() - this sets up basic option attributes
        super().__init__(current_price, strike_price, expiry, option_type, today_date)
        
        # Additional attributes specific to European options
        self.sigma = sigma  # Volatility: how much the stock price fluctuates (e.g., 0.20 = 20% annual volatility)
        
        # If no interest rate provided, bootstrap it from market data using our custom function
        # This demonstrates conditional assignment and function call
        self.interest_rate = interest_rate if interest_rate is not None else get_zero_rate(self.time_to_maturity())
        
        self.dividend_yield = dividend_yield  # Annual dividend rate (default 0 = no dividends)

    def option_price(self):
        """
        Calculate option price using Black-Scholes formula
        
        This method calls the black_scholes_price function from our functions module.
        Notice how we pass self.attribute to access instance variables.
        """
        return black_scholes_price(
            option_type=self.option_type,
            current_price=self.current_price,
            strike_price=self.strike_price,
            time_to_maturity=self.time_to_maturity(),  # Method call 
            interest_rate=self.interest_rate,
            sigma=self.sigma,
            dividend_yield=self.dividend_yield
        )
    
    def greeks(self):
        """
        Calculate option Greeks using finite differences method
        
        Greeks = sensitivities of option price to market factors.
        Finite differences = approximate derivative by bumping inputs slightly.
        Note: Analytical Greeks exist for Black-Scholes but this shows the general method.
        """
        # Define bump sizes: small changes to inputs to approximate derivatives
        h_price = self.current_price * 0.005  # 0.5% bump for stock price (proportional)
        h_vol = 0.005      # 0.5% absolute bump for volatility
        h_rate = 0.005     # 0.5% absolute bump for interest rate
        time_bump_days = 1 # Move forward 1 day for theta calculation
        time_bump_years = time_bump_days / 365.25

        base_price = self.option_price()  # Calculate current option price

        # Store original values: we'll modify self.attributes temporarily, then reset them
        # This is important because we don't want to permanently change the option's properties
        original_current_price = self.current_price
        original_today_date = self.today_date
        original_sigma = self.sigma
        original_interest_rate = self.interest_rate

        # DELTA: sensitivity to stock price changes (∂V/∂S)
        # We calculate option price with stock price bumped up and down, then find slope
        self.current_price = original_current_price + h_price  # Bump up
        price_up = self.option_price()
        self.current_price = original_current_price - h_price  # Bump down
        price_down = self.option_price()
        self.current_price = original_current_price  # Reset to original
        
        # Central difference formula: (f(x+h) - f(x-h)) / (2h) ≈ f'(x)
        delta = (price_up - price_down) / (2 * h_price)
        
        # GAMMA: convexity - how delta changes as stock price changes (∂²V/∂S²)
        # Second derivative formula: (f(x+h) - 2f(x) + f(x-h)) / h²
        gamma = (price_up - 2 * base_price + price_down) / (h_price ** 2)
        
        # THETA: time decay - how option loses value as time passes (∂V/∂t)
        # Move date forward by 1 day (reducing time to maturity)
        self.today_date = date.fromordinal(original_today_date.toordinal() + time_bump_days) 
        price_theta = self.option_price()
        self.today_date = original_today_date  # Reset
        theta = (price_theta - base_price) / time_bump_years  # Change per year
        
        # VEGA: volatility sensitivity - how price changes with volatility (∂V/∂σ)
        self.sigma = original_sigma + h_vol  # Bump volatility up
        price_vega = self.option_price()
        self.sigma = original_sigma  # Reset
        # Scale to 1% volatility change: if h_vol=0.005, we want change for 0.01 move
        vega = ((price_vega - base_price) / h_vol) * 0.01
        
        # RHO: interest rate sensitivity - how price changes with rates (∂V/∂r)
        self.interest_rate = original_interest_rate + h_rate  # Bump rate up
        price_rho = self.option_price()
        self.interest_rate = original_interest_rate  # Reset
        # Scale to 1% rate change
        rho = ((price_rho - base_price) / h_rate) * 0.01
        
        # Return dictionary: convenient way to return multiple related values
        return {
            'delta': delta,                    # Stock price sensitivity
            'gamma': gamma,                    # Convexity (curvature)
            'theta': theta / 365.25,           # Time decay per day (convert from per year)
            'vega': vega,                      # Volatility sensitivity
            'rho': rho                         # Interest rate sensitivity
        }


class AmericanOption(Option):
    """
    American option: can be exercised at any time before expiration
    
    Uses binomial tree method because no analytical solution exists.
    Early exercise feature makes these more valuable than European options.
    """
    def __init__(self, current_price, strike_price, expiry, option_type, today_date, sigma, n_steps, interest_rate=None, dividend_yield=0.0):
        super().__init__(current_price, strike_price, expiry, option_type, today_date)
        
        self.sigma = sigma
        self.n_steps = n_steps  # Number of time steps in binomial tree (more steps = more accuracy)
        self.interest_rate = interest_rate if interest_rate is not None else get_zero_rate(self.time_to_maturity())
        self.dividend_yield = dividend_yield

    def option_price(self):
        """
        Price using binomial tree with early exercise capability
        
        The american=True parameter tells the binomial tree to check for early exercise
        at each node in the tree.
        """
        return binomial_tree_price(
            current_price=self.current_price,
            strike_price=self.strike_price,
            time_to_maturity=self.time_to_maturity(),
            interest_rate=self.interest_rate,
            sigma=self.sigma,
            option_type=self.option_type,
            american=True,  # Key difference: enables early exercise
            n_steps=self.n_steps,
            dividend_yield=self.dividend_yield
        )
    
    def greeks(self):
        """
        Calculate Greeks using finite differences for American options
        
        Same method as European, but calculations are more expensive because
        each price calculation requires building and solving a binomial tree.
        """
        # Store original values for safe reset after calculations
        original_current_price = self.current_price
        original_sigma = self.sigma
        original_interest_rate = self.interest_rate
        original_today_date = self.today_date

        # Bump sizes: slightly larger for American options due to tree discretization
        h_price = original_current_price * 0.01  # 1% proportional bump for stock price
        if h_price == 0: h_price = 0.0001  # Safety check: avoid zero bump if stock price somehow zero

        base_price = self.option_price()  # This calls binomial tree calculation
        
        # DELTA calculation: same finite difference approach as European
        self.current_price = original_current_price + h_price
        up_price = self.option_price()  # Another tree calculation
        self.current_price = original_current_price - h_price
        down_price = self.option_price()  # Another tree calculation
        self.current_price = original_current_price  # Reset
        delta = (up_price - down_price) / (2 * h_price)
        
        # GAMMA: second derivative with respect to stock price
        gamma = (up_price - 2 * base_price + down_price) / (h_price ** 2)
        
        # THETA: time decay calculation
        time_bump_days = 1
        time_bump_years = time_bump_days / 365.25
        
        self.today_date = date.fromordinal(original_today_date.toordinal() + time_bump_days)
        theta_price = self.option_price()  # Tree calculation with less time
        self.today_date = original_today_date  # Reset
        theta = (theta_price - base_price) / time_bump_years
        
        # VEGA: volatility sensitivity
        h_vol_american = 0.01  # Slightly larger bump for tree-based calculation
        self.sigma = original_sigma + h_vol_american
        vega_price = self.option_price()  # Tree with higher volatility
        self.sigma = original_sigma  # Reset
        vega = ((vega_price - base_price) / h_vol_american) * 0.01  # Scale to 1% vol change
        
        # RHO: interest rate sensitivity
        h_rate_american = 0.01
        self.interest_rate = original_interest_rate + h_rate_american
        rho_price = self.option_price()  # Tree with higher interest rate
        self.interest_rate = original_interest_rate  # Reset
        rho = ((rho_price - base_price) / h_rate_american) * 0.01  # Scale to 1% rate change
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta / 365.25,  # Convert to per-day theta
            'vega': vega,
            'rho': rho
        }


class BarrierOption(Option):
    """
    Barrier option: only becomes active if stock hits a barrier level
    
    Up-and-in: barrier above current price, option activates if stock moves up to barrier.
    More complex than regular options because payoff depends on entire price path.
    """
    def __init__(self, current_price, strike_price, expiry, option_type, today_date, sigma, barrier_price, method, n_steps, interest_rate=None, dividend_yield=0.0):
        super().__init__(current_price, strike_price, expiry, option_type, today_date)
        
        self.sigma = sigma
        self.barrier_price = barrier_price  # Level that must be hit to activate option
        self.method = method                # "binomial" or "monte-carlo"
        self.n_steps = n_steps             # For binomial: tree steps; for Monte Carlo: number of paths
        self.interest_rate = interest_rate if interest_rate is not None else get_zero_rate(self.time_to_maturity())
        self.dividend_yield = dividend_yield

    def option_price(self):
        """
        Calculate barrier option price using specified method
        
        Demonstrates polymorphism: same method name, different behavior based on input.
        """
        if self.method == "monte-carlo":
            # Monte Carlo: simulate many price paths, check if barrier hit during each path
            n_mc_sim_steps = 100  # How many time steps to simulate in each path
            return monte_carlo_barrier_price(
                current_price=self.current_price,
                strike_price=self.strike_price,
                time_to_maturity=self.time_to_maturity(),
                interest_rate=self.interest_rate,
                sigma=self.sigma,
                barrier_price=self.barrier_price,
                n_paths=self.n_steps,      # self.n_steps = number of simulation paths
                n_steps=n_mc_sim_steps,    # Time steps within each path
                dividend_yield=self.dividend_yield
            )
        elif self.method == "binomial":
            # Binomial tree: track barrier breach status at each node
            return binomial_barrier_price_up_and_in(
                option_type=self.option_type,
                current_price=self.current_price,
                strike_price=self.strike_price,
                time_to_maturity=self.time_to_maturity(),
                interest_rate=self.interest_rate,
                sigma=self.sigma,
                barrier_price=self.barrier_price,
                n_steps=self.n_steps,
                dividend_yield=self.dividend_yield
            )
        else:
            # Error handling: raise exception if invalid method specified
            raise ValueError("Unsupported method. Use 'monte-carlo' or 'binomial'.")

    def greeks(self):
        """
        Calculate Greeks for barrier options using finite differences
        
        Barrier options can have discontinuous behavior near the barrier,
        so we use smaller bump sizes to get more accurate derivatives.
        """
        # Store original values
        original_current_price = self.current_price
        original_sigma = self.sigma
        original_interest_rate = self.interest_rate
        original_today_date = self.today_date

        # Smaller bump sizes due to potential discontinuities near barrier
        h_price = original_current_price * 0.001  # 0.1% bump (smaller than regular options)
        if h_price == 0: h_price = 0.0001  # Safety check
        h_vol = 0.005   # 0.5% volatility bump
        h_rate = 0.005  # 0.5% rate bump
        time_bump_days = 1
        time_bump_years = time_bump_days / 365.25

        base_price = self.option_price()  # Can be expensive if using Monte Carlo
        
        # DELTA calculation with smaller bumps
        self.current_price = original_current_price + h_price
        up_price = self.option_price()
        self.current_price = original_current_price - h_price
        down_price = self.option_price()
        self.current_price = original_current_price  # Reset
        delta = (up_price - down_price) / (2 * h_price)
        
        # GAMMA calculation
        gamma = (up_price - 2 * base_price + down_price) / (h_price ** 2)
        
        # VEGA calculation
        self.sigma = original_sigma + h_vol
        vega_price = self.option_price()
        self.sigma = original_sigma  # Reset
        vega = ((vega_price - base_price) / h_vol) * 0.01
        
        # THETA calculation
        self.today_date = date.fromordinal(original_today_date.toordinal() + time_bump_days)
        theta_price = self.option_price()
        self.today_date = original_today_date  # Reset
        theta = (theta_price - base_price) / time_bump_years
        
        # RHO calculation
        self.interest_rate = original_interest_rate + h_rate
        rho_price = self.option_price()
        self.interest_rate = original_interest_rate  # Reset
        rho = ((rho_price - base_price) / h_rate) * 0.01
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta / 365.25,  # Per-day theta
            'vega': vega,
            'rho': rho
        }


class BasketOption(Option):
    """
    Basket option: payoff depends on weighted combination of multiple assets
    
    Example: option on portfolio of 60% AAPL + 40% GOOGL.
    Challenge: assets are correlated, need to model how they move together.
    """
    def __init__(self, current_prices, weights, strike_price, expiry, option_type, today_date, sigmas, correlation_matrix, interest_rate=None, dividend_yields=None):
        # Calculate weighted basket value for the parent Option class
        # sum() with zip() is Python idiom for weighted sum: Σ(weight_i * price_i)
        basket_value = sum(w * p for w, p in zip(weights, current_prices))
        super().__init__(basket_value, strike_price, expiry, option_type, today_date)
        
        # Convert to numpy arrays for efficient mathematical operations
        # Lists are fine for small data, but numpy arrays are much faster for math
        self.current_prices = np.array(current_prices)  # Individual asset prices
        self.weights = np.array(weights)                # Portfolio weights (must sum to 1)
        self.sigmas = np.array(sigmas)                  # Individual asset volatilities
        self.correlation_matrix = np.array(correlation_matrix)  # How assets move together
        self.interest_rate = interest_rate if interest_rate is not None else get_zero_rate(self.time_to_maturity())
        
        # Handle dividend yields: each asset can have different dividend rate
        if dividend_yields is None:
            self.dividend_yields = np.zeros(len(current_prices))  # Default: no dividends
        else:
            self.dividend_yields = np.array(dividend_yields)
        
        # Error checking: common programming practice to validate inputs
        if len(self.dividend_yields) != len(current_prices):
            raise ValueError("Length of dividend_yields must match length of current_prices")

    def option_price(self):
        """
        Price basket option using Monte Carlo simulation
        
        Only realistic method for multi-asset options with correlation.
        Monte Carlo handles correlation through Cholesky decomposition.
        """
        return monte_carlo_basket_price(
            current_prices=self.current_prices,
            weights=self.weights,
            strike_price=self.strike_price,
            time_to_maturity=self.time_to_maturity(),
            interest_rate=self.interest_rate,
            sigma=self.sigmas,  # Pass array of individual volatilities
            correlation_matrix=self.correlation_matrix,
            option_type=self.option_type,
            n_paths=100000,  # High number for accuracy (but slower computation)
            dividend_yields=self.dividend_yields
        )
    
    def greeks(self):
        """
        Calculate basket option Greeks
        
        More complex than single-asset options: we calculate delta for EACH underlying asset.
        Shows how basket option price changes when individual asset prices change.
        """
        # Store original values: we'll modify multiple arrays
        original_current_prices = self.current_prices.copy()  # .copy() creates independent copy
        original_sigmas = self.sigmas.copy()
        original_interest_rate = self.interest_rate
        original_today_date = self.today_date

        base_option_price = self.option_price()  # Expensive Monte Carlo calculation
        deltas = []  # List to store delta for each underlying asset
        
        # Calculate individual asset deltas using loop
        # Range(len(list)) is common Python pattern to iterate over indices
        for i in range(len(self.current_prices)):
            h_asset_i = original_current_prices[i] * 0.001  # 0.1% bump for asset i
            if h_asset_i == 0: h_asset_i = 0.00001  # Avoid zero bump

            # Bump UP asset i's price, recalculate basket value and option price
            temp_prices_up = original_current_prices.copy()  # Copy to avoid modifying original
            temp_prices_up[i] += h_asset_i  # Modify only asset i
            self.current_prices = temp_prices_up
            # Recalculate overall basket value (needed for option pricing)
            self.current_price = sum(w * p for w, p in zip(self.weights, self.current_prices))
            price_up = self.option_price()  # Monte Carlo with asset i bumped up
            
            # Bump DOWN asset i's price
            temp_prices_down = original_current_prices.copy()
            temp_prices_down[i] -= h_asset_i  # Modify only asset i downward
            self.current_prices = temp_prices_down
            self.current_price = sum(w * p for w, p in zip(self.weights, self.current_prices))
            price_down = self.option_price()  # Monte Carlo with asset i bumped down
            
            # Calculate delta for asset i using central difference
            delta_i = (price_up - price_down) / (2 * h_asset_i)
            deltas.append(delta_i)  # Add to our list of deltas
        
        # Reset to original values after delta calculations
        self.current_prices = original_current_prices
        self.current_price = sum(w * p for w, p in zip(self.weights, self.current_prices))

        # BASKET GAMMA: sensitivity of option to uniform change in all asset prices
        # This simulates what happens if entire market moves up/down together
        h_basket_simulation = 0.001  # 0.1% uniform bump to all assets
        
        # Multiply all prices by (1 + bump): uniform percentage increase
        temp_prices_gamma_up = original_current_prices * (1 + h_basket_simulation)
        self.current_prices = temp_prices_gamma_up
        self.current_price = sum(w * p for w, p in zip(self.weights, self.current_prices))
        price_gamma_up = self.option_price()

        # Uniform decrease
        temp_prices_gamma_down = original_current_prices * (1 - h_basket_simulation)
        self.current_prices = temp_prices_gamma_down
        self.current_price = sum(w * p for w, p in zip(self.weights, self.current_prices))
        price_gamma_down = self.option_price()

        # Calculate gamma based on change in total basket value
        original_basket_value = sum(w * p for w, p in zip(self.weights, original_current_prices))
        h_basket_value_change = original_basket_value * h_basket_simulation

        if h_basket_value_change == 0:  # Avoid division by zero (defensive programming)
            basket_gamma = 0.0
        else:
            # Second derivative formula for gamma
            basket_gamma = (price_gamma_up - 2 * base_option_price + price_gamma_down) / (h_basket_value_change ** 2)

        # Reset everything back to original state
        self.current_prices = original_current_prices
        self.current_price = original_basket_value

        # VEGA: sensitivity to parallel shift in ALL volatilities
        # What if all assets become more/less volatile together?
        h_vol = 0.005  # Add 0.5% to all asset volatilities
        temp_sigmas_up = original_sigmas + h_vol  # Numpy array addition: adds to each element
        self.sigmas = temp_sigmas_up
        price_vega_up = self.option_price()
        self.sigmas = original_sigmas  # Reset
        basket_vega = ((price_vega_up - base_option_price) / h_vol) * 0.01  # Scale to 1% vol change
        
        # THETA: time decay for basket option
        time_bump_days = 1
        time_bump_years = time_bump_days / 365.25
        self.today_date = date.fromordinal(original_today_date.toordinal() + time_bump_days)
        price_theta = self.option_price()
        self.today_date = original_today_date  # Reset
        basket_theta = (price_theta - base_option_price) / time_bump_years
        
        # RHO: interest rate sensitivity for basket option
        h_rate = 0.005  # 0.5% rate bump
        self.interest_rate = original_interest_rate + h_rate
        price_rho = self.option_price()
        self.interest_rate = original_interest_rate  # Reset
        basket_rho = ((price_rho - base_option_price) / h_rate) * 0.01  # Scale to 1% rate change

        # Return comprehensive Greeks dictionary
        return {
            'deltas': deltas,           # List of deltas, one for each underlying asset
            'gamma': basket_gamma,      # Overall basket convexity
            'vega': basket_vega,        # Sensitivity to volatility changes
            'theta': basket_theta / 365.25,  # Time decay per day
            'rho': basket_rho           # Interest rate sensitivity
        }
    
    def _calculate_portfolio_volatility(self):
        """
        Calculate portfolio volatility using modern portfolio theory
        
        Portfolio volatility ≠ weighted average of individual volatilities.
        Must account for correlations: uncorrelated assets reduce total risk.
        Formula: σ_portfolio² = Σᵢ Σⱼ wᵢ wⱼ σᵢ σⱼ ρᵢⱼ
        """
        n = len(self.weights)
        portfolio_variance = 0
        
        # Double loop: consider all pairs of assets (including asset with itself)
        for i in range(n):
            for j in range(n):
                # Add contribution of asset pair (i,j) to total portfolio variance
                portfolio_variance += (self.weights[i] * self.weights[j] *    # Weight contribution
                                     self.sigmas[i] * self.sigmas[j] *        # Volatility contribution  
                                     self.correlation_matrix[i,j])            # Correlation contribution
        
        # Defensive programming: variance should never be negative with valid inputs
        if portfolio_variance < 0:
            return 0.0
        
        # Convert variance to volatility (standard deviation)
        return np.sqrt(portfolio_variance)

