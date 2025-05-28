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
    def __init__(self, current_price, strike_price, expiry, option_type, today_date):
        self.current_price = current_price
        self.strike_price = strike_price
        self.expiry = expiry
        self.option_type = option_type
        self.today_date = today_date

    def time_to_maturity(self):
        return (self.expiry - self.today_date).days / 365.25


class EuropeanOption(Option):
    def __init__(self, current_price, strike_price, expiry, option_type, today_date, sigma, interest_rate=None, dividend_yield=0.0):
        super().__init__(current_price, strike_price, expiry, option_type, today_date)
        self.sigma = sigma
        self.interest_rate = interest_rate if interest_rate is not None else get_zero_rate(self.time_to_maturity())
        self.dividend_yield = dividend_yield

    def option_price(self):
        return black_scholes_price(
            option_type=self.option_type,
            current_price=self.current_price,
            strike_price=self.strike_price,
            time_to_maturity=self.time_to_maturity(),
            interest_rate=self.interest_rate,
            sigma=self.sigma,
            dividend_yield=self.dividend_yield
        )
    
    def greeks(self):
        """Approximate Greeks using finite differences for European options (could be analytical)"""
        # Note: For European options, analytical Greeks are available and generally preferred.
        # This finite difference method is kept for consistency or if analytical Greeks are not implemented.
        
        h_price = self.current_price * 0.005 # Smaller bump for price based Greeks for potentially more stability
                                          # For European BS, even 0.01 is fine, but being slightly more precise.
        h_vol = 0.005 # 0.5% bump for vega
        h_rate = 0.005 # 0.5% bump for rho
        time_bump_days = 1
        time_bump_years = time_bump_days / 365.25

        base_price = self.option_price()

        # Store original values to ensure they are reset, especially important if greeks are called multiple times
        original_current_price = self.current_price
        original_today_date = self.today_date
        original_sigma = self.sigma
        original_interest_rate = self.interest_rate

        # Delta: sensitivity to underlying price
        self.current_price = original_current_price + h_price
        price_up = self.option_price()
        self.current_price = original_current_price - h_price
        price_down = self.option_price()
        self.current_price = original_current_price # Reset
        delta = (price_up - price_down) / (2 * h_price)
        
        # Gamma: second derivative with respect to underlying price
        # price_up, price_down, and base_price are already calculated correctly for this formula
        gamma = (price_up - 2 * base_price + price_down) / (h_price ** 2)
        
        # Theta: sensitivity to time (approximate)
        # We move today_date forward by one day, so time to maturity decreases
        self.today_date = date.fromordinal(original_today_date.toordinal() + time_bump_days) 
        price_theta = self.option_price()
        self.today_date = original_today_date # Reset
        # Theta is typically quoted as dV/dt where dt is positive (time passes). 
        # So (new_price - old_price) / (time_passed). Since new_price (less TTM) < old_price, theta is negative for long options.
        theta = (price_theta - base_price) / time_bump_years 
        
        # Vega: sensitivity to volatility
        self.sigma = original_sigma + h_vol
        price_vega = self.option_price()
        self.sigma = original_sigma # Reset
        vega = ((price_vega - base_price) / h_vol) * 0.01
        
        # Rho: sensitivity to interest rate
        self.interest_rate = original_interest_rate + h_rate
        price_rho = self.option_price()
        self.interest_rate = original_interest_rate # Reset
        rho = ((price_rho - base_price) / h_rate) * 0.01
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta, # Theta is per year. Divide by 365 for per day.
            'vega': vega,   # Vega is change for 1 vol point (e.g. from 20% to 21%). If h_vol is 0.005 (0.5%), then vega is scaled up.
            'rho': rho     # Rho is change for 1 rate point (e.g. from 2% to 3%).
        }



class AmericanOption(Option):
    def __init__(self, current_price, strike_price, expiry, option_type, today_date, sigma, n_steps, interest_rate=None, dividend_yield=0.0):
        super().__init__(current_price, strike_price, expiry, option_type, today_date)
        self.sigma = sigma
        self.n_steps = n_steps
        self.interest_rate = interest_rate if interest_rate is not None else get_zero_rate(self.time_to_maturity())
        self.dividend_yield = dividend_yield

    def option_price(self):
        return binomial_tree_price(
            current_price=self.current_price,
            strike_price=self.strike_price,
            time_to_maturity=self.time_to_maturity(),
            interest_rate=self.interest_rate,
            sigma=self.sigma,
            option_type=self.option_type,
            american=True,
            n_steps=self.n_steps,
            dividend_yield=self.dividend_yield
        )
    
    def greeks(self):
        """Approximate Greeks using finite differences for American options, now dividend-aware"""
        # Store original values to ensure they are reset
        original_current_price = self.current_price
        original_sigma = self.sigma
        original_interest_rate = self.interest_rate
        original_today_date = self.today_date
        # original_dividend_yield = self.dividend_yield # Not strictly needed to store/reset if it doesn't change during greeks calculation

        # Use small perturbations to estimate Greeks
        # Proportional bump for price-related Greeks
        h_price = original_current_price * 0.01  # 1% proportional bump for price
        if h_price == 0: h_price = 0.0001 # Avoid zero bump if price is zero

        base_price = self.option_price() # This call will now use self.dividend_yield
        
        # Delta: sensitivity to underlying price
        self.current_price = original_current_price + h_price
        up_price = self.option_price() # This call will use self.dividend_yield
        self.current_price = original_current_price - h_price
        down_price = self.option_price() # This call will use self.dividend_yield
        self.current_price = original_current_price  # Reset to original
        delta = (up_price - down_price) / (2 * h_price)
        
        # Gamma: second derivative with respect to underlying price
        gamma = (up_price - 2 * base_price + down_price) / (h_price ** 2)
        
        # Theta: sensitivity to time (approximate)
        time_bump_days = 1
        time_bump_years = time_bump_days / 365.25
        
        self.today_date = date.fromordinal(original_today_date.toordinal() + time_bump_days)
        theta_price = self.option_price() # This call will use self.dividend_yield
        self.today_date = original_today_date  # Reset
        theta = (theta_price - base_price) / time_bump_years
        
        # Vega: sensitivity to volatility
        h_vol_american = 0.01
        self.sigma = original_sigma + h_vol_american
        vega_price = self.option_price() # This call will use self.dividend_yield
        self.sigma = original_sigma  # Reset
        vega = ((vega_price - base_price) / h_vol_american) * 0.01 # Scaled as per previous fix
        
        # Rho: sensitivity to interest rate
        h_rate_american = 0.01
        self.interest_rate = original_interest_rate + h_rate_american
        rho_price = self.option_price() # This call will use self.dividend_yield
        self.interest_rate = original_interest_rate  # Reset
        rho = ((rho_price - base_price) / h_rate_american) * 0.01 # Scaled as per previous fix
        
        # Rho_q (sensitivity to dividend yield) - Optional, but good to consider for completeness
        # h_div = 0.005 # 0.5% bump for dividend yield sensitivity
        # original_div_yield = self.dividend_yield
        # self.dividend_yield = original_div_yield + h_div
        # price_rho_q = self.option_price()
        # self.dividend_yield = original_div_yield # Reset
        # rho_q = ((price_rho_q - base_price) / h_div) * 0.01

        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
            # 'rho_q': rho_q # If you decide to add sensitivity to dividend yield
        }



class BarrierOption(Option):
    def __init__(self, current_price, strike_price, expiry, option_type, today_date, sigma, barrier_price, method, n_steps, interest_rate=None, dividend_yield=0.0):
        super().__init__(current_price, strike_price, expiry, option_type, today_date)
        self.sigma = sigma
        self.barrier_price = barrier_price
        self.method = method
        self.n_steps = n_steps # For binomial, this is tree steps; for MC, this could be paths or simulation steps for barrier monitoring
        self.interest_rate = interest_rate if interest_rate is not None else get_zero_rate(self.time_to_maturity())
        self.dividend_yield = dividend_yield

    def option_price(self):
        if self.method == "monte-carlo":
            # Assuming self.n_steps passed to BarrierOption was intended for n_paths for MC pricing
            # And n_mc_sim_steps for monitoring the barrier along the path if different from 1 (default in func is 100)
            n_mc_sim_steps = 100 # Or make this an attribute if you want to control it per option instance
            return monte_carlo_barrier_price(
                current_price=self.current_price,
                strike_price=self.strike_price,
                time_to_maturity=self.time_to_maturity(),
                interest_rate=self.interest_rate,
                sigma=self.sigma,
                barrier_price=self.barrier_price,
                n_paths=self.n_steps, # Using self.n_steps as n_paths for MC
                n_steps=n_mc_sim_steps, # Path simulation steps for barrier check
                dividend_yield=self.dividend_yield
            )
        elif self.method == "binomial":
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
            raise ValueError("Unsupported method. Use 'monte-carlo' or 'binomial'.")

    def greeks(self):
        """Calculate Greeks using finite differences for barrier options"""
        # Store original values
        original_current_price = self.current_price
        original_sigma = self.sigma
        original_interest_rate = self.interest_rate
        original_today_date = self.today_date

        # Define bump sizes
        # Proportional bump for price, smaller absolute bumps for vol and rate
        h_price = original_current_price * 0.001 
        if h_price == 0: h_price = 0.0001 # Avoid zero bump if price is zero
        h_vol = 0.005  # 0.5% bump for vega
        h_rate = 0.005 # 0.5% bump for rho
        time_bump_days = 1
        time_bump_years = time_bump_days / 365.25

        base_price = self.option_price()
        
        # Delta
        self.current_price = original_current_price + h_price
        up_price = self.option_price()
        self.current_price = original_current_price - h_price
        down_price = self.option_price()
        self.current_price = original_current_price # Reset
        delta = (up_price - down_price) / (2 * h_price)
        
        # Gamma
        # Uses the same up_price, down_price, base_price
        gamma = (up_price - 2 * base_price + down_price) / (h_price ** 2)
        
        # Vega
        self.sigma = original_sigma + h_vol
        vega_price = self.option_price()
        self.sigma = original_sigma # Reset
        vega = ((vega_price - base_price) / h_vol) * 0.01
        
        # Theta
        self.today_date = date.fromordinal(original_today_date.toordinal() + time_bump_days)
        theta_price = self.option_price()
        self.today_date = original_today_date # Reset
        theta = (theta_price - base_price) / time_bump_years
        
        # Rho
        self.interest_rate = original_interest_rate + h_rate
        rho_price = self.option_price()
        self.interest_rate = original_interest_rate # Reset
        rho = ((rho_price - base_price) / h_rate) * 0.01
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }


class BasketOption(Option):
    def __init__(self, current_prices, weights, strike_price, expiry, option_type, today_date, sigmas, correlation_matrix, interest_rate=None, dividend_yields=None):
        # Calculate basket value for the base Option class
        basket_value = sum(w * p for w, p in zip(weights, current_prices))
        super().__init__(basket_value, strike_price, expiry, option_type, today_date)
        self.current_prices = np.array(current_prices) # Ensure numpy array for operations
        self.weights = np.array(weights)
        self.sigmas = np.array(sigmas) # Changed from self.sigma to self.sigmas for clarity (list of sigmas)
        self.correlation_matrix = np.array(correlation_matrix)
        self.interest_rate = interest_rate if interest_rate is not None else get_zero_rate(self.time_to_maturity())
        # Handle dividend yields for each asset in the basket
        if dividend_yields is None:
            self.dividend_yields = np.zeros(len(current_prices))
        else:
            self.dividend_yields = np.array(dividend_yields)
        if len(self.dividend_yields) != len(current_prices):
            raise ValueError("Length of dividend_yields must match length of current_prices")


    def option_price(self):
        return monte_carlo_basket_price(
            current_prices=self.current_prices,
            weights=self.weights,
            strike_price=self.strike_price,
            time_to_maturity=self.time_to_maturity(),
            interest_rate=self.interest_rate,
            sigma=self.sigmas, # Pass self.sigmas as the 'sigma' argument
            correlation_matrix=self.correlation_matrix,
            option_type=self.option_type,
            n_paths=100000, # Consider making this an attribute or parameter
            dividend_yields=self.dividend_yields # Pass dividend yields
        )
    
    def greeks(self):
        """Calculate basket option Greeks - returns delta for each underlying"""
        # Store original values
        original_current_prices = self.current_prices.copy()
        original_sigmas = self.sigmas.copy()
        original_interest_rate = self.interest_rate
        original_today_date = self.today_date
        # original_current_basket_price = self.current_price # This is the weighted sum, calculated by super().__init__

        base_option_price = self.option_price()
        deltas = []
        
        # Deltas for each underlying asset
        for i in range(len(self.current_prices)):
            h_asset_i = original_current_prices[i] * 0.001 # Proportional bump for this specific asset
            if h_asset_i == 0: h_asset_i = 0.00001 # Avoid zero bump

            # Bump up i-th asset price
            temp_prices_up = original_current_prices.copy()
            temp_prices_up[i] += h_asset_i
            self.current_prices = temp_prices_up
            self.current_price = sum(w * p for w, p in zip(self.weights, self.current_prices)) # Update overall basket price for option pricer
            price_up = self.option_price()
            
            # Bump down i-th asset price
            temp_prices_down = original_current_prices.copy()
            temp_prices_down[i] -= h_asset_i
            self.current_prices = temp_prices_down
            self.current_price = sum(w * p for w, p in zip(self.weights, self.current_prices)) # Update overall basket price
            price_down = self.option_price()
            
            delta_i = (price_up - price_down) / (2 * h_asset_i)
            deltas.append(delta_i)
        
        self.current_prices = original_current_prices # Reset asset prices
        self.current_price = sum(w * p for w, p in zip(self.weights, self.current_prices)) # Reset overall basket price

        # Vega (sensitivity to a parallel shift in all volatilities)
        # This is a simplified basket vega. More complex vegas could be calculated per asset.
        h_vol = 0.005 # 0.5% bump for all volatilities
        temp_sigmas_up = original_sigmas + h_vol
        self.sigmas = temp_sigmas_up
        price_vega_up = self.option_price()
        self.sigmas = original_sigmas # Reset sigmas
        basket_vega = ((price_vega_up - base_option_price) / h_vol) * 0.01 # Scaled
        
        # Theta
        time_bump_days = 1
        time_bump_years = time_bump_days / 365.25
        self.today_date = date.fromordinal(original_today_date.toordinal() + time_bump_days)
        price_theta = self.option_price()
        self.today_date = original_today_date # Reset
        basket_theta = (price_theta - base_option_price) / time_bump_years
        
        # Rho
        h_rate = 0.005 # 0.5% bump for rho
        self.interest_rate = original_interest_rate + h_rate
        price_rho = self.option_price()
        self.interest_rate = original_interest_rate # Reset
        basket_rho = ((price_rho - base_option_price) / h_rate) * 0.01 # Scaled
        
        # Portfolio volatility (already calculated, but useful to return)
        # This is an attribute of the basket itself, not a greek of the option value in the traditional sense, but often reported alongside.
        # Ensure _calculate_portfolio_volatility is correct and uses self.sigmas
        # portfolio_vol = self._calculate_portfolio_volatility() 
        # For now, I will remove portfolio_vol from greeks output as it's not a sensitivity of the option price in the same way.
        # If needed, it can be called separately from the BasketOption instance.

        return {
            'deltas': deltas,  # Individual deltas for each asset
            'vega': basket_vega, # Already scaled
            'theta': basket_theta, # Per year
            'rho': basket_rho # Already scaled
            # 'portfolio_volatility': portfolio_vol # Removed for now
        }
    
    def _calculate_portfolio_volatility(self):
        """Calculate the portfolio volatility given weights, individual volatilities, and correlation matrix"""
        n = len(self.weights)
        portfolio_variance = 0
        
        # Ensure sigmas is used here, not self.sigma (which was an old single value)
        for i in range(n):
            for j in range(n):
                portfolio_variance += (self.weights[i] * self.weights[j] * 
                                     self.sigmas[i] * self.sigmas[j] * # Changed from self.sigma[i] to self.sigmas[i]
                                     self.correlation_matrix[i,j]) # Assuming correlation_matrix is a numpy array
        
        if portfolio_variance < 0: # Should not happen with valid inputs
            return 0.0
        return np.sqrt(portfolio_variance)
