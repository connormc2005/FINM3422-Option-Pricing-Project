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
    def __init__(self, current_price, strike_price, expiry, option_type, today_date, sigma, interest_rate=None):
        super().__init__(current_price, strike_price, expiry, option_type, today_date)
        self.sigma = sigma
        self.interest_rate = interest_rate if interest_rate is not None else get_zero_rate(self.time_to_maturity())

    def option_price(self):
        return black_scholes_price(
            option_type=self.option_type,
            current_price=self.current_price,
            strike_price=self.strike_price,
            time_to_maturity=self.time_to_maturity(),
            interest_rate=self.interest_rate,
            sigma=self.sigma
        )
    
    def greeks(self):
        """Approximate Greeks using finite differences for American options"""
        # Use small perturbations to estimate Greeks
        h = 0.01  # 1% bump
        base_price = self.option_price()
        
        # Delta: sensitivity to underlying price
        self.current_price += h
        up_price = self.option_price()
        self.current_price -= 2 * h
        down_price = self.option_price()
        self.current_price += h  # Reset to original
        delta = (up_price - down_price) / (2 * h)
        
        # Gamma: second derivative with respect to underlying price
        gamma = (up_price - 2 * base_price + down_price) / (h ** 2)
        
        # Theta: sensitivity to time (approximate)
        original_ttm = self.time_to_maturity()
        time_bump = 1/365  # 1 day
        self.today_date = date.fromordinal(self.today_date.toordinal() + 1)
        theta_price = self.option_price()
        self.today_date = date.fromordinal(self.today_date.toordinal() - 1)  # Reset
        theta = (theta_price - base_price) / time_bump
        
        # Vega: sensitivity to volatility
        self.sigma += 0.01  # 1% volatility bump
        vega_price = self.option_price()
        self.sigma -= 0.01  # Reset
        vega = (vega_price - base_price) / 0.01
        
        # Rho: sensitivity to interest rate
        self.interest_rate += 0.01  # 1% rate bump
        rho_price = self.option_price()
        self.interest_rate -= 0.01  # Reset
        rho = (rho_price - base_price) / 0.01
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }



class AmericanOption(Option):
    def __init__(self, current_price, strike_price, expiry, option_type, today_date, sigma, n_steps, interest_rate=None):
        super().__init__(current_price, strike_price, expiry, option_type, today_date)
        self.sigma = sigma
        self.n_steps = n_steps
        self.interest_rate = interest_rate if interest_rate is not None else get_zero_rate(self.time_to_maturity())

    def option_price(self):
        return binomial_tree_price(
            current_price=self.current_price,
            strike_price=self.strike_price,
            time_to_maturity=self.time_to_maturity(),
            interest_rate=self.interest_rate,
            sigma=self.sigma,
            option_type=self.option_type,
            american=True,
            n_steps=self.n_steps
        )
    
    def greeks(self):
        """Approximate Greeks using finite differences for American options"""
        # Use small perturbations to estimate Greeks
        h = 0.01  # 1% bump
        base_price = self.option_price()
        
        # Delta: sensitivity to underlying price
        self.current_price += h
        up_price = self.option_price()
        self.current_price -= 2 * h
        down_price = self.option_price()
        self.current_price += h  # Reset to original
        delta = (up_price - down_price) / (2 * h)
        
        # Gamma: second derivative with respect to underlying price
        gamma = (up_price - 2 * base_price + down_price) / (h ** 2)
        
        # Theta: sensitivity to time (approximate)
        original_ttm = self.time_to_maturity()
        time_bump = 1/365  # 1 day
        self.today_date = date.fromordinal(self.today_date.toordinal() + 1)
        theta_price = self.option_price()
        self.today_date = date.fromordinal(self.today_date.toordinal() - 1)  # Reset
        theta = (theta_price - base_price) / time_bump
        
        # Vega: sensitivity to volatility
        self.sigma += 0.01  # 1% volatility bump
        vega_price = self.option_price()
        self.sigma -= 0.01  # Reset
        vega = (vega_price - base_price) / 0.01
        
        # Rho: sensitivity to interest rate
        self.interest_rate += 0.01  # 1% rate bump
        rho_price = self.option_price()
        self.interest_rate -= 0.01  # Reset
        rho = (rho_price - base_price) / 0.01
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }



class BarrierOption(Option):
    def __init__(self, current_price, strike_price, expiry, option_type, today_date, sigma, barrier_price, method, n_steps, interest_rate=None):
        super().__init__(current_price, strike_price, expiry, option_type, today_date)
        self.sigma = sigma
        self.barrier_price = barrier_price
        self.method = method
        self.n_steps = n_steps
        self.interest_rate = interest_rate if interest_rate is not None else get_zero_rate(self.time_to_maturity())

    def option_price(self):
        if self.method == "monte-carlo":
            return monte_carlo_barrier_price(
                self.current_price,
                self.strike_price,
                self.time_to_maturity(),
                self.interest_rate,
                self.sigma,
                self.barrier_price,
                n_paths=10000,
                n_steps=100
            )
        elif self.method == "binomial":
            return binomial_barrier_price_up_and_in(
                self.option_type,
                self.current_price,
                self.strike_price,
                self.time_to_maturity(),
                self.interest_rate,
                self.sigma,
                self.barrier_price,
                self.n_steps
            )
        else:
            raise ValueError("Unsupported method. Use 'monte-carlo' or 'binomial'.")

    def greeks(self):
        """Calculate Greeks using finite differences for barrier options"""
        h = 0.01
        base_price = self.option_price()
        
        # Delta
        self.current_price += h
        up_price = self.option_price()
        self.current_price -= 2 * h
        down_price = self.option_price()
        self.current_price += h
        delta = (up_price - down_price) / (2 * h)
        
        # Gamma
        gamma = (up_price - 2 * base_price + down_price) / (h ** 2)
        
        # Vega
        self.sigma += 0.01
        vega_price = self.option_price()
        self.sigma -= 0.01
        vega = (vega_price - base_price) / 0.01
        
        # Theta (approximate)
        self.today_date = date.fromordinal(self.today_date.toordinal() + 1)
        theta_price = self.option_price()
        self.today_date = date.fromordinal(self.today_date.toordinal() - 1)
        theta = (theta_price - base_price) / (1/365)
        
        # Rho
        self.interest_rate += 0.01
        rho_price = self.option_price()
        self.interest_rate -= 0.01
        rho = (rho_price - base_price) / 0.01
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }  


class BasketOption(Option):
    def __init__(self, current_prices, weights, strike_price, expiry, option_type, today_date, sigma, correlation_matrix, interest_rate=None):
        # Calculate basket value for the base Option class
        basket_value = sum(w * p for w, p in zip(weights, current_prices))
        super().__init__(basket_value, strike_price, expiry, option_type, today_date)
        self.current_prices = current_prices
        self.weights = weights
        self.sigma = sigma
        self.correlation_matrix = correlation_matrix
        self.interest_rate = interest_rate if interest_rate is not None else get_zero_rate(self.time_to_maturity())

    def option_price(self):
        return monte_carlo_basket_price(
            current_prices=self.current_prices,
            weights=self.weights,
            strike_price=self.strike_price,
            time_to_maturity=self.time_to_maturity(),
            interest_rate=self.interest_rate,
            sigma=self.sigma,
            correlation_matrix=self.correlation_matrix,
            option_type=self.option_type,
            n_paths=100000
        )
    
    def greeks(self):
        """Calculate basket option Greeks - returns delta for each underlying"""
        base_price = self.option_price()
        deltas = []
        
        h = 0.01  # 1% bump for each underlying
        
        # Calculate delta for each underlying asset
        for i in range(len(self.current_prices)):
            # Bump the i-th asset price
            self.current_prices[i] += h
            self.current_price = sum(price * weight for price, weight in zip(self.current_prices, self.weights))
            up_price = self.option_price()
            
            # Bump down
            self.current_prices[i] -= 2 * h
            self.current_price = sum(price * weight for price, weight in zip(self.current_prices, self.weights))
            down_price = self.option_price()
            
            # Reset
            self.current_prices[i] += h
            self.current_price = sum(price * weight for price, weight in zip(self.current_prices, self.weights))
            
            # Delta for i-th asset
            delta_i = (up_price - down_price) / (2 * h)
            deltas.append(delta_i)
        
        # Overall basket volatility for vega calculation
        portfolio_vol = self._calculate_portfolio_volatility()
        
        # Vega - sensitivity to overall volatility
        # Bump all volatilities proportionally
        vol_bump = 0.01
        original_vols = self.sigma.copy()
        self.sigma = [vol + vol_bump for vol in self.sigma]
        vega_price = self.option_price()
        self.sigma = original_vols
        vega = (vega_price - base_price) / vol_bump
        
        # Theta
        self.today_date = date.fromordinal(self.today_date.toordinal() + 1)
        theta_price = self.option_price()
        self.today_date = date.fromordinal(self.today_date.toordinal() - 1)
        theta = (theta_price - base_price) / (1/365)
        
        # Rho
        self.interest_rate += 0.01
        rho_price = self.option_price()
        self.interest_rate -= 0.01
        rho = (rho_price - base_price) / 0.01
        
        return {
            'deltas': deltas,  # Individual deltas for each asset
            'vega': vega,
            'theta': theta,
            'rho': rho,
            'portfolio_volatility': portfolio_vol
        }
    
    def _calculate_portfolio_volatility(self):
        """Calculate the portfolio volatility given weights, individual volatilities, and correlation matrix"""
        n = len(self.weights)
        portfolio_variance = 0
        
        for i in range(n):
            for j in range(n):
                portfolio_variance += (self.weights[i] * self.weights[j] * 
                                     self.sigma[i] * self.sigma[j] * 
                                     self.correlation_matrix[i][j])
        
        return np.sqrt(portfolio_variance)
