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
            n_paths=10000
        )

