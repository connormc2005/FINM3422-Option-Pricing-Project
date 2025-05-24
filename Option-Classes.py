# option_classes.py - Clean version of your classes

from datetime import date
from functions import black_scholes_price, binomial_tree_price


class Option:
    def __init__(self, current_price, strike_price, expiry, option_type, today_date):
        self.current_price = current_price
        self.strike_price = strike_price
        self.expiry = expiry
        self.option_type = option_type
        self.today_date = today_date


class EuropeanOption(Option):
    def __init__(self, current_price, strike_price, expiry, option_type, today_date, interest_rate, sigma):
        super().__init__(current_price, strike_price, expiry, option_type, today_date)
        self.interest_rate = interest_rate
        self.sigma = sigma 

    def time_to_maturity(self):
        return (self.expiry - self.today_date).days / 365.25

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
    def __init__(self, current_price, strike_price, expiry, option_type, today_date, interest_rate, sigma, n_steps):
        super().__init__(current_price, strike_price, expiry, option_type, today_date)
        self.interest_rate = interest_rate
        self.sigma = sigma
        self.n_steps = n_steps

    def time_to_maturity(self):
        return (self.expiry - self.today_date).days / 365.25 
    
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