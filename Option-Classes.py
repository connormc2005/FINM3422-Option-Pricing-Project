# Necessary libraries
import numpy as np
import pandas as pd
import datetime as dt 



class Option:
    def __init__(self, current_price, strike_price, expiry, option_type, today_date):
        self.current_price = current_price
        self.strike_price = strike_price
        self.expiry = expiry
        self.option_type = option_type
        self.today_date = today_date


class EuropeanOption(Option):
    def__init__(self, current_price, strike_price, expiry, option_type, interest_rate, sigma):
        super().__init__(current_price, strike_price, expiry, option_type, today_date):
        self.interest_rate = interest_rate
        self.sigma = sigma 

    def time_to_expiry(self):
        return (self.expiry - self.today_date).days / 365.25

    def option_price(self):
       from functions import black_scholes_price
       
       return black_scholes_price(self.current_price, 
                                  self.strike_price, 
                                  self.time_to_expiry(), 
                                  self.interest_rate, 
                                  self.sigma, 
                                  self.option_type):




class AmericanOption(Option):
    def __init__(self, current_price, strike_price, expiry, option_type, today_date, interest_rate, sigma, n_steps):
        super().__init__(current_price, strike_price, expiry, option_type, today_date)
        self.interest_rate = interest_rate
        self.sigma = sigma
        self.n_steps = n_steps:

    def time_to_expiry(self):
        return (self.expiry - self.today_date).days / 365.25 
    
    def option_price(self):
        from functions import binomial_tree_price

        return binomial_tree_price(self.current_price, 
                                  self.strike_price, 
                                  self.time_to_expiry(), 
                                  self.interest_rate, 
                                  self.sigma, 
                                  self.option_type,
                                  self.n_steps):
    



    class BarrierOption(Option):