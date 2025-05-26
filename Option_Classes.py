from datetime import date
from functions import (
    black_scholes_price,
    binomial_tree_price,
    monte_carlo_price,
    monte_carlo_barrier_price,
    binomial_barrier_price_up_and_in
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
    def __init__(self, current_price, strike_price, expiry, option_type, today_date, interest_rate, sigma):
        super().__init__(current_price, strike_price, expiry, option_type, today_date)
        self.interest_rate = interest_rate
        self.sigma = sigma

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
    def __init__(self, current_price, strike_price, expiry, option_type, today_date, interest_rate, sigma, barrier_price, method, n_steps):
        super().__init__(current_price, strike_price, expiry, option_type, today_date)
        self.interest_rate = interest_rate
        self.sigma = sigma
        self.barrier_price = barrier_price
        self.n_steps = n_steps
        self.method = method

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

# ZeroCurve will storre the zero rates and discount factors for given maturities and zero rates 
# the class will have the following methods:
# - add_zero_rate(maturity, zero_rate): add a zero rate to the curve
# - get_zero_rate(maturity): get the zero rate for a given maturity

# import necessary libraries
import numpy as np
import math

import math

class ZeroCurve:
    def __init__(self):
        self.maturities = []
        self.zero_rates = []
        self.AtMats = []
        self.discount_factors = []

    def add_zero_rate(self, maturity, zero_rate):
        self.maturities.append(maturity)
        self.zero_rates.append(zero_rate)
        self.AtMats.append(math.exp(zero_rate * maturity))
        self.discount_factors.append(1 / self.AtMats[-1])

    def add_discount_factor(self, maturity, discount_factor):
        self.maturities.append(maturity)
        self.discount_factors.append(discount_factor)
        self.AtMats.append(1 / discount_factor)
        self.zero_rates.append(math.log(1 / discount_factor) / maturity)

    def get_AtMat(self, maturity):
        if maturity in self.maturities:
            return self.AtMats[self.maturities.index(maturity)]
        else:
            return exp_interp(self.maturities, self.AtMats, maturity)

    def get_discount_factor(self, maturity):
        if maturity in self.maturities:
            return self.discount_factors[self.maturities.index(maturity)]
        else:
            return exp_interp(self.maturities, self.discount_factors, maturity)

    def get_zero_rate(self, maturity):
        if maturity in self.maturities:
            return self.zero_rates[self.maturities.index(maturity)]
        else:
            return math.log(self.get_AtMat(maturity)) / maturity

    def get_zero_curve(self):
        return self.maturities, self.discount_factors

    def npv(self, cash_flows):
        npv = 0
        for maturity in cash_flows.get_maturities():
            npv += cash_flows.get_cash_flow(maturity) * self.get_discount_factor(maturity)
        return npv


def exp_interp(xs, ys, x):
    xs = list(xs)
    ys = list(ys)

    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1] 

    idx = max(i for i in range(len(xs)) if xs[i] <= x)
    x0, x1 = xs[idx], xs[idx + 1]
    y0, y1 = ys[idx], ys[idx + 1]

    rate = (math.log(y1) - math.log(y0)) / (x1 - x0)
    y = y0 * math.exp(rate * (x - x0))
    return y


class YieldCurve(ZeroCurve):
    def __init__(self):
        super().__init__()
        self.portfolio = []

    def set_constituent_portfolio(self, portfolio):
        self.portfolio = portfolio
        self.portfolio.set_cash_flows()

    def bootstrap(self):
        bank_bills = self.portfolio.get_bank_bills()
        bonds = self.portfolio.get_bonds()

        self.add_zero_rate(0, 0)

        for bank_bill in bank_bills:
            self.add_discount_factor(
                bank_bill.get_maturity(),
                bank_bill.get_price() / bank_bill.get_face_value()
            )

        for bond in bonds:
            pv = 0
            bond_dates = [cf[0] for cf in bond.get_cash_flows()]
            bond_amounts = [cf[1] for cf in bond.get_cash_flows()]

            for i in range(1, len(bond_amounts) - 1):
                pv += bond_amounts[i] * self.get_discount_factor(bond_dates[i])

            print("PV of all the cashflows except maturity is:", pv)
            print("The bond price is:", bond.get_price())
            print("The last cashflow is:", bond_amounts[-1])

            df = (bond.get_price() - pv) / bond_amounts[-1]
            self.add_discount_factor(bond.get_maturity(), df)
