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

class Cashflows:
    def __init__(self):
        self.cashflow_dict = {}  # {maturity: amount}

    def add_cash_flow(self, maturity, amount):
        """Add a cash flow at a specific maturity."""
        if maturity in self.cashflow_dict:
            self.cashflow_dict[maturity] += amount
        else:
            self.cashflow_dict[maturity] = amount

    def get_cash_flow(self, maturity):
        """Get the cash flow amount at a specific maturity."""
        return self.cashflow_dict.get(maturity, 0.0)

    def get_maturities(self):
        """Return a sorted list of all maturities."""
        return sorted(self.cashflow_dict.keys())

    def get_cash_flows(self):
        """Return a list of (maturity, amount) tuples sorted by maturity."""
        return sorted(self.cashflow_dict.items())

    def present_value(self, zero_curve):
        """Calculate the present value of the cashflows using a ZeroCurve."""
        pv = 0.0
        for maturity, amount in self.get_cash_flows():
            df = zero_curve.get_discount_factor(maturity)
            pv += amount * df
        return pv

    def __str__(self):
        return "\n".join([f"Maturity: {m}, Amount: {a}" for m, a in self.get_cash_flows()])
    
class Bank_bill:
    def __init__(self):
        self.maturity = None
        self.ytm = None
        self.cash_flows = []

    def set_maturity(self, maturity):
        self.maturity = maturity

    def set_ytm(self, ytm):
        self.ytm = ytm

    def set_cash_flows(self):
        self.cash_flows = [(self.maturity, 100)]

    def get_cash_flows(self):
        return self.cash_flows

    def get_maturities(self):
        return [cf[0] for cf in self.cash_flows]

    def get_amounts(self):
        return [cf[1] for cf in self.cash_flows]

    def pull_request(self, zero_curve):
        return zero_curve.get_zero_rate(self.maturity)
    

class Bond:
    def __init__(self):
        self.maturity = None
        self.coupon = None
        self.frequency = None
        self.face_value = None
        self.ytm = None
        self.cash_flows = []

    def set_maturity(self, maturity):
        self.maturity = maturity

    def set_coupon(self, coupon):
        self.coupon = coupon

    def set_frequency(self, frequency):
        self.frequency = frequency

    def set_face_value(self, face_value):
        self.face_value = face_value

    def set_ytm(self, ytm):
        self.ytm = ytm

    def set_cash_flows(self):
        self.cash_flows = []
        total_payments = int(self.maturity * self.frequency)
        coupon_payment = self.coupon / self.frequency * self.face_value
        for i in range(1, total_payments + 1):
            t = i / self.frequency
            amount = coupon_payment
            if i == total_payments:
                amount += self.face_value
            self.cash_flows.append((t, amount))

    def get_cash_flows(self):
        return self.cash_flows

    def get_maturities(self):
        return [cf[0] for cf in self.cash_flows]

    def get_amounts(self):
        return [cf[1] for cf in self.cash_flows]

    def pull_request(self, zero_curve):
        return [zero_curve.get_zero_rate(t) for t in self.get_maturities()]


class Portfolio:
    def __init__(self):
        self.bonds = []
        self.bank_bills = []
        self.cash_flows = []

    def add_bond(self, bond):
        self.bonds.append(bond)

    def add_bank_bill(self, bank_bill):
        self.bank_bills.append(bank_bill)

    def set_cash_flows(self):
        self.cash_flows = []
        for bond in self.bonds:
            self.cash_flows.extend(bond.get_cash_flows())
        for bank_bill in self.bank_bills:
            self.cash_flows.extend(bank_bill.get_cash_flows())

    def get_cash_flows(self):
        return self.cash_flows

    def get_maturities(self):
        return [cf[0] for cf in self.cash_flows]

    def get_amounts(self):
        return [cf[1] for cf in self.cash_flows]

    def pull_request(self, zero_curve):
        rates = []
        for bond in self.bonds:
            rates.extend(bond.pull_request(zero_curve))
        for bill in self.bank_bills:
            rates.append(bill.pull_request(zero_curve))
        return rates

import numpy as np

class ZeroCurve:
    def __init__(self):
        self.zero_rates = {}

    def add_zero_rate(self, maturity, rate):
        self.zero_rates[maturity] = rate

    def get_zero_rate(self, maturity):
        maturities = sorted(self.zero_rates.keys())
        rates = [self.zero_rates[m] for m in maturities]
        return float(np.interp(maturity, maturities, rates))

    def get_discount_factor(self, maturity):
        rate = self.get_zero_rate(maturity)
        return np.exp(-rate * maturity)

    def get_AtMat(self, maturity):
        return 1 / self.get_discount_factor(maturity)

    def get_zero_curve(self):
        maturities = sorted(self.zero_rates.keys())
        dfs = [self.get_discount_factor(m) for m in maturities]
        return [maturities, dfs]

    def npv(self, instrument):
        pv = 0
        for t, amt in instrument.get_cash_flows():
            rate = self.get_zero_rate(t)
            discount_factor = (1 + rate) ** (-t)
            pv += amt * discount_factor
        return pv   
