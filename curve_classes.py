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