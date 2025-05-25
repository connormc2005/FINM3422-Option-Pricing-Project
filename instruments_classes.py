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
