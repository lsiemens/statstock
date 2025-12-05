import numpy as np
from matplotlib import pyplot as plt


class Forcast:
    def __init__(self, rebalance, weights, V_0=1000, cash_flow=None):
        """
        Parameters
        ----------
        rebalance : Rebalance
            Subclass of Rebalance for sampling of returns
        weights : array
            Weights of stocks in portfolio
        V_0 : float
            Initial portfolio value
        cash_flow : function(int year)
            the function cash_flow should return the target cash flow given the
            year since inception, if the function is None cash flows are assumed
            to be zero
        """

        self.ticker = rebalance.tickers
        self.width = rebalance.width
        self.logprice = rebalance.logprice[:, -1]
        self.interval = rebalance.interval
        self.weights = weights
        self.cash_flow = cash_flow

        self.V_0 = V_0

        self.sample_lnRet = rebalance.sample_lnRet

        match self.interval:
            case "OneDay":
                self.intervals_per_year = 5*52
            case "OneWeek":
                self.intervals_per_year = 52
            case _:
                raise NotImplimentedError(f"The interval {self.interval} is not implemented")


        self.spread = 0.0001 # 0.01%
        self.brokerage_fees = 10 # $10 on each trade
        self.tax_rate = 0.15 # 15%
        self.inclusion_rate = 0.5 # 50%

    def fees(self, delta_shares, logprices):
        did_trade = np.abs(delta_shares) > 1
        brokerage_fees = np.sum(self.brokerage_fees * did_trade)
        spread = self.spread*(np.abs(delta_shares) @ np.exp(logprices))
        return brokerage_fees + spread

    def tax(self, delta_shares, logprices, ACB):
        delta_price = np.exp(logprices) - ACB
        net = -np.minimum(delta_shares, 0) @ delta_price
        net = net - self.fees(delta_shares, logprices)
        return self.inclusion_rate*self.tax_rate*np.maximum(net, 0)

    def rebalance(self, n_shares, logprices, ACB, year):
        """
        Outline:
        calculate currrent weights
        find target cash flow
        if withdrals
        """

        if self.cash_flow is None:
            target_cash_flow = 0
        else:
            target_cash_flow = self.cash_flow(year)

        frictionless_value = target_cash_flow + n_shares @ np.exp(logprices)
        target_shares = frictionless_value*self.weights*np.exp(-logprices)

        def final_cash(alpha):
          delta_shares = target_shares*(1 - alpha) - n_shares
          
          book_value = (1 - alpha)*frictionless_value
          fees = self.fees(delta_shares, logprices)
          taxes = self.tax(delta_shares, logprices, ACB)
          cash = frictionless_value - book_value - fees - taxes
          return cash

        alpha_min = 0
        cash_min = final_cash(alpha_min)
        alpha_max = 1
        cash_max = final_cash(alpha_max)

        assert cash_min < 0, "Balancing is expected to have costs"
        if cash_max < 0:
            print("Portfolio can not support withdrawals") 
            return (0*n_shares, 0, 0, 0)        

        max_depth = 20
        target = 0.005
        for i in range(max_depth):
            alpha_mid = (alpha_max + alpha_min)/2
            cash_mid = final_cash(alpha_mid)

            if (cash_mid > 0) and (cash_mid < 2*target):
                break

            if cash_mid < target:
                alpha_min = alpha_mid
                cash_min = cash_mid
            else:
                alpha_max = alpha_mid
                cash_min = cash_mid

        delta_shares = (1 - alpha_mid)*target_shares - n_shares
        fees = self.fees(delta_shares, logprices)
        taxes = self.tax(delta_shares, logprices, ACB)
        return delta_shares, taxes, fees, target_cash_flow

    def single_forcast(self, n_intervals):
        n_shares = self.V_0*self.weights*np.exp(-self.logprice)

        lnRet = self.sample_lnRet(n_intervals)

        logprices = self.logprice[:, None] + np.cumsum(lnRet, axis=1)

        value = np.empty(n_intervals + 1)
        value[0] = self.V_0
        taxes = [1e-2]
        fees = [1e-2]
        contributions = [self.V_0]
        start = 0
        year = 0
        ACB = np.exp(logprices[:, 0])
        while (start < n_intervals):
            end = start + self.intervals_per_year
            value[1 + start:1 + end] = n_shares @ np.exp(logprices[:, start:end])

            if end < n_intervals:
                delta_shares, tax, fee, cash_flow = self.rebalance(n_shares,
                                                                   logprices[:, end],
                                                                   ACB, year)
                mask = delta_shares > 0
                ACB[mask] = ((n_shares*ACB + delta_shares*np.exp(logprices[:, end]))/(n_shares + delta_shares))[mask]
            else:
                delta_shares = 0*n_shares
                tax = 0
                fee = 0
                cash_flow = 0
            taxes.append(taxes[-1] + tax)
            fees.append(fees[-1] + fee)
            contributions.append(contributions[-1] + cash_flow)
            n_shares = n_shares + delta_shares

            start += self.intervals_per_year
            year += 1

        taxes = np.array(taxes)
        fees = np.array(fees)
        contributions = np.array(contributions)
        return value, taxes, fees, contributions

    def n_forcasts(self, n_intervals, m_samples):
        values = np.empty((m_samples, n_intervals + 1))
        taxes = []
        fees = []
        contributions = []
        for m in range(m_samples):
            value, tax, fee, contribution = self.single_forcast(n_intervals)
            values[m, :] = value
            taxes.append(tax)
            fees.append(fee)
            contributions.append(contribution)
        taxes = np.array(taxes)
        fees = np.array(fees)
        contributions = np.array(contributions)
        return values, taxes, fees, contributions

    def show_n_forcasts(self, values, taxes, fees, contributions):
        n_rebalance = self.intervals_per_year*np.arange(len(taxes[0])) + 0.5
        n_rebalance[0] = 0
        n_rebalance[-1] = len(values[0]) - 1

        mean_ln_taxes = np.mean(np.log(taxes), axis=0)
        var_ln_taxes = np.var(np.log(taxes), axis=0)

        mean_taxes = np.mean(taxes, axis=0)
        median_taxes = np.exp(mean_ln_taxes)
        low_taxes = np.exp(mean_ln_taxes - np.sqrt(var_ln_taxes))
        high_taxes = np.exp(mean_ln_taxes + np.sqrt(var_ln_taxes))

        mean_ln_fees = np.mean(np.log(fees), axis=0)
        var_ln_fees = np.var(np.log(fees), axis=0)

        mean_fees = np.mean(fees, axis=0)
        median_fees = np.exp(mean_ln_fees)
        low_fees = np.exp(mean_ln_fees - np.sqrt(var_ln_fees))
        high_fees = np.exp(mean_ln_fees + np.sqrt(var_ln_fees))

        plt.plot(np.log(values.T/values[:, 0]), "k-", alpha=0.1)
        plt.scatter(n_rebalance, np.log(mean_taxes/values[0, 0]), color="r", marker=".", label="Mean taxes")
        plt.scatter(n_rebalance, mean_ln_taxes - np.log(values[0, 0]), color="r", marker="o", label="Median taxes")
        plt.fill_between(n_rebalance, np.log(low_taxes/values[0, 0]), np.log(high_taxes/values[0, 0]), color="r", alpha=0.5, label="Tax band")
        plt.scatter(n_rebalance, np.log(mean_fees/values[0, 0]), color="b", marker=".", label="Mean fees")
        plt.scatter(n_rebalance, mean_ln_fees - np.log(values[0, 0]), color="b", marker="o", label="Median fees")
        plt.fill_between(n_rebalance, np.log(low_fees/values[0, 0]), np.log(high_fees/values[0, 0]), color="b", alpha=0.5, label="Fee band")
        plt.step(n_rebalance, np.log(contributions[0]/values[0, 0]), "g-.", where="post", label="Contributions")
        plt.title("Sample portfolios: ln relative value")
        plt.xlabel(self.interval[3:])
        plt.ylabel("Relative ln portfolio value")
        plt.legend()
        plt.show()

        mean_ln_value = np.mean(np.log(values), axis=0)
        var_ln_value = np.var(np.log(values), axis=0)

        mean_value = np.exp(mean_ln_value + 0.5*var_ln_value)
        mode_value = np.exp(mean_ln_value - var_ln_value)
        band_center = np.exp(mean_ln_value)
        band_low = np.exp(mean_ln_value - np.sqrt(var_ln_value))
        band_high = np.exp(mean_ln_value + np.sqrt(var_ln_value))
        index = np.arange(len(band_center))

        plt.plot(mean_value, "k--", label="$\\bar{p}$")
        plt.plot(band_center, "k-", label="$median(p)$")
        plt.plot(mode_value, "k:", label="$mode(p)$")
        plt.fill_between(index, band_low, band_high, alpha=0.5, label="$\\bar{p} \\pm \\sigma$")
        plt.scatter(n_rebalance, mean_taxes, color="r", marker=".", label="Mean taxes")
        plt.scatter(n_rebalance, np.exp(mean_ln_taxes), color="r", marker="o", label="Median taxes")
        plt.fill_between(n_rebalance, low_taxes, high_taxes, color="r", alpha=0.5, label="Tax band")
        plt.scatter(n_rebalance, mean_fees, color="b", marker=".", label="Mean fees")
        plt.scatter(n_rebalance, np.exp(mean_ln_fees), color="b", marker="o", label="Median fees")
        plt.fill_between(n_rebalance, low_fees, high_fees, color="b", alpha=0.5, label="Fee band")
        plt.step(n_rebalance, contributions[0], "g-.", where="post", label="Contributions")
        plt.title("Expected portfolio range")
        plt.xlabel(self.interval[3:])
        plt.ylabel("Portfolio value (CAD)")
        plt.legend()
        plt.show()

        plt.plot(np.log(mean_value), "k--", label="$\\bar{p}$")
        plt.plot(np.log(band_center), "k-", label="$median(p)$")
        plt.plot(np.log(mode_value), "k:", label="$mode(p)$")
        plt.fill_between(index, np.log(band_low), np.log(band_high), alpha=0.5, label="$\\bar{p} \\pm \\sigma$")
        plt.scatter(n_rebalance, np.log(mean_taxes), color="r", marker=".", label="Mean taxes")
        plt.scatter(n_rebalance, mean_ln_taxes, color="r", marker="o", label="Median taxes")
        plt.fill_between(n_rebalance, np.log(low_taxes), np.log(high_taxes), color="r", alpha=0.5, label="Tax band")
        plt.scatter(n_rebalance, np.log(mean_fees), color="b", marker=".", label="Mean fees")
        plt.scatter(n_rebalance, mean_ln_fees, color="b", marker="o", label="Median fees")
        plt.fill_between(n_rebalance, np.log(low_fees), np.log(high_fees), color="b", alpha=0.5, label="Fee band")
        plt.step(n_rebalance, np.log(contributions[0]), "g-.", where="post", label="Contributions")
        plt.title("Expected portfolio ln range")
        plt.xlabel(self.interval[3:])
        plt.ylabel("Relative ln portfolio value")
        plt.legend()
        plt.show()

        plt.hist(np.log(values[:, -1]), bins=50, density=True, color="k", alpha=0.5, label="Final value")
        plt.hist(np.log(taxes[:, -1]), bins=50, density=True, color="r", alpha=0.5, label="Total taxes")
        plt.hist(np.log(fees[:, -1]), bins=50, density=True, color="b", alpha=0.5, label="Total fees")
        plt.axvline(np.log(contributions[0, -1]), color="g", label="Net contributions")
        plt.title("Expected final portfolio distribution")
        plt.xlabel("ln dollar value")
        plt.ylabel("Probability density")
        plt.legend()
        plt.show()

    def show_single_forcast(self, value, taxes, fees, contributions):
        n_rebalance = self.intervals_per_year*np.arange(len(taxes)) + 0.5
        n_rebalance[0] = 0
        n_rebalance[-1] = len(value) - 1

        plt.plot(value, "k-", label="Portfolio value")
        plt.step(n_rebalance, taxes, "r--", where="post", label="Taxes")
        plt.step(n_rebalance, fees, "b:", where="post", label="Fees")
        plt.step(n_rebalance, contributions, "g-.", where="post", marker="o", label="Contributions")
        plt.title("Sample portfolio: value")
        plt.xlabel(self.interval[3:])
        plt.ylabel("Portfolio value (CAD)")
        plt.legend()
        plt.show()

        plt.plot(np.log(value/value[0]), "k-")
        plt.title("Sample portfolio: ln relative value")
        plt.xlabel(self.interval[3:])
        plt.ylabel("Relative ln portfolio value")
        plt.show()
