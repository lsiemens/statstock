import numpy as np
from matplotlib import pyplot as plt


class Forcast:
    def __init__(self, rebalance, weights, V_0=1000):
        self.ticker = rebalance.tickers
        self.width = rebalance.width
        self.logprice = rebalance.logprice[:, -1]
        self.interval = rebalance.interval

        self.V_0 = V_0
        self.n_shares = V_0*weights*np.exp(-self.logprice)

        self.sample_lnRet = rebalance.sample_lnRet

    def single_forcast(self, n_intervals):
        lnRet = self.sample_lnRet(n_intervals)

        logprices = self.logprice[:, None] + np.cumsum(lnRet, axis=1)

        value = np.empty(n_intervals + 1)
        value[0] = self.V_0
        value[1:] = self.n_shares @ np.exp(logprices)
        return value

    def n_forcasts(self, n_intervals, m_samples):
        values = np.empty((m_samples, n_intervals + 1))
        for m in range(m_samples):
            values[m, :] = self.single_forcast(n_intervals)
        return values

    def show_n_forcasts(self, values):
        for value in values:
            plt.plot(value)
            plt.title("Sample portfolios: value")
            plt.xlabel(self.interval[3:])
            plt.ylabel("Portfolio value (CAD)")
        plt.show()

        for value in values:
            plt.plot(np.log(value/value[0]))
            plt.title("Sample portfolios: ln relative value")
            plt.xlabel(self.interval[3:])
            plt.ylabel("Relative ln portfolio value")
        plt.show()

        mean_ln_value = np.mean(np.log(values), axis=0)
        var_ln_value = np.var(np.log(values), axis=0)

        mean_value = np.exp(mean_ln_value + 0.5*var_ln_value)
        band_center = np.exp(mean_ln_value)
        band_low = np.exp(mean_ln_value - np.sqrt(var_ln_value))
        band_high = np.exp(mean_ln_value + np.sqrt(var_ln_value))
        index = np.arange(len(band_center))

        plt.plot(mean_value, "k--", label="$\\bar{p}$")
        plt.plot(band_center, "k-", label="$median(p)$")
        plt.fill_between(index, band_low, band_high, alpha=0.5, label="$\\bar{p} \\pm \\sigma$")
        plt.title("Expected portfolio range")
        plt.xlabel(self.interval[3:])
        plt.ylabel("Portfolio value (CAD)")
        plt.legend()
        plt.show()

        band_center = mean_ln_value - mean_ln_value[0]
        band_low = band_center - np.sqrt(var_ln_value)
        band_high = band_center + np.sqrt(var_ln_value)
        plt.plot(band_center, "k-", label="$\\bar{\\ln{p}}$")
        plt.fill_between(index, band_low, band_high, alpha=0.5, label="$\\bar{\\ln{p}} \\pm \\sigma$")
        plt.title("Expected portfolio ln range")
        plt.xlabel(self.interval[3:])
        plt.ylabel("Relative ln portfolio value")
        plt.show()

        for value in values[:10]:
            plt.plot(np.log(value/value[0]))
        plt.fill_between(index, band_low, band_high, alpha=0.5, label="$\\bar{\\ln{p}} \\pm \\sigma$")
        plt.title("Expected portfolio ln range")
        plt.xlabel(self.interval[3:])
        plt.ylabel("Relative ln portfolio value")
        plt.show()

    def show_single_forcast(self, value):
        plt.plot(value, "k-")
        plt.title("Sample portfolio: value")
        plt.xlabel(self.interval[3:])
        plt.ylabel("Portfolio value (CAD)")
        plt.show()

        plt.plot(np.log(value/value[0]), "k-")
        plt.title("Sample portfolio: ln relative value")
        plt.xlabel(self.interval[3:])
        plt.ylabel("Relative ln portfolio value")
        plt.show()
