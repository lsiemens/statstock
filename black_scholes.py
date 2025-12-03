import numpy as np
from matplotlib import pyplot as plt

import dataQuest

def autocoef(data, max_offset):
    if (max_offset > len(data)//4):
        raise ValueError("length of data should be > 4*max_offset")
    data = np.asarray(data)
    coefficents = np.array([np.correlate(data[::i + 1], np.roll(data[::i + 1], 1))[0] for i in range(max_offset)])
    #coefficents = np.array([np.correlate(data, np.roll(data, i))[0] for i in range(max_offset)])
    return coefficents/coefficents[0]


class MCMC:
    def __init__(self, x_0, ln_f, sampler_sigma=1.0, burnin=100, stride=100,
                 log=False):
        self.x_0 = x_0
        self.ln_f = ln_f
        self.x = self.x_0
        self.ln_f_value = self.ln_f(self.x)
        self.sampler_sigma = sampler_sigma
        self.stride = stride
        self.log = log

        self.burnin = burnin

        self._values_passed = 0
        self._values_tried = 0

        self.rng = np.random.default_rng()

    def reset(self):
        self.x = self.x_0
        self.ln_f_value = self.ln_f(self.x)
        self._values_passed = 0
        self._values_tried = 0

    def generate_samples(self, n):
        n_to_burnin = np.max(self.burnin - self._values_tried, 0)
        _ = [self._generate_sample() for i in range(n_to_burnin)]

        samples = [self._generate_sample() for i in range(n*self.stride)]
        return np.array(samples[::self.stride])

    def _generate_sample(self):
        self._values_tried += 1
        x_new = None
        if self.log:
            x_new = self.rng.lognormal(np.log(self.x), self.sampler_sigma)
        else:
            x_new = self.rng.normal(self.x, self.sampler_sigma)
        ln_f_value_new = self.ln_f(x_new)

        test_fraction = self.rng.uniform(0, 1)
        if np.log(test_fraction) <= ln_f_value_new - self.ln_f_value:
            self._values_passed += 1
            self.x = x_new
            self.ln_f_value = ln_f_value_new
            return self.x
        return self.x

    def plot_function(self, x_range):
        xs = np.linspace(*x_range, 1000)
        plt.plot(xs, [self.ln_f(x) for x in xs])
        plt.title("Distribution function $f(x)$")
        plt.xlabel("$x$")
        plt.ylabel("$\\ln{f(x)}$")
        plt.show()

    def plot_burnin(self, samples):
        self.reset()
        x_value = [self._generate_sample() for i in range(samples)]
        plt.plot(np.arange(len(x_value)), x_value, label="value")
        plt.axvline(self.burnin, color="k", linestyle=":", label="burnin")
        plt.title("Show burnin")
        plt.xlabel("sample #")
        plt.ylabel("sampled x values")
        plt.legend()
        plt.show()

    def plot_correlation(self, x_samples, max_offset=100, samples=10):
        n_to_burnin = np.max(self.burnin - self._values_tried, 0)
        _ = [self._generate_sample() for i in range(n_to_burnin)]

        auto_correlation = np.empty((samples, max_offset))
        for i in range(samples):
            x_value = [self._generate_sample() for i in range(x_samples)]
            auto_correlation[i] = autocoef(x_value, max_offset)
        AC_mean = np.mean(auto_correlation, axis=0)
        AC_error = np.std(auto_correlation, axis=0)

        offsets = np.arange(max_offset)
        plt.plot(offsets, AC_mean, "k-", label=f"AC, efficency={self._values_passed/self._values_tried:.3f}")
        plt.fill_between(offsets, AC_mean - AC_error, AC_mean + AC_error,
                         alpha=0.5, label="AC error")
        plt.axvline(self.stride, color="k", linestyle=":", label="stride")
        plt.title("Autocorrelation")
        plt.xlabel("sample offset")
        plt.ylabel("Auto correlation coefficient")
        plt.legend()
        plt.show()

        plt.loglog(offsets, AC_mean, "k-", label=f"AC, efficency={self._values_passed/self._values_tried:.3f}")
        plt.fill_between(offsets, AC_mean - AC_error, AC_mean + AC_error,
                         alpha=0.5, label="AC error")
        plt.axvline(self.stride, color="k", linestyle=":", label="stride")
        plt.title("Autocorrelation")
        plt.xlabel("sample offset")
        plt.ylabel("Auto correlation coefficient")
        plt.legend()
        plt.show()


def ln_gaussian(observed, target):
    obs, obs_err = observed
    val, val_err = target
    sigma_2 = obs_err**2 + val_err**2
    a = np.log(np.sqrt(2*np.pi*sigma_2))
    if not np.isfinite(a):
        print(sigma_2, a)
    #print(sigma_2, np.log(np.sqrt(2*np.pi*sigma_2)))
    return -0.5*(obs - val)**2/sigma_2 - np.log(np.sqrt(2*np.pi*sigma_2))


def european_price(stock_price, ln_sigma, ln_risk_free_return, strike_price,
                   expiry_time, option_type, samples=1000, rng=None):
    """
    Parameters
    ----------
    stock_price : float
        current stock price
    ln_sigma : float
        annualized standard deviation of ln returns
    ln_risk_free_return : float
        annualized ln of the risk free return rate
    strike_price : float
        the target strike price
    expiry_time : float
        Time until expiry in weeks
    option_type : str
        use "call" for call options and "put" for put options
    samples : int
        Number of samples of the price evolution
    """

    if rng is None:
        rng = np.random.default_rng()

    mu = ln_risk_free_return/52
    sigma = ln_sigma/np.sqrt(52)

    ln_returns = rng.normal(mu*expiry_time, sigma*np.sqrt(expiry_time), samples)
    price_T = stock_price*np.exp(ln_returns)

    option_returns = None
    match option_type:
        case "call":
            option_returns = np.maximum(price_T - strike_price, 0)
        case "put":
            option_returns = np.maximum(strike_price - price_T, 0)

    discount = np.exp(-mu*expiry_time)
    option_price = discount*np.mean(option_returns)
    option_price_error = discount*np.std(option_returns)/np.sqrt(samples)
    return (option_price, option_price_error), discount*np.std(option_returns)


class search:
    def __init__(self, f, x_range, log=True):
        self.f = f
        self.x_range = x_range
        self.log = log
        self.rng = np.random.default_rng()

    def _search(self, samples=10, subsamples=2, depth=7):
        x_min, x_max = self.x_range[0], self.x_range[1]
        x = None

        for _ in range(depth):
            if self.log:
                x = np.exp(self.rng.uniform(np.log(x_min), np.log(x_max), samples))
            else:
                x = self.rng.uniform(x_min, x_max, samples)

            f_values = None
            f_values = [[self.f(x_value) for _ in range(subsamples)] for x_value in x]
            f_values = np.array(f_values)
            f_values = np.mean(f_values, axis=1)

            index = np.argmax(f_values)

            x_center = x[index]
            if self.log:
                x_min = np.sqrt(x_min*x_center)
                x_max = np.sqrt(x_center*x_max)
            else:
                x_min = (x_min + x_center)/2
                x_max = (x_center + x_max)/2
        if self.log:
            return np.sqrt(x_min*x_max)
        else:
            return (x_min + x_max)/2

    def search(self, samples=10):
        x = [self._search() for _ in range(samples)]
        return np.mean(x), np.std(x)/np.sqrt(samples)


if __name__ == "__main__":
    client = dataQuest.QuestradeClient()
    ticker = "TSLA"

    stock_price, _ = client.get_quote(ticker)
    options = client.options(ticker, stock_price, 10)
    options = [option for option in options if (option[0] < 60)]
    options = [option for option in options if (option[0] > 30)]
    options = [option for option in options if (option[1] < 7)]
    options = [option for option in options if not np.isclose(option[3], option[4])]
    print(options)

    def f(ln_sigma):
        value = 0
        value_err = 0
        for option in options:
            days_to_expiry, last_trade_day, strike, high, low, _, open_intrest, option_type = option
            T = days_to_expiry/7
            option_price = ((high + low)/2, (high - low)/2)
            modle_option_price, _ = european_price(stock_price, ln_sigma, 0.038,
                                                   strike, T, option_type,
                                                   samples=100)
            value += ln_gaussian(option_price, modle_option_price)
        return value

    A = search(f, (0.01, 1))
    print(A.search())
    print(A.search())
    print(A.search())
