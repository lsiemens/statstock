import numpy as np
from matplotlib import pyplot as plt
import datetime
import hashlib
import json
import pickle
import os

import dataQuest
import vectors


class Portfolio:
    """Define a portfolio from a csv file of holdings.
    """

    EVector_mode = "rogers-satchell"

    def __init__(self, fname, length, interval):
        self.fname = fname
        self.length = length
        self.interval = interval

        self.width = None
        self.tickers = None
        self.n_shares = None

        # Cached vectors
        self.logprice = None
        self.logerror = None
        self.options = None

        data = np.genfromtxt(self.fname, delimiter=",", dtype=None,
                             encoding=None, skip_header=1, usecols=(0, 1),
                             unpack=True)
        tickers, n_shares = data

        self.width = len(tickers)
        self.tickers = [str(ticker).strip() for ticker in tickers]
        self.n_shares = n_shares

        self._load()

    def _load(self):
        """Load parameters derived from market data

        This can be slow when getting the data from questrade so this
        information is cached locally.
        """

        cache_name = "./cache/" + str(self._cache_key()) + ".pkl"

        if (os.path.isfile(cache_name)):
            self._unpickle(cache_name)
            return

        client = dataQuest.QuestradeClient()

        self.logprice = np.empty(shape=(self.width, self.length))
        self.logerror = np.empty(shape=(self.width, self.length))
        for i in range(self.width):
            print(f"Load candles: {self.tickers[i]}")
            symbol = client.find_symbol(self.tickers[i])

            data = client.get_n_candles(symbol, self.length, self.interval)
            evector = vectors.makeEVector(data, self.EVector_mode)
            self.logprice[i, :] = evector.vector[:, 0]
            self.logerror[i, :] = evector.vector[:, 1]

        self.options = []
        for i in range(self.width):
            ticker = self.tickers[i]

            print(f"Load options: {ticker}")
            price, symbol = client.get_quote(ticker)
            self.options.append(client.options(symbol, price, 10))

        self._pickle(cache_name)

    def _unpickle(self, fname):
        with open(fname, "rb") as fin:
            data = pickle.load(fin)
        self.logprice, self.logerror, self.options = data
        print("Load data from cache")

    def _pickle(self, fname):
        data = (self.logprice, self.logerror, self.options)
        with open(fname, "wb") as fout:
            pickle.dump(data, fout, protocol=pickle.HIGHEST_PROTOCOL)
        print("Save data to cache")

    def _cache_key(self):
        """Generate a hash for the portfolio
        """

        date = datetime.datetime.today().strftime("%Y-%m-%d")
        n_shares = tuple([int(100*n_share) for n_share in self.n_shares])

        payload = {"length": self.length, "interval": self.interval,
                   "date": date, "tickers": self.tickers,
                   "100*shares": n_shares}

        data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.shake_256(data).hexdigest(8)

    def plot_market(self):
        for i in range(self.width):
            logprice = self.logprice[i, :] - self.logprice[i, -1]
            logerror = self.logerror[i, :]

            plt.plot(np.exp(logprice))
            plt.fill_between(np.arange(len(logprice)),
                             np.exp(logprice + logerror),
                             np.exp(logprice - logerror), alpha=0.5, label=self.tickers[i])
        plt.title("Market")
        plt.xlabel(self.interval[3:])
        plt.ylabel("Relative value")
        plt.legend()
        plt.show()

    def plot_portfolio(self):
        logprice = self.logprice[:, :] - self.logprice[:, -1][:, None]
        index = np.sum(self.n_shares[:, None]*np.exp(logprice), axis=0)
        index_low = np.sum(self.n_shares[:, None]*np.exp(logprice - self.logerror), axis=0)
        index_high = np.sum(self.n_shares[:, None]*np.exp(logprice + self.logerror), axis=0)

        plt.plot(index, "k-", label="Portfolio $\\bar{p}$")
        plt.fill_between(np.arange(self.length), index_low, index_high,
                         color="k", alpha=0.5, label="Portfolio $\\bar{p} \\pm \\sigma$")
        plt.legend()
        plt.title("Portfolio")
        plt.xlabel(self.interval[3:])
        plt.ylabel("Relative value")
        plt.show()


if __name__ == "__main__":
    portfolio = Portfolio("./all_holdings.csv", 2000, "OneWeek")
    portfolio.plot_market()
    portfolio.plot_portfolio()
