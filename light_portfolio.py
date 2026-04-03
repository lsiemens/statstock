import numpy as np
from matplotlib import pyplot as plt
import datetime
import hashlib
import json
import pickle
import os

import dataQuest

class LightPortfolio:
    """Define a portfolio from a csv file of holdings.
    """

    USDtoCAD = None

    def __init__(self, fname):
        self.fname = fname

        self.width = None
        self.tickers = None
        self.n_shares = None

        # Cached vectors
        self.price = None

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

        self.price = np.empty(self.width)
        for i in range(self.width):
            print(f"Load quotes: {self.tickers[i]}")
            symbol = client.find_symbol(self.tickers[i])

            quote = client.get_quote(symbol)
            self.price[i] = self._toCAD(quote)

        self._pickle(cache_name)

    def _toCAD(self, quote):
        price, (_, ticker, currency) = quote
        match currency:
            case "CAD":
                pass
            case "USD":
                if self.USDtoCAD is None:
                    LightPortfolio.USDtoCAD = float(input("Exchange rate USD to CAD:"))
                price = price*self.USDtoCAD
            case _:
                raise NotImplementedError(f"The currency {currency} has not been implemented.")
        return price

    def _unpickle(self, fname):
        with open(fname, "rb") as fin:
            data = pickle.load(fin)
        self.price = data
        print("Load data from cache")

    def _pickle(self, fname):
        data = self.price
        with open(fname, "wb") as fout:
            pickle.dump(data, fout, protocol=pickle.HIGHEST_PROTOCOL)
        print("Save data to cache\n")

    def _cache_key(self):
        """Generate a hash for the portfolio
        """

        date = datetime.datetime.today().strftime("%Y-%m-%d")
        n_shares = tuple([int(100*n_share) for n_share in self.n_shares])

        payload = {"date": date, "tickers": self.tickers,
                   "100*shares": n_shares}

        data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.shake_256(data).hexdigest(8)

    def print_portfolio(self):
        for i in range(self.width):
            print(f"Ticker \"{self.tickers[i]}\": {self.n_shares[i]} shares at ${self.price[i]:.2f} in CAD")
            print(f"\tTotal value: ${self.n_shares[i]*self.price[i]:.2f} in CAD")

if __name__ == "__main__":
    portfolio = LightPortfolio("./all_holdings.csv")
    portfolio.print_portfolio()
