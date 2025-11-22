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
    EVector_mode = "rogers-satchell"

    def __init__(self, fname, length, interval):
        self.fname = fname
        self.length = length
        self.interval = interval
        
        data = np.genfromtxt(self.fname, delimiter=",", dtype=None,
                             skip_header=1, usecols=(0, 1, 3), unpack=True)
        tickers, quantities, currencies = data
        
        self.width = len(tickers)
        self.tickers = [str(ticker).strip() for ticker in tickers]
        currencies = [str(currency).strip() for currency in currencies]
        self.quantities = quantities

        self.date = datetime.datetime.today().strftime("%Y-%m-%d")
        self._client = dataQuest.QuestradeClient()

        # Cached quantities
        self.logprice = None
        self.logerror = None
        self.weights = None
        
        self._load()

    def _load(self):
        """Load parameters derived from market data

        This can be slow so this information is cached locally.
        """

        cache_key = self._cache_key()
        cache_name = "./cache/" + str(self._cache_key()) + ".pkl"

        if (os.path.isfile(cache_name)):
            self._unpickle(cache_name)
            return
        
        self.logprice = np.empty(shape=(self.width, self.length))
        self.logerror = np.empty(shape=(self.width, self.length))
        for i in range(self.width):
            symbol = self._client.find_symbol(self.tickers[i])
            
            candles = self._client.get_n_candles(symbol, self.length, self.interval)
            data = vectors.makeEVector(candles, self.EVector_mode)
            self.logprice[i, :] = data.vector[:, 0]
            self.logerror[i, :] = data.vector[:, 1]
      
        self.weights = np.empty(self.width)
        for i in range(self.width):
            self.weights[i] = self.quantities[i]*np.exp(self.logprice[i, -1])
        self.weights = self.weights/np.sum(self.weights)

        self._pickle(cache_name)

    def _unpickle(self, fname):
      with open(fname, "rb") as fin:
          data = pickle.load(fin)
      self.logprice, self.logerror, self.weights = data
      print("Load data from cache")
      
    def _pickle(self, fname):
        data = (self.logprice, self.logerror, self.weights)
        with open(fname, "wb") as fout:
            pickle.dump(data, fout, protocol=pickle.HIGHEST_PROTOCOL)
        print("Save data to cache")

    def _cache_key(self):
        quantities = tuple([int(100*quantity) for quantity in self.quantities])
        payload = {
            "length": self.length, "interval": self.interval, "date": self.date,
            "tickers": self.tickers, "quantities": quantities}

        data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.shake_256(data).hexdigest(8)

    def plot_market(self):
        for i in range(self.width):
            plt.plot(np.exp(self.logprice[i]), label=self.tickers[i])
        plt.legend()
        plt.show()

    def plot_portfolio(self):
        index = np.sum(self.weights[:, None]*np.exp(self.logprice), axis=0)
        index = index/np.nansum(self.weights*np.exp(self.logprice[:, -1]))
        plt.plot(index, "k-", label="Portfolio")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    portfolio = Portfolio("./holding.csv", 30, "OneDay")
    portfolio.plot_market()
    portfolio.plot_portfolio()
