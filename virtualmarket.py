""" 
Modules for setting up a virtual market and test portfolio strategies
"""

import sys

import numpy
from matplotlib import pyplot

import stockdata

dir, extension = "/DATA/lsiemens/Data/stocks/", ".us.txt"

class NoDataError(IOError):
    pass

class InvalidTickerError(ValueError):
    pass

class TickerNotTradingError(ValueError):
    pass

class Market:
    """ 
    A virtual market place.
    """
    _time_ = 0
    _tickers_ = 1
    _properties_ = 2
    _is_trading_ = 3
    _len_state_ = 4

    def __init__(self, tickers, clients, use_common_time=False):
        """ 
        list tickers that are avaliable in the virtual market

        if not use_common_time start at earlyest data point
        """
        self._tickers = tickers
        self._clients = clients
        self._raw_data = [None]*len(self._tickers)
        self._indices = [0]*len(self._tickers)
        self._current_time = None
        self._aproximate_len = None
        self.state = (None, None, None, None) # time, [tickers], [{properties}], [is trading]

        for i in range(len(self._clients)):
            self._clients[i]._market = self

        files = [(dir + ticker + extension, ticker) for ticker in tickers]
        time_list = []

        for i, (path, ticker) in enumerate(files):
            self._raw_data[i] = stockdata.csv(path, ticker)
            self._raw_data[i].data["time"] = numpy.asarray(self._raw_data[i].data["time"]//(24*60*60), dtype=int)
            time_list.append(self._raw_data[i].data["time"])

            progress_bar(i, len(files), "Loading: ")
        progress_bar_end("Loading: ")

        self._aproximate_len = max([times[-1] for times in time_list])

        time_list = [times[0] for times in time_list]
        if use_common_time:
            self._current_time = max(time_list)
        else:
            self._current_time = min(time_list)

        self._aproximate_len -= self._current_time

        self._update_state(increment=False)

    def _update_state(self, increment=True, count=0):
        state = [None]*self._len_state_
        dead = 0
        for i in range(len(self._tickers)):
            if self._indices[i] == len(self._raw_data[i].data["time"]) - 1:
                dead += 1
        if dead == len(self._tickers):
            raise NoDataError("no more data to read.")

        if increment:
            self._current_time += 1

        tickers, data, is_trading = [], [], []
        for i in range(len(self._tickers)):
            while True:
                if self._current_time > self._raw_data[i].data["time"][self._indices[i]]:
                    if self._indices[i] < len(self._raw_data[i].data["time"]) - 1:
                        self._indices[i] += 1
                    else:
                        break
                else:
                    break

            time = self._raw_data[i].data["time"][self._indices[i]]

            if self._current_time != time:
                if self._indices[i] == 0:
                    # if the current time is before the first datapoint dont add data to state
                    continue
                tickers.append(self._tickers[i])
                element_data = {}
                for key in self._raw_data[i].data.keys():
                    element_data[key] = self._raw_data[i].data[key][self._indices[i]]
                data.append(element_data)
                is_trading.append(False)
            else:
                tickers.append(self._tickers[i])
                element_data = {}
                for key in self._raw_data[i].data.keys():
                    element_data[key] = self._raw_data[i].data[key][self._indices[i]]
                data.append(element_data)
                is_trading.append(True)

        state[self._time_] = self._current_time
        state[self._tickers_] = tickers
        state[self._properties_] = data
        state[self._is_trading_] = is_trading
        self.state = tuple(state)

        if any(is_trading):
            return count + 1
        else:
            return self._update_state(count=count) + 1

    def start(self):
        i = 0
        while True:
            for j in range(len(self._clients)):
                self._clients[j].start_day()

            try:
                i += self._update_state()
            except NoDataError:
                progress_bar_end("Computing: ")

                for j in range(len(self._clients)):
                    self._clients[j].last_report()
                break

            progress_bar(i, self._aproximate_len, "Computing: ")

    def buy(self, ticker, num, id):
        if ticker not in self.state[self._tickers_]:
            raise InvalidTickerError()
        ticker_index = self.state[self._tickers_].index(ticker)
        if not self.state[self._is_trading_]:
            raise TickerNotTradingError()

        num = min(num, int(numpy.sqrt(self.state[self._properties_][ticker_index]["volume"])))
        return num, self.state[self._properties_][ticker_index]["open"]

    def sell(self, ticker, num, id):
        if ticker not in self.state[self._tickers_]:
            raise InvalidTickerError()
        ticker_index = self.state[self._tickers_].index(ticker)
        if not self.state[self._is_trading_]:
            raise TickerNotTradingError()

        num = min(num, int(numpy.sqrt(self.state[self._properties_][ticker_index]["volume"])))
        return num, self.state[self._properties_][ticker_index]["open"]

class Client:
    """ 
    A virtual client.
    """
    _class_id = 1

    def __init__(self, balance):
        # this should only be modified by the market object
        self._market = None
        self._id = Client._class_id
        self._balance = balance
        self._total_fees = 0.0
        self._total_taxes = 0.0
        self._holdings = {} #{"ticker":[(price, number)]} each list used as a FIFO que
        self.state = None

        Client._class_id += 1

    def __str__(self):
        text = "id: " + str(self._id) + " balance: " + str(round(self._balance, 2)) + " holdings: [" + " ".join([ticker + ": " + str(self.holdings(ticker)[1]) + " at " + str(round(self.holdings(ticker)[0], 2)) for ticker in self._holdings.keys()]) + "]"
        return text

    def _fees(self, num, price):
        return 0.0


    # this has just been prototyped ---------------------------#
    def quote(self, ticker, num=1, include_fees=False):
        print("quote() is just a prototype")
        price = self._market.quote(ticker):
        if not include_fees:
            return price*num
        fees = self._fees(num, price)
        return price*num - fees

    def buy(self, ticker, num):
#        print("try to buy", num, " stocks of", ticker)
        num, price = self._market.buy(ticker, num, self._id)
        if num == 0:
            return
#        print("market validate buy num", num, "at price", price)
        fees = self._fees(num, price)
        self._balance -= price*num + fees
        if ticker not in self._holdings:
            self._holdings[ticker] = []
        self._holdings[ticker].append([price, num])

#    def sell(self, ticker, num):
#        num, price = self._market.sell(ticker, num, self._id)
#        if num == 0:
#            return
#        fees = self._fees(num, price)
#        taxes = 0.0 #calculate taxes
#        #remove old stocks from self._holdings que
#        self._balance += num*price - (fees + taxes)

    def holdings(self, ticker):
        """ 
        return mean price and number of stocks
        """
        if ticker not in self._holdings:
            return None, 0

        num, cost = sum([num for price, num in self._holdings[ticker]]), sum([num*price for price, num in self._holdings[ticker]])
        return cost/float(num), num

    def start_day(self):
        self.state = self._market.state

    def last_report(self):
        print(self)

class Trivial_BuyHold(Client):
    def __init__(self, balance, buy_tickers, buy_num):
        super().__init__(balance=balance)
        self.buy_tickers = buy_tickers
        self.buy_num = buy_num

    def start_day(self):
        super().start_day()
        for i in range(len(self.buy_tickers)):
            mean_price, num = self.holdings(self.buy_tickers[i])
            if num < self.buy_num[i]:
                try:
                    self.buy(self.buy_tickers[i], (self.buy_num[i] - num)//3)
                except InvalidTickerError:
                    pass

def progress_bar(i, total, msg=""):
    if total >= 1000:
        if i%(total//1000) == 0:
            sys.stdout.write("\r" + msg + "%f%%" % (100.0*i/total))
            sys.stdout.flush()
    elif total >= 100:
        if i%(total//100) == 0:
            sys.stdout.write("\r" + msg + "%f%%" % (100.0*i/total))
            sys.stdout.flush()
    elif total >= 10:
        if i%(total//10) == 0:
            sys.stdout.write("\r" + msg + "%f%%" % (100.0*i/total))
            sys.stdout.flush()

def progress_bar_end(msg=""):
    sys.stdout.write("\r" + msg + "%f%%" % (100.0))
    sys.stdout.write("\n")
    sys.stdout.flush()
