import os
import time
import datetime

import numpy

class StockData:
    """ 
    Base class for loading historical stock data.
    """
    
    def __init__(self, fname, ticker):
        """ 
        Load historical finance data.

        Parameters
        ----------
        fname : string
            The file to load.
        ticker : string
            The stock ticker of the data to load.
        """

        self.path = os.path.abspath(fname)
        self.ticker = ticker
        self.data = {}

    def price_to_relative(self, price):
        """ 
        Get relative change in price.
        """
        
        data = numpy.empty(shape=price.shape)
        data[1:] = price[1:]/price[:-1]
        data[0] = 1.0
        return data

    def price_to_normalized(self, time, price):
        """ 
        Get relative change in price, normalized to the average time step
        and scaled such that the total relative change is uneffected.
        
        """
        
        data = numpy.empty(shape=price.shape)
        dt = time[1:] - time[:-1]
        data[1:] = numpy.power(price[1:]/price[:-1], dt.mean()/dt)
        data[0] = 1.0
        normalization = (price[-1]/price[0])/(numpy.prod(data))
        data[1:] = data[1:]*numpy.power(normalization, 1.0/float(len(data[1:])))
        return data

class Yahoo(StockData):
    """ 
    Load .csv files of historical stock data from Yahoo finance.

    Attributes
    ----------
    path : string
        Path to the loaded file.
    ticker : string
        The stock ticker of the loaded data.
    data : dictonary
        A dictonary containing the loaded data. The dictonary keys are;
        "time", "open", "high", "low", "close", "adj_close", and "volume".
        The time is reported as a unix timestamp.
    """

    _header = "Date,Open,High,Low,Close,Adj Close,Volume"

    def __init__(self, fname, ticker):
        """ 
        Load historical finance data.

        Parameters
        ----------
        fname : string
            The file to load.
        ticker : string
            The stock ticker of the data to load.
        """

        super().__init__(fname, ticker)
        
        data = None
        with open(self.path, "r") as fin:
            data = fin.read()

        data = data.strip().split("\n")
        header, data = data[0], data[1:]
        if header != self._header:
            raise IOError("(" + self.path + "): Malformed header.")
        
        #initalize empty numpy arrays
        self.data["time"] = numpy.empty((len(data),))
        self.data["open"] = numpy.empty((len(data),))
        self.data["high"] = numpy.empty((len(data),))
        self.data["low"] = numpy.empty((len(data),))
        self.data["close"] = numpy.empty((len(data),))
        self.data["adj_close"] = numpy.empty((len(data),))
        self.data["volume"] = numpy.empty((len(data),))
        
        for i, row in enumerate(data):
            #As of 12/21/2017 Yahoo finiance csv files have the format
            #date, open, high, low, close, adj_close, volume
            row = row.split(",")
            if len(row) != len(self._header.split(",")):
                raise IOError("(" + self.path + ") line " + str(i + 2) + ": wrong number of fileds.")
            
            try:
                self.data["time"][i] = time.mktime(datetime.datetime.strptime(row[0], "%Y-%m-%d").timetuple())
            except ValueError as exception:
                raise IOError("(" + self.path + ") line " + str(i + 2) + ": " + str(exception))
            self.data["open"][i] = float(row[1])
            self.data["high"][i] = float(row[2])
            self.data["low"][i] = float(row[3])
            self.data["close"][i] = float(row[4])
            self.data["adj_close"][i] = float(row[5])
            self.data["volume"][i] = int(row[6])
