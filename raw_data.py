import time
import datetime
import numpy

class Yahoo:
    def __init__(self, fname, ticker):
        self.fname = fname
        self.ticker = ticker
        self.data = {}
        
        data = None
        with open(self.fname, "r") as fin:
            data = fin.read()
        data = data.split("\n")[1:-1]
        
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
            self.data["time"][i] = time.mktime(datetime.datetime.strptime(row[0], "%Y-%m-%d").timetuple())
            self.data["open"][i] = float(row[1])
            self.data["high"][i] = float(row[2])
            self.data["low"][i] = float(row[3])
            self.data["close"][i] = float(row[4])
            self.data["adj_close"][i] = float(row[5])
            self.data["volume"][i] = int(row[6])
