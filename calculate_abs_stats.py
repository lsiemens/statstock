import time
import sys

import numpy
from matplotlib import pyplot

import stockdata
import stat_model

dir = "/DATA/lsiemens/Data/stocks/"
extension = ".us.txt"

etfs = []
with open("etf_all_tickers.txt", "r") as fin:
    etfs = fin.read().split("\n")

files = stockdata.list_stock_files(dir, extension)
#files = [("/DATA/lsiemens/Data/stocks/" + ticker + ".us.txt", ticker) for ticker in etfs]

#num_days = [20, 120, 240, 360, 480, 600]
num_days = range(1, 20*12*3, 3)
all_data = [[]]*len(num_days)
for i, (path, ticker) in enumerate(files):
    try:
        stock_data = stockdata.csv(path, ticker)
    except OSError:
        print("header error: " + ticker)
        continue

    time = stock_data.data["time"]
    data_open = stock_data.data["open"]
    for j, days in enumerate(num_days):
        data_rel = data_open[::days]
        data_rel = data_rel[1:]/data_rel[:-1]

        m, s = numpy.nanmean(data_rel), numpy.nanstd(data_rel)

        if False:
            print(ticker)
            pyplot.hist(data_rel - m, bins=int(numpy.sqrt(len(data_rel))))
            pyplot.show()

        all_data[j] = all_data[j] + list(data_rel[numpy.abs(data_rel - m) < 100*s])

    if len(files) >= 1000:
        if i%(len(files)//1000) == 0:
            sys.stdout.write("\r%f%%" % (100.0*i/len(files)))
            sys.stdout.flush()
    elif len(files) >= 100:
        if i%(len(files)//100) == 0:
            sys.stdout.write("\r%f%%" % (100.0*i/len(files)))
            sys.stdout.flush()

_m, _median, _s = [], [], []

print()
for i, days in enumerate(num_days):
#    print(" ")
#    print("days:", days)
    data_rel = numpy.array(all_data[i])
    m = numpy.mean(data_rel)
    s = numpy.std(data_rel)
    d1 = data_rel[numpy.logical_and(data_rel < m + 10*s, data_rel > m - 10*s)]
    m = numpy.mean(d1)
    median = numpy.median(d1)
    s = numpy.std(d1)
    _m.append(m)
    _median.append(median)
    _s.append(s)

#    mu = numpy.log(m/numpy.sqrt(1+s**2/m**2))
#    sigma_2 = numpy.log(1 + s**2/m**2)
#    print("mean", m)
#    print("median", median)
#    print("standard deviation", s)
#    print("error of the mean", s/numpy.sqrt(len(d1)))

#    x = numpy.linspace(max(0, m - 10*s), m + 10*s, 1000)
#    y = numpy.exp(-(x - m)**2/(2*s**2))/(s*numpy.sqrt(2*numpy.pi))
#    z = numpy.exp(-(numpy.log(x) - mu)**2/(2*sigma_2))/(x*numpy.sqrt(2*numpy.pi*sigma_2))

#    pyplot.hist(d1, bins=int(numpy.sqrt(len(d1))), density=True)
#    pyplot.plot(x, y)
#    pyplot.plot(x, z)
#    pyplot.show()

pyplot.plot(num_days, _m, "r")
pyplot.plot(num_days, _median, "k")
pyplot.show()
pyplot.plot(num_days, _s)
pyplot.show()
