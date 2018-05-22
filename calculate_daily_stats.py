""" 
all data mean log
1.0000185755471513
0.033875014224974805
8.121575667990694e-06

all data mean
1.0005225698056357
0.035813079882944396
8.586568585847744e-06

1000 data mean log
1.0000267017874378
0.03268133195536982
2.27056536719153e-05

1000 data mean
1.0005099310504377
0.042647990177942595
2.9631124554019704e-05

spy data mean log
1.0002812469945537
0.011681365505899816
0.0002064670611101673

spy data mean
1.0003493202074054
0.011651402241253902
0.00020593746316294244

"""

import time
import sys

import numpy
from matplotlib import pyplot

import data
import stat_model

dir = "/DATA/lsiemens/Data/stocks/"
extension = ".us.txt"

#files = data.list_stock_files(dir, extension)
files = [("/DATA/lsiemens/Data/stocks/glw.us.txt", "dia")]

all_data = []
for i, (path, ticker) in enumerate(files):
    try:
        stock_data = data.csv(path, ticker)
    except OSError:
        print("header error: " + ticker)
        continue
    print("init abs", stock_data.data["open"][0])
    print("fin abs", stock_data.data["open"][-1])
    print("num elem", len(stock_data.data["open"]))
    time = stock_data.data["time"]
##    data_open = stock_data.price_to_relative(stock_data.data["open"])
    data_open = numpy.log(stock_data.price_to_relative(stock_data.data["open"]))
#    print("exp tot rel dif", numpy.exp(numpy.sum(data_open)))

    m, s = numpy.nanmean(data_open), numpy.nanstd(data_open)

#    a = numpy.count_nonzero(data_open > m + 10*s)
#    if a != 0:
#        print(ticker)
#        f, (ax1, ax2) = pyplot.subplots(1, 2, sharex=True)
#        ax1.plot(data_open)
#        ax2.plot(stock_data.data["open"])
#        pyplot.show()

#    all_data = all_data + list(data_open[data_open < 1e308])
    all_data = all_data + list(data_open[data_open < m + 10*s])

    if len(files) >= 1000:
        if i%(len(files)//1000) == 0:
            sys.stdout.write("\r%f%%" % (100.0*i/len(files)))
            sys.stdout.flush()
    elif len(files) >= 100:
        if i%(len(files)//100) == 0:
            sys.stdout.write("\r%f%%" % (100.0*i/len(files)))
            sys.stdout.flush()

all_data = numpy.array(all_data)
m = numpy.mean(all_data)
s = numpy.std(all_data)
print()
##print(m)
##print(s)
##print(s/numpy.sqrt(len(all_data)))
print(numpy.exp(m))
print(numpy.exp(m)*s)
print(numpy.exp(m)*s/numpy.sqrt(len(all_data)))

d1 = all_data[numpy.logical_and(all_data < m + s, all_data > m - s)]
x = numpy.linspace(m - s, m + s, 1000)
y = numpy.exp(-(x - m)**2/(2*s**2))/(s*numpy.sqrt(2*numpy.pi))

pyplot.hist(d1, bins=int(numpy.sqrt(len(d1))), density=True)
pyplot.plot(x, y)
pyplot.show()

pyplot.hist(all_data, bins=int(numpy.sqrt(len(all_data))), density=True)
pyplot.show()
