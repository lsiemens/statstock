import time
import sys

import numpy
from matplotlib import pyplot

import stockdata
import stat_model

analysis_simulation_rato = 0.5

dir = "/DATA/lsiemens/Data/stocks/"
extension = ".us.txt"

#files = stockdata.list_stock_files(dir, extension)
files = [(dir + "spy" + extension, "spy")]
path, ticker = files[0]

#for i, (path, ticker) in enumerate(files):
try:
    stock_data = stockdata.csv(path, ticker)
except OSError:
#        print("header error: " + ticker)
    exit()
#    continue
time = stock_data.data["time"]
#    data_open = stock_data.data["open"]
data_open = stock_data.price_to_relative(stock_data.data["open"])
estimator = stat_model.StatEstimate(20, alpha=0.05)

try:
    time, mean, std, error, data_blocks = estimator.computeStats(time, data_open)
except ValueError:
#        print("too little data error: " + ticker)
    exit()
#    continue

model = stat_model.trivial(plot=0, verbose=False)
#    model = stat_model.random(estimator.n_points, plot=0, verbose=False)
ptime, pmean, pstd, perror = model.predictStats(time, mean, std, error)

pmean=pmean[:-1]
pstd=pstd[:-1]
data_blocks = data_blocks[-1, 1:]
split = int(analysis_simulation_rato*len(pmean))

pmean = pmean[split:]
pstd = pstd[split:]
price = data_blocks[split - 1:]
#inital_price, price = price[0], price[1:]

n = estimator.n_points
median = pmean**(2*n)/numpy.sqrt((pmean**2 + pstd**2)**n)


print((median - 1) - 10*price[:-1])
print(price[1:]/price[:-1])



#if len(files) >= 1000:
#    if i%(len(files)//1000) == 0:
#        sys.stdout.write("\r%f%%" % (100.0*i/len(files)))
#        sys.stdout.flush()
#elif len(files) >= 100:
#    if i%(len(files)//100) == 0:
#        sys.stdout.write("\r%f%%" % (100.0*i/len(files)))
#        sys.stdout.flush()
