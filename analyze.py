import time
import sys

import numpy
from matplotlib import pyplot

import data
import stat_model

dir = "/DATA/lsiemens/Data/stocks/"
extension = ".us.txt"

files = data.list_stock_files(dir, extension)

p_mean_all = []
p_std_all = []

renorm_mean_all, renorm_std_all = [], []
for i, (path, ticker) in enumerate(files):
    try:
        stock_data = data.csv(path, ticker)
    except OSError:
#        print("header error: " + ticker)
        continue
    time = stock_data.data["time"]
#    data_open = stock_data.data["open"]
    data_open = stock_data.price_to_relative(stock_data.data["open"])
    estimator = stat_model.StatEstimate(60, alpha=0.05)

    try:
        time, mean, std, error, data_blocks = estimator.computeStats(time, data_open)
    except ValueError:
#        print("too little data error: " + ticker)
        continue

    model = stat_model.trivial(plot=0, verbose=False)
#    model = stat_model.random(estimator.n_points, plot=0, verbose=False)
    ptime, pmean, pstd, perror = model.predictStats(time, mean, std, error)
    data_blocks = (data_blocks[:, 1:] - pmean[:-1])/pstd[:-1]
    renorm_mean = numpy.mean(data_blocks, axis=1)
    renorm_std = numpy.std(data_blocks, axis=1, ddof=1)
    renorm_mean_all = renorm_mean_all + list(renorm_mean)
    renorm_std_all = renorm_std_all + list(renorm_std)

    p_value_mean, p_value_std, prediction = model.evaluateModel(time, mean, std, error)
    p_mean_all.append(p_value_mean)
    p_std_all.append(p_value_std)
#    print("prob of more extream mean: " + str(p_value_mean))
#    print("prob of more extream std: " + str(p_value_std))

    if len(files) >= 1000:
        if i%(len(files)//1000) == 0:
            sys.stdout.write("\r%f%%" % (100.0*i/len(files)))
            sys.stdout.flush()
    elif len(files) >= 100:
        if i%(len(files)//100) == 0:
            sys.stdout.write("\r%f%%" % (100.0*i/len(files)))
            sys.stdout.flush()

renorm_mean_all = numpy.array(renorm_mean_all)
renorm_mean_all = renorm_mean_all[~numpy.isnan(renorm_mean_all)]
renorm_mean_all = renorm_mean_all[numpy.abs(renorm_mean_all) < 1e300]
m = numpy.mean(renorm_mean_all)
s = numpy.std(renorm_mean_all)
print()
renorm_mean_all = renorm_mean_all[numpy.abs(renorm_mean_all) < m + 3*s]
m = numpy.mean(renorm_mean_all)
s = numpy.std(renorm_mean_all)
print(m)
print(s)
x = numpy.linspace(m - 4*s, m + 4*s, 1000)
y = numpy.exp(-(x - m)**2/(2*s**2))/(s*numpy.sqrt(2*numpy.pi))
pyplot.plot(x, y)
z = renorm_mean_all[abs(renorm_mean_all) < m + 4*s]
pyplot.hist(z, bins=int(numpy.sqrt(len(z))), density=True)
pyplot.show()

renorm_std_all = numpy.array(renorm_std_all)
renorm_std_all = renorm_std_all[~numpy.isnan(renorm_std_all)]
renorm_std_all = renorm_std_all[numpy.abs(renorm_std_all) < 1e300]
renorm_std_all = renorm_std_all
m = numpy.mean(renorm_std_all)
s = numpy.std(renorm_std_all)
print()
renorm_std_all = renorm_std_all[numpy.abs(renorm_std_all) < m + 3*s]
m = numpy.mean(renorm_std_all)
s = numpy.std(renorm_std_all)
print(m)
print(s)
x = numpy.linspace(m - 4*s, m + 4*s, 1000)
y = numpy.exp(-(x - m)**2/(2*s**2))/(s*numpy.sqrt(2*numpy.pi))
pyplot.plot(x, y)
z = renorm_std_all[abs(renorm_std_all) < m + 4*s]
pyplot.hist(z, bins=int(numpy.sqrt(len(z))), density=True)
pyplot.show()

print("\naverage prob of more extream mean: " + str(numpy.nanmean(p_mean_all)))
print("average prob of more extream std: " + str(numpy.nanmean(p_std_all)))
