"Valid"

import numpy
import scipy.stats
import scipy.special
from matplotlib import pyplot

lognorm = scipy.stats.lognorm

mean_rel = 1.002
std_rel = 0.04
n = 450
m = 10000
x = numpy.linspace(1, n, n)

p = scipy.special.erf(1/numpy.sqrt(2))
interval = (0.5 - p/2, 0.5 + p/2)

normal = scipy.stats.norm(loc=mean_rel, scale=std_rel)

data = []
for i in range(m):
    relative_value = normal.rvs(n)
    absolute_value = [numpy.prod(relative_value[:i + 1]) for i in range(len(relative_value))]
    data.append(absolute_value)
data = numpy.array(data)

mean_abs = mean_rel**x
std_abs = numpy.sqrt((mean_rel**2 + std_rel**2)**x - mean_rel**(2*x))

mean_lognorm = numpy.log(mean_rel**(2*x)/numpy.sqrt((mean_rel**2 + std_rel**2)**x))
std_lognorm = numpy.sqrt(numpy.log((mean_rel**2 + std_rel**2)**x/mean_rel**(2*x)))
mode_lognorm = mean_rel**(4*x)/numpy.power(mean_rel**2 + std_rel**2, 3*x/2)

for i in range(0, n, n//2):
    print(i)
    mean, std = mean_abs[i], std_abs[i]
    ln_mean, ln_std, ln_mode = mean_lognorm[i], std_lognorm[i], mode_lognorm[i]

    z = numpy.linspace(max(0, mean - 5*std), mean + 5*std, 1000)[1:]
    y = numpy.exp(-(z - mean)**2/(2*std**2))/(std*numpy.sqrt(2*numpy.pi))
    pyplot.plot(z, y, "r-")

    lognorm_err = lognorm(s=ln_std, loc=0.0, scale=numpy.exp(ln_mean))
    w = numpy.exp(-(numpy.log(z) - ln_mean)**2/(2*ln_std**2))/(z*ln_std*numpy.sqrt(2*numpy.pi))
    pyplot.plot(z, w, "g-")

    pyplot.hist(data[:, i], bins=int(numpy.sqrt(m)), density=True, histtype="step", color="k")
    pyplot.hist(data[:, i], bins=int(numpy.sqrt(m)), density=True, alpha=0.3, color="k")
    pyplot.axvline(x=mean + std, c="r", linestyle="-.")
    pyplot.axvline(x=max(0, mean - std), c="r", linestyle="-.")
    pyplot.axvline(x=mean, c="k", linestyle="-")
    pyplot.axvline(x=ln_mode, c="b", linestyle="-.")
    pyplot.axvline(x=lognorm_err.ppf(0.5), c="b", linestyle="--")
    pyplot.axvline(x=lognorm_err.ppf(interval[0]), c="g", linestyle="-.")
    pyplot.axvline(x=lognorm_err.ppf(interval[1]), c="g", linestyle="-.")
    pyplot.show()

pyplot.plot(x, mean_abs + std_abs, "r-.")
pyplot.plot(x, numpy.maximum(0, mean_abs - std_abs), "r-.")
pyplot.plot(x, mean_abs, "k-")
pyplot.plot(x, mode_lognorm, "b-.")
pyplot.plot(x, lognorm.ppf(0.5, s=std_lognorm, loc=0.0, scale=numpy.exp(mean_lognorm)), "b--")
pyplot.plot(x, lognorm.ppf(interval[0], s=std_lognorm, loc=0.0, scale=numpy.exp(mean_lognorm)), "g-.")
pyplot.plot(x, lognorm.ppf(interval[1], s=std_lognorm, loc=0.0, scale=numpy.exp(mean_lognorm)), "g-.")

data = data.flatten()
x = numpy.array([x]*m).flatten()
pyplot.hexbin(x, data, bins=50, extent=(0, n, 0, mean_abs[-1] + 2*std_abs[-1]))
pyplot.show()
