""" 
Valid
"""

import numpy
import scipy.stats
import scipy.special
from matplotlib import pyplot

mean_rel = 1.00237
std_rel =  0.04142
norm_n = 90
n = 100000

interval = scipy.special.erf(1/numpy.sqrt(2))

# expected mean and std of product distribution
mean = mean_rel**norm_n
std = numpy.sqrt((mean_rel**2 + std_rel**2)**norm_n - mean_rel**(2*norm_n))
print("# target dist# mean: " + str(mean) + " std: " + str(std))

# "\mu" and "\sigma" of lognormal distribution nedded to match target mean and std
mean_lognorm = numpy.log(mean_rel**(2*norm_n)/numpy.power(std_rel**2 + mean_rel**2, norm_n/2.0))
std_lognorm = numpy.sqrt(numpy.log(numpy.power(std_rel**2 + mean_rel**2, norm_n)/numpy.power(mean_rel, 2*norm_n)))
mode_lognorm = mean_rel**(4*norm_n)/numpy.power(std_rel**2 + mean_rel**2, 3*norm_n/2.0)
print("# lognorm settings# mean: " + str(mean_lognorm) + " std: " + str(std_lognorm))

# define scipy distributions
normal = scipy.stats.norm(loc=mean_rel, scale=std_rel)
lognormal = scipy.stats.lognorm(s=std_lognorm, loc=0.0, scale=numpy.exp(mean_lognorm))

# get product distribution samples
relative_value = normal.rvs((n, norm_n))
absolute_value = numpy.prod(relative_value, axis=1)
print("# product dist# mean: " + str(numpy.mean(absolute_value)) + " std: " + str(numpy.std(absolute_value)))

# get lognormal distribution samples
lognorm_value = lognormal.rvs(n)
print("# lognorm dist# mean: " + str(numpy.mean(lognorm_value)) + " std: " + str(numpy.std(lognorm_value)))

x = numpy.linspace(max(0, mean - 5*std), mean + 5*std, 1000)[1:]

# normal pdf equation
norm_pdf = numpy.exp(-(x - mean)**2/(2*std**2))/(std*numpy.sqrt(2*numpy.pi))
# lognormal pdf equation
lognorm_pdf = numpy.exp(-(numpy.log(x) - mean_lognorm)**2/(2*std_lognorm**2))/(x*std_lognorm*numpy.sqrt(2*numpy.pi))

pyplot.plot(x, norm_pdf, "r-", label="normal distribution pdf")
pyplot.axvline(x=mean, color="k", linestyle="-", label="mean")
pyplot.axvline(x=mean + std, color="r", linestyle="-.", label="normal distribution confidence interval")
pyplot.axvline(x=mean - std, color="r", linestyle="-.")
pyplot.plot(x, lognorm_pdf, "g-", label="lognorm distribution pdf")

range = (0.5 - interval/2, 0.5 + interval/2)
range = (lognormal.ppf(range[0]), lognormal.ppf(range[1]))
pyplot.axvline(x=range[0], color="g", linestyle="-.", label="lognormal distribution confidence interval")
pyplot.axvline(x=range[1], color="g", linestyle="-.")
pyplot.axvline(x=lognormal.ppf(0.5), color="b", linestyle="--", label="lognormal distribution median")
pyplot.axvline(x=mode_lognorm, color="b", linestyle="-.", label="lognormal distribution mode")

pyplot.hist(absolute_value, bins=100, density=True, histtype="step", color="k")
pyplot.hist(absolute_value, bins=100, density=True, alpha=0.3, color="k", label="product distribution data")
pyplot.hist(lognorm_value, bins=100, density=True, histtype="step", color="g")
pyplot.hist(lognorm_value, bins=100, density=True, alpha=0.3, color="g", label="lognormal distribution data")
pyplot.xlim(max(0, mean - 5*std), mean + 5*std)
pyplot.legend(loc=0)
pyplot.show()
